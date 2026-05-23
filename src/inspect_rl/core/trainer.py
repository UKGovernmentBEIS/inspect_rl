"""Entry point for Inspect-RL training with GRPO.

Wires together:
- An Inspect Task (dataset + scorers) as the reward signal
- TRL's GRPOTrainer for policy gradient optimization
- The TRLVLLMProvider + rollout function for Inspect-owned generation
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import inspect_rl.core.trl_vllm_provider as _  # noqa: F401  # registers @modelapi("trl-vllm")
from inspect_ai import Task, eval as inspect_eval, task_with
from inspect_ai.dataset import MemoryDataset
from inspect_ai.model import GenerateConfig, get_model
from peft import PeftConfig
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from trl.trainer.grpo_trainer import RewardFunc

from inspect_rl.core import rollout as _rollout_mod
from inspect_rl.core.rollout import (
    INSPECT_EVAL_LOCK,
    inspect_dataset_to_hf,
    make_inspect_rollout_func,
    make_rank_gated_rollout_func,
)
from inspect_rl.util.display import (
    log_eval,
    log_off_policy_decision,
    log_run_start,
    log_run_summary,
    set_step_counter,
)
from inspect_rl.util.heartbeat import set_phase, start_heartbeat, stop_heartbeat
from inspect_rl.util.logging_config import capture_inspect_log, configure_logging
from inspect_rl.util.run_dir import create_run_dir
from inspect_rl.util import summary as _summary

logger = logging.getLogger(__name__)


def inspect_rl_train(
    task: Task,
    model: str,
    grpo_config: GRPOConfig,
    vllm_base_url: str = "http://localhost:8000",
    peft_config: PeftConfig | None = None,
    dataset_limit: int | None = None,
    eval_task: Task | None = None,
    eval_steps: int = 50,
    eval_limit: int = 50,
    resume_from: str | Path | None = None,
    max_resample_rounds: int = 0,
    off_policy_steps: int = 0,
) -> GRPOTrainer:
    """Train a model using GRPO with an Inspect Task as the reward signal.

    `off_policy_steps` defaults to 0 (synchronous rollouts). Setting >0 or -1
    enables the prefetcher (`FreshestPrefetchRolloutFunc` / `AutoCalibratingRolloutFunc`),
    which is currently unstable under multi-rank DDP — see
    journal/009_h200_smoke/002_8gpu_ddp_smoke.md (KeyError: 'inspect_scores').
    """
    logger.info("=== inspect_rl_train start: model=%s ===", model)
    _summary.reset_state()
    # Accelerate's launch sets env vars on every rank; PartialState reads them
    # on first instantiation. We use this to size the prefetcher cursor and
    # rank-gate side effects (vLLM health probe, run-dir creation, wandb).
    from accelerate import PartialState

    distributed_state = PartialState()
    world_size = distributed_state.num_processes
    is_main = distributed_state.is_main_process
    if is_main:
        logger.info("Checking vLLM server at %s", vllm_base_url)
        _check_vllm_server(vllm_base_url)
    _configure_wandb_defaults()

    checkpoint_path: str | None = None

    # Only rank 0 creates the timestamped run dir / runs resume IO; the path
    # (and resume step) is then broadcast so every rank writes into the same
    # location. Without this, distinct timestamps per rank would race.
    run_dir_payload: dict[str, Any] | None
    if is_main:
        if resume_from is not None:
            run_dir = Path(resume_from).resolve()
            if not run_dir.is_dir():
                raise FileNotFoundError(f"Resume run dir not found: {run_dir}")
            checkpoint_path, resume_step = _prepare_resume(run_dir)
            set_step_counter(resume_step)
            logger.info("Resuming from %s (step %d)", checkpoint_path, resume_step)
        else:
            logger.info("Creating run directory under %s", grpo_config.output_dir)
            run_dir = create_run_dir(
                base=grpo_config.output_dir,
                config={
                    "model": model,
                    "max_steps": grpo_config.max_steps,
                    "batch_size": grpo_config.per_device_train_batch_size,
                    "num_generations": grpo_config.num_generations,
                    "learning_rate": grpo_config.learning_rate,
                    "max_completion_length": grpo_config.max_completion_length,
                    "peft": bool(peft_config),
                    "dataset_limit": dataset_limit,
                    "eval_steps": eval_steps if eval_task else None,
                    "eval_limit": eval_limit if eval_task else None,
                    "world_size": world_size,
                },
            )
            resume_step = 0
        run_dir_payload = {
            "run_dir": str(run_dir),
            "checkpoint_path": checkpoint_path,
            "resume_step": resume_step,
        }
    else:
        run_dir_payload = None

    if world_size > 1:
        from accelerate.utils import broadcast_object_list

        obj_list = [run_dir_payload]
        broadcast_object_list(obj_list, from_process=0)
        run_dir_payload = obj_list[0]
        assert run_dir_payload is not None
        run_dir = Path(run_dir_payload["run_dir"])
        checkpoint_path = run_dir_payload["checkpoint_path"]
        if not is_main:
            set_step_counter(run_dir_payload["resume_step"])

    grpo_config.output_dir = str(run_dir / "checkpoints")
    # TRL's GRPOTrainer opens its own vLLM client for weight sync (NCCL
    # init_communicator + update_named_param) using grpo_config.vllm_server_host
    # /port. Examples hardcode localhost; override from vllm_base_url so remote
    # vLLM (multi-node) works without touching every example.
    _parsed = urlparse(vllm_base_url)
    if _parsed.hostname:
        grpo_config.vllm_server_host = _parsed.hostname
    if _parsed.port:
        grpo_config.vllm_server_port = _parsed.port
    eval_log_dir = str(run_dir / "eval_logs")
    logger.info("Run directory: %s", run_dir)
    configure_logging(log_file=run_dir / "train.log", is_main_process=is_main)
    if is_main:
        log_run_start(run_dir / "train.log")
        logger.info("Log file: %s", run_dir / "train.log")

    logger.info(
        "Building rollout function (eval_log_dir=%s, resample_rounds=%d, off_policy_steps=%d)",
        eval_log_dir,
        max_resample_rounds,
        off_policy_steps,
    )
    rollout_func = make_inspect_rollout_func(
        task,
        vllm_base_url=vllm_base_url,
        eval_log_dir=eval_log_dir,
        max_resample_rounds=max_resample_rounds,
    )

    scorer_names = _extract_scorer_names(task)
    reward_funcs = [_make_scores_reward_func(name) for name in scorer_names]
    logger.info("Scorers → reward funcs: %s", scorer_names)

    logger.info("Converting train dataset → HF (limit=%s)", dataset_limit)
    train_dataset = inspect_dataset_to_hf(task.dataset, limit=dataset_limit)
    logger.info("Train dataset size: %d", len(train_dataset))

    if off_policy_steps != 0:
        from inspect_rl.perf.prefetch import (
            AutoCalibratingRolloutFunc,
            FreshestPrefetchRolloutFunc,
        )

        if off_policy_steps < 0:
            rollout_func = AutoCalibratingRolloutFunc(
                inner_fn=rollout_func,
                hf_dataset=train_dataset,
                batch_size=grpo_config.per_device_train_batch_size,
                num_generations=grpo_config.num_generations,
                world_size=world_size,
            )
            logger.info(
                "Wrapped rollout_func with AutoCalibratingRolloutFunc "
                "(warmup then enable freshest prefetch if rollout > train)"
            )
        else:
            rollout_func = FreshestPrefetchRolloutFunc(
                inner_fn=rollout_func,
                hf_dataset=train_dataset,
                batch_size=grpo_config.per_device_train_batch_size,
                num_generations=grpo_config.num_generations,
                world_size=world_size,
            )
            logger.info("Wrapped rollout_func with FreshestPrefetchRolloutFunc")
            log_off_policy_decision("fixed", 1)
    else:
        log_off_policy_decision("fixed", 0)

    # Rank-gating must be the outermost wrapper: only rank 0 runs the prefetch
    # producer / synchronous eval; all other ranks block on broadcast each
    # step. With world_size==1 this is a no-op fast path.
    rollout_func = make_rank_gated_rollout_func(rollout_func)

    step_timer = _StepTimerCallback()
    callbacks: list[TrainerCallback] = [step_timer]
    if eval_task is not None:
        Path(eval_log_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            "Registering val-eval callback (every %d steps, %d samples)",
            eval_steps,
            eval_limit,
        )
        callbacks.append(
            _InspectEvalCallback(
                eval_task=eval_task,
                eval_steps=eval_steps,
                eval_limit=eval_limit,
                model_name=model,
                vllm_base_url=vllm_base_url,
                max_completion_length=grpo_config.max_completion_length,
                eval_log_dir=eval_log_dir,
                report_to=grpo_config.report_to,
                skip_baseline=resume_from is not None,
            )
        )

    import inspect_rl.util.display as _display

    _display.max_steps = grpo_config.max_steps

    logger.info("Constructing GRPOTrainer (this loads model weights)…")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        rollout_func=rollout_func,
        peft_config=peft_config,
        callbacks=callbacks or None,
    )
    logger.info("GRPOTrainer ready")
    from transformers.trainer_callback import PrinterCallback

    trainer.remove_callback(PrinterCallback)
    _save_wandb_run_id(run_dir)
    logger.info(
        "=== trainer.train() starting (max_steps=%d) ===", grpo_config.max_steps
    )
    if is_main:
        # 30s tick is short enough to catch a 1-2 min hang quickly without
        # flooding train.log on healthy runs (~120 lines/hr).
        start_heartbeat(interval_s=30.0)
        set_phase(
            "training", step=run_dir_payload["resume_step"] if run_dir_payload else 0
        )
    try:
        trainer.train(resume_from_checkpoint=checkpoint_path)
    except KeyboardInterrupt:
        logger.warning("Interrupted — saving checkpoint…")
        trainer.save_model(grpo_config.output_dir + "/checkpoint-interrupted")
        logger.info("Saved to %s/checkpoint-interrupted", grpo_config.output_dir)
    except Exception:
        logger.exception("Training crashed")
        raise
    finally:
        set_phase("shutdown")
        stop_heartbeat()
        if hasattr(rollout_func, "shutdown"):
            rollout_func.shutdown()
        if is_main:
            _emit_run_summary(step_timer, rollout_func)
            _write_run_report(run_dir, model, grpo_config, step_timer, rollout_func)
    logger.info("=== inspect_rl_train done ===")
    return trainer


def _write_run_report(
    run_dir: Path,
    model: str,
    grpo_config: GRPOConfig,
    step_timer: _StepTimerCallback,
    rollout_func: Any,
) -> None:
    n = step_timer.total_steps
    if n == 0:
        return
    avg_rollout = step_timer.sum_rollout_s / n if step_timer.sum_rollout_s else None
    avg_train = (
        step_timer.sum_weight_update_s / n if step_timer.sum_weight_update_s else None
    )
    if hasattr(rollout_func, "prefetch_enabled"):
        enabled = rollout_func.prefetch_enabled()
    elif hasattr(rollout_func, "_thread"):
        enabled = True
    else:
        enabled = False
    discarded = getattr(rollout_func, "discarded_count", None)
    if callable(discarded):
        discarded = discarded()

    wandb_run_id: str | None = None
    try:
        import wandb

        if wandb.run is not None:
            wandb_run_id = wandb.run.id
    except ImportError:
        pass

    try:
        _summary.write_report(
            run_dir,
            model=model,
            max_steps=grpo_config.max_steps,
            n_steps_run=n,
            total_s=step_timer.total_run_s(),
            avg_rollout_s=avg_rollout,
            avg_train_s=avg_train,
            prefetch_enabled=enabled,
            discarded=discarded if isinstance(discarded, int) else None,
            wandb_run_id=wandb_run_id,
        )
    except Exception:
        logger.exception("Failed to write report.md")


def _emit_run_summary(step_timer: _StepTimerCallback, rollout_func: Any) -> None:
    n = step_timer.total_steps
    if n == 0:
        return
    avg_rollout = step_timer.sum_rollout_s / n if step_timer.sum_rollout_s else None
    avg_train = (
        step_timer.sum_weight_update_s / n if step_timer.sum_weight_update_s else None
    )
    if hasattr(rollout_func, "prefetch_enabled"):
        enabled = rollout_func.prefetch_enabled()
    elif hasattr(rollout_func, "_thread"):
        enabled = True
    else:
        enabled = False
    discarded = getattr(rollout_func, "discarded_count", None)
    if callable(discarded):
        discarded = discarded()
    log_run_summary(
        n_steps=n,
        total_s=step_timer.total_run_s(),
        avg_rollout_s=avg_rollout,
        avg_train_s=avg_train,
        prefetch_enabled=enabled,
        discarded=discarded if isinstance(discarded, int) else None,
    )


class _StepTimerCallback(TrainerCallback):
    """Log per-step weight-update duration and aggregate for the run summary.

    Step time breaks into:
      - rollout: logged from rollout.rollout_func
      - weight update: on_step_end - rollout_end (loss + backward + optim + sync)
    """

    def __init__(self) -> None:
        self._step_start: float = 0.0
        self._run_start: float = 0.0
        self.total_steps: int = 0
        self.sum_rollout_s: float = 0.0
        self.sum_weight_update_s: float = 0.0

    def on_train_begin(
        self, args: Any, state: Any, control: Any, **kwargs: Any
    ) -> None:
        self._run_start = time.perf_counter()

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._step_start = time.perf_counter()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        now = time.perf_counter()
        total = now - self._step_start
        rollout_end = _rollout_mod.last_rollout_end_ts
        self.total_steps += 1
        if rollout_end is not None and rollout_end >= self._step_start:
            wu = now - rollout_end
            self.sum_rollout_s += rollout_end - self._step_start
            self.sum_weight_update_s += wu
            logger.info(
                "[step %d] weight update done in %.1fs (step total %.1fs)",
                state.global_step,
                wu,
                total,
            )
        else:
            self.sum_weight_update_s += total
            logger.info("[step %d] step total %.1fs", state.global_step, total)

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        logger.info(
            "[step %d] checkpoint saved → %s", state.global_step, args.output_dir
        )

    def total_run_s(self) -> float:
        return time.perf_counter() - self._run_start if self._run_start else 0.0


class _InspectEvalCallback(TrainerCallback):
    """Run Inspect eval on a held-out set every N training steps."""

    def __init__(
        self,
        eval_task: Task,
        eval_steps: int,
        eval_limit: int,
        model_name: str,
        vllm_base_url: str,
        max_completion_length: int,
        eval_log_dir: str | None = None,
        report_to: str | list[str] = "none",
        skip_baseline: bool = False,
    ) -> None:
        self.eval_task = eval_task
        self.eval_steps = eval_steps
        self.eval_limit = eval_limit
        self.model_name = model_name
        self.vllm_base_url = vllm_base_url
        self.max_completion_length = max_completion_length
        self.eval_log_dir = eval_log_dir
        self.report_to = report_to
        self.skip_baseline = skip_baseline
        self._wandb_metrics_defined = False

    def on_train_begin(
        self, args: Any, state: Any, control: Any, **kwargs: Any
    ) -> None:
        if self.skip_baseline:
            return
        if not state.is_world_process_zero:
            return
        logger.info("[eval step 0] baseline val eval starting")
        self._run_eval(step=0, **kwargs)

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if state.global_step % self.eval_steps != 0:
            return
        if not state.is_world_process_zero:
            return
        logger.info("[eval step %d] val eval starting", state.global_step)
        self._run_eval(step=state.global_step, **kwargs)

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if not state.is_world_process_zero:
            return
        if state.global_step % self.eval_steps == 0:
            return
        logger.info("[eval step %d] final val eval starting", state.global_step)
        self._run_eval(step=state.global_step, **kwargs)

    def _run_eval(self, step: int, **kwargs: Any) -> None:
        tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
        inspect_model = get_model(
            f"trl-vllm/{self.model_name}",
            base_url=self.vllm_base_url,
            tokenizer=tokenizer,
            config=GenerateConfig(
                max_tokens=self.max_completion_length,
                temperature=0.0,
            ),
            batch_timeout=1.0,
            max_batch_size=self.eval_limit,
        )

        samples = list(self.eval_task.dataset)[: self.eval_limit]
        eval_t = task_with(
            self.eval_task,
            dataset=MemoryDataset(samples=samples),
            epochs=1,
        )

        eval_kwargs: dict[str, Any] = {}
        if self.eval_log_dir:
            step_log_dir = Path(self.eval_log_dir) / f"{step:03d}_val"
            step_log_dir.mkdir(parents=True, exist_ok=True)
            eval_kwargs["log_dir"] = str(step_log_dir)

        t0 = time.perf_counter()
        set_phase("eval", step=step)
        try:
            with (
                INSPECT_EVAL_LOCK,
                capture_inspect_log(step_log_dir if self.eval_log_dir else None),
            ):
                logs = inspect_eval(
                    eval_t,
                    model=inspect_model,
                    max_samples=self.eval_limit * 2,
                    display="log",
                    log_level="warning",
                    fail_on_error=False,
                    **eval_kwargs,
                )
            elapsed = time.perf_counter() - t0
            logger.info(
                "[eval step %d] done in %.1fs → %s",
                step,
                elapsed,
                str(step_log_dir) if self.eval_log_dir else "?",
            )
            metrics = self._extract_metrics(logs[0])
            log_eval(metrics, step, elapsed)
            self._log_wandb(metrics, step)
        except Exception:
            logger.exception("[eval step %d] validation failed", step)
        finally:
            set_phase("training", step=step)
            if hasattr(inspect_model, "aclose"):
                try:
                    asyncio.run(inspect_model.aclose())
                except RuntimeError:
                    pass

    def _extract_metrics(self, log: Any) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if log.results and log.results.scores:
            for eval_score in log.results.scores:
                for metric_name, metric in eval_score.metrics.items():
                    metrics[f"eval/{eval_score.name}/{metric_name}"] = metric.value
        return metrics

    def _log_wandb(self, metrics: dict[str, float], step: int) -> None:
        use_wandb = self.report_to == "wandb" or (
            isinstance(self.report_to, list) and "wandb" in self.report_to
        )
        if not use_wandb:
            return
        try:
            import wandb

            if wandb.run is not None:
                if not self._wandb_metrics_defined:
                    wandb.define_metric("eval/step")
                    wandb.define_metric("eval/*", step_metric="eval/step")
                    self._wandb_metrics_defined = True
                wandb.log({**metrics, "eval/step": step})
        except ImportError:
            pass


def _prepare_resume(run_dir: Path) -> tuple[str, int]:
    """Find the latest checkpoint and clean up eval logs after it.

    Returns (checkpoint_path, resume_step).
    """
    ckpt_dir = run_dir / "checkpoints"
    checkpoints = sorted(ckpt_dir.glob("checkpoint-*"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    def _ckpt_step(p: Path) -> int:
        m = re.search(r"checkpoint-(\d+)", p.name)
        return int(m.group(1)) if m else -1

    latest = max(checkpoints, key=_ckpt_step)
    resume_step = _ckpt_step(latest)

    eval_log_dir = run_dir / "eval_logs"
    if eval_log_dir.is_dir():
        for entry in sorted(eval_log_dir.iterdir()):
            m = re.match(r"(\d+)_", entry.name)
            if m and int(m.group(1)) > resume_step:
                logger.info(
                    "Cleaning up %s (after checkpoint step %d)", entry.name, resume_step
                )
                shutil.rmtree(entry)

    # Resume wandb run if manifest records the run ID.
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        wandb_id = manifest.get("wandb_run_id")
        if wandb_id:
            os.environ["WANDB_RUN_ID"] = wandb_id
            # `must` (not `allow`) so we fail loudly if the cloud run was deleted
            # — silently creating a fresh run would split metrics across two
            # charts. The first heartbeat from wandb.init() then flips the
            # server-side state from crashed/cancelled back to running.
            os.environ["WANDB_RESUME"] = "must"
            _revive_wandb_run(wandb_id)
            logger.info("Resuming wandb run %s (WANDB_RESUME=must)", wandb_id)

    return str(latest), resume_step


def _revive_wandb_run(wandb_id: str) -> None:
    """Query the Public API and log the prior state so the operator can see
    that a crashed/cancelled run is about to be revived.

    The actual state flip happens automatically once wandb.init() lands a
    heartbeat — we just surface what the wandb server currently thinks. Best
    effort: missing wandb, offline mode, or unset entity/project just skip.
    """
    try:
        import wandb

        entity = os.environ.get("WANDB_ENTITY")
        project = os.environ.get("WANDB_PROJECT")
        if not (entity and project):
            return
        api = wandb.Api(timeout=10)
        run = api.run(f"{entity}/{project}/{wandb_id}")
        state = getattr(run, "state", "unknown")
        if state in ("crashed", "killed", "failed", "cancelled"):
            logger.info(
                "Prior wandb run state=%s; will revive on first heartbeat",
                state,
            )
        else:
            logger.info("Prior wandb run state=%s", state)
    except Exception as exc:  # ImportError, CommError, 404, offline — non-fatal
        logger.debug("Could not query wandb run state: %s", exc)


def _extract_scorer_names(task: Task) -> list[str]:
    if task.scorer is None:
        return []
    scorers = task.scorer if isinstance(task.scorer, list) else [task.scorer]
    names = []
    for s in scorers:
        name = (
            getattr(s, "__qualname__", None)
            or getattr(s, "__name__", None)
            or type(s).__name__
        )
        base = name.split(".<locals>")[0] if ".<locals>" in name else name
        names.append(base)
    return names


def _make_scores_reward_func(scorer_name: str) -> RewardFunc:
    """Create a reward function that reads pre-computed scores from the rollout."""

    def reward_func(
        prompts: list[Any],
        completions: list[Any],
        **kwargs: Any,
    ) -> list[float]:
        inspect_scores = kwargs.get("inspect_scores", [])
        if not inspect_scores:
            return [0.0] * len(prompts)
        return [s.get(scorer_name, 0.0) for s in inspect_scores]

    reward_func.__name__ = f"inspect_{scorer_name}"  # type: ignore[attr-defined]
    return reward_func


def _check_vllm_server(base_url: str) -> None:
    import httpx

    try:
        resp = httpx.get(f"{base_url}/health/", timeout=5)
        resp.raise_for_status()
        logger.info("vLLM server healthy at %s", base_url)
    except (httpx.ConnectError, httpx.TimeoutException):
        raise RuntimeError(
            f"vLLM server not reachable at {base_url}. "
            f"Start it with: just serve <model>"
        )

    try:
        httpx.post(
            f"{base_url}/close_communicator/",
            json={},
            timeout=5,
        )
    except Exception:
        pass


def _save_wandb_run_id(run_dir: Path) -> None:
    """Persist wandb run ID to manifest.json so resumed runs can rejoin."""
    try:
        import wandb

        if wandb.run is None:
            return
        manifest_path = run_dir / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        if "wandb_run_id" not in manifest:
            manifest["wandb_run_id"] = wandb.run.id
            manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
            logger.info("Saved wandb run ID %s to manifest", wandb.run.id)
    except ImportError:
        pass


def _configure_wandb_defaults() -> None:
    """Set wandb project/entity env vars if not already configured.

    Scopes runs to the current user so they don't land in a shared team project.
    """
    if "WANDB_PROJECT" not in os.environ:
        os.environ["WANDB_PROJECT"] = "inspect-rl"
    if "WANDB_ENTITY" not in os.environ:
        user = os.environ.get("USER", "unknown").replace(".", "-").lower()
        os.environ["WANDB_ENTITY"] = user
