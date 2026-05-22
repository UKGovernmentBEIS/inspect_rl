# 002 — Code fixes for observability + disk

Cleanup pass focused on "I want to watch a 60 min training run and understand what's happening." All internal wiring — no reward/curriculum changes.

## What changed

**Logging**
- New `inspect_rl/logging_config.py`: one call sets up the `inspect_rl` namespace with a timestamped stdout handler + a per-run `train.log` file handler. `tail -f run_dir/train.log` matches notebook output.
- Section-boundary `logger.info` in `trainer.py` (vLLM check, run dir, dataset convert, GRPOTrainer construct, train start) and `rollout.py` (rollout start/done). Prior "hang" before step 1 was always the baseline val eval (1m44s, 32 samples) with no log to prove the process was alive.
- `display="log"` replaces `display="none"` for every `inspect_eval` call. Kills the `:: Output()` Jupyter widget placeholders that `rich.Live` emits regardless of display level.
- Third-party loggers (`inspect_ai`, `transformers`) pinned at WARNING so per-sample INFO chatter doesn't leak in.

**Timing**
- `[rollout N] done in X.Xs` — generation + scoring wall time.
- `[step N] weight update done in X.Xs (total Y.Ys)` — `on_step_end - rollout_end` via a `_StepTimerCallback`, so we can see if the bottleneck is inspect_eval or gradient update.
- `[eval step N] done in X.Xs` — full val-eval cost.

**Eval log layout**
- Single `eval_logs/` dir. Train rollouts → `001_rollout.eval`, val evals → `010_val.eval`. `ls eval_logs/` sorts chronologically. Old `val_eval_logs/` split removed.

**Disk**
- `outputs/` is now a symlink → `$SCRATCHDIR/outputs/inspect-rl/` (matches `.venv`). Home quota trips at ~35GB per full-state 3B checkpoint; six of those per run blew the quota mid-training.
- `GRPOConfig(save_total_limit=1)` — only keep the latest full-state checkpoint.
- `INSPECT_TELEMETRY` popped at package import (the `<redacted: telemetry provider package>` CloudWatch PutLogEvents 408s were flooding 50-line tracebacks per sample).

## Where we are experimentally

Still on the same POC as iter5/iter6: Qwen2.5-3B-Instruct on GSM8K-main, tools = [calculator, submit], `beta=0.05`, `lr=5e-6`, `bs=8 num_gen=4`, reward weights `[4.0, 0.5, 0.1, -0.5]`. Fixed 32-sample val set from GSM8K-test, `eval_steps=10`.

Prior iter6 run (killed at step 25 by the disk-quota issue before fixes) had this curve through step 20:

| metric              | step 0 | step 10 | step 20 |
|---------------------|-------:|--------:|--------:|
| correctness         | 0.25   | 0.22    | 0.25    |
| uses_calculator     | 0.75   | 1.00    | 1.00    |
| valid_submit        | 0.41   | 0.53    | 0.72    |
| tool_call_failures  | 0.06   | 0.09    | 0.00    |

Read: format (calculator + submit) learned very fast; correctness still noise around baseline at step 20. Prior experience says 0.5B collapses around step 30; 3B hasn't so we expect a real correctness signal by step 50–80 if the reward shape is right.

## Second pass (same PR)

**Eval log layout v2**
- Switched from flat `001_rollout.eval` files to subdirectories: `001_rollout/`, `010_val/`. Inspect writes its `.eval` file inside the subdir. Fixes `FileNotFoundError` from inspect_ai's post-eval hooks trying to rename files we'd already moved.

**Notebook vs train.log split**
- `_ConsoleFilter` on the stdout handler: only `inspect_rl.display` records and WARNING+ reach the notebook cell. Everything else (timing, rollout details, crash tracebacks) goes to `train.log` only.
- `log_run_start()` prints the log file path as the first notebook line so users know where to `tail -f`.

**inspect_ai log noise**
- Pinned `inspect_ai` logger to WARNING. inspect_ai reconfigures its own loggers inside `inspect_eval()`, but since we no longer share handlers between namespaces, our file handler stays clean.
- Rollout evals use `display="log"`, `log_level="warning"` — controls inspect's output at the source.

**Run directory improvements**
- `_REPO_ROOT` in `run_dir.py` resolves output paths relative to repo root regardless of cwd (previously broke when Jupyter kernel started from a different directory).
- Atomic `outputs/latest` symlink (tmp symlink + `Path.replace`) so `tail -f outputs/latest/train.log` always works.

**Checkpoint/eval cadence**
- `save_steps` default changed from 25 → 10 to match `eval_steps=10`. Every checkpoint now has a corresponding val eval.
- Added `on_train_end` to `_InspectEvalCallback` — runs a final val eval unless the last step already triggered one.

**Resume support**
- `inspect_rl_train(resume_from="outputs/<ts>")` reuses the run dir, finds the latest `checkpoint-*`, deletes all eval_log entries after that step, and calls `trainer.train(resume_from_checkpoint=...)`.
- HF Trainer restores `global_step`, optimizer, scheduler automatically. Rollout numbering (`step_idx = global_step + 1`) is correct without changes.
- Display step counter initialized from checkpoint step via `set_step_counter()`.
- Wandb run ID saved to `manifest.json` on first run; `WANDB_RUN_ID` + `WANDB_RESUME=allow` set on resume so metrics continue on the same chart.
- `skip_baseline=True` on resume (no redundant step-0 eval).

## First run on new stack

Step-20 val eval showed correctness jumping from 0.156 → 0.406 (step 0 → 10 → 20 pending). Run hung during the step-20 val eval — inspect_ai's async runner wedged while vLLM was healthy (confirmed via curl). Kernel restart was required; no checkpoint existed (old `save_steps=25`). New `save_steps=10` and resume support should prevent data loss next time.

## Next

- Resume or restart the 150-step run with the new checkpoint cadence.
- If correctness still climbing at step 50, let it run to completion and capture final val.
- If it plateaus, investigate reward-weight sensitivity or inspect failure modes via `inspect view`.
