"""`irl` CLI — entry points for serving vLLM and running example training jobs.

Installed via the ``irl`` console script. Run ``irl --help`` for the full tree.

The serve command is a thin wrapper around ``trl vllm-serve`` — we set
sensible defaults and let Typer own argument parsing. The train commands
mirror the signatures of the per-example ``train()`` functions.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys

import typer

app = typer.Typer(
    add_completion=False,
    help="Inspect-RL CLI — serve vLLM and run example training jobs.",
    no_args_is_help=True,
)

train_app = typer.Typer(
    help="Run an example training job.",
    no_args_is_help=True,
)
app.add_typer(train_app, name="train")


# CUDA_HOME autodetect lives in inspect_rl.__init__ so it fires before torch
# is first imported. Override via the CUDA_HOME env var — a CLI flag here
# would be too late for the in-process trainer.


# ---------------------------------------------------------------------------
# irl serve
# ---------------------------------------------------------------------------


@app.command()
def serve(
    model: str = typer.Argument(
        "Qwen/Qwen2.5-3B-Instruct",
        help="Model ID to load into vLLM.",
    ),
    tensor_parallel_size: int = typer.Option(1, help="vLLM tensor parallel size."),
    gpu_memory_utilization: float = typer.Option(0.9, help="vLLM GPU memory fraction."),
    max_model_len: int = typer.Option(8192, help="vLLM max context length."),
    enforce_eager: bool = typer.Option(
        False,
        "--enforce-eager",
        help=(
            "Skip vLLM's CUDA graph capture. Cuts startup by ~30-60s at the "
            "cost of slower per-token inference — good for smoke runs and "
            "iterative debugging, leave off for real training throughput."
        ),
    ),
    devices: str = typer.Option(
        "0",
        "--devices",
        "--cuda-visible-devices",
        help=(
            "CUDA_VISIBLE_DEVICES for the vLLM process. Comma-separated list "
            "like '0,1' shards the model across those GPUs when paired with "
            "--tensor-parallel-size."
        ),
    ),
) -> None:
    """Start a vLLM inference server via ``trl vllm-serve``.

    Defaults target a single GPU. Override ``--devices`` and
    ``--tensor-parallel-size`` to shard across more devices.
    """
    env = os.environ.copy()
    env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    env["CUDA_VISIBLE_DEVICES"] = devices

    cmd = [
        "trl",
        "vllm-serve",
        "--model",
        model,
        "--tensor_parallel_size",
        str(tensor_parallel_size),
        "--gpu_memory_utilization",
        str(gpu_memory_utilization),
        "--max_model_len",
        str(max_model_len),
    ]
    if enforce_eager:
        cmd.append("--enforce_eager")
    typer.echo(f"$ CUDA_VISIBLE_DEVICES={devices} {shlex.join(cmd)}")
    rc = subprocess.call(cmd, env=env)
    raise typer.Exit(code=rc)


# ---------------------------------------------------------------------------
# irl train <example>
# ---------------------------------------------------------------------------
#
# Each example exposes a `train()` function with its own defaults. We wire
# them in as Typer commands so `irl train --help` shows the matrix of jobs.
#
# Train-wide options (device masking + distributed stub) live on the train
# sub-app callback so users write `irl train --devices 0,1 math-agent …`
# rather than repeating the flags per subcommand.
# ---------------------------------------------------------------------------


def _apply_train_env(devices: str | None, num_processes: int) -> None:
    """Set CUDA_VISIBLE_DEVICES and gate the distributed stub."""
    if devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
    if num_processes != 1:
        # Multi-process runs must come up via `accelerate launch` so each
        # rank gets LOCAL_RANK / WORLD_SIZE / MASTER_ADDR in its env. The
        # rollout-side rank-gating + run-dir broadcast in inspect_rl.core.trainer
        # then handles the rest. We refuse to silently re-exec the CLI
        # ourselves — clearer to let the user own the launcher.
        raise typer.BadParameter(
            "multi-process training must be started via accelerate, e.g.\n"
            "  uv run accelerate launch --num_processes "
            f"{num_processes} -m inspect_rl.cli train …\n"
            "Do NOT pass --num-processes to plain `irl train`.",
            param_hint="--num-processes",
        )


@train_app.callback()
def _train_bootstrap(
    devices: str = typer.Option(
        None,
        "--devices",
        "--cuda-visible-devices",
        help=(
            "CUDA_VISIBLE_DEVICES for the trainer. Single-process today; "
            "comma-separated '0,1,2,3' is intended for future distributed "
            "runs (see --num-processes)."
        ),
    ),
    num_processes: int = typer.Option(
        1,
        "--num-processes",
        help=(
            "Distributed world size. Only 1 is implemented; the flag exists "
            "so `accelerate launch irl train --num-processes N …` will Just "
            "Work once DDP/FSDP wiring lands."
        ),
    ),
) -> None:
    _apply_train_env(devices, num_processes)


@train_app.command("tldr")
def train_tldr(
    model: str = "google/gemma-3-270m-it",
    vllm_base_url: str = "http://localhost:8000",
    output_dir: str = "irl_output",
    max_steps: int = 200,
    per_device_train_batch_size: int = 8,
    num_generations: int = 8,
    dataset_limit: int = 1000,
    max_completion_length: int = 512,
    learning_rate: float = 1e-5,
    eval_steps: int = 20,
    eval_limit: int = 32,
    resample_rounds: int = 3,
    off_policy_steps: int = 0,
    resume: str | None = None,
    wandb: bool = False,
    verbose: bool = False,
) -> None:
    """TL;DR summarization — simplest single-turn example."""
    from inspect_rl.example.tldr import train

    train(
        model=model,
        vllm_base_url=vllm_base_url,
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        num_generations=num_generations,
        dataset_limit=dataset_limit,
        max_completion_length=max_completion_length,
        learning_rate=learning_rate,
        eval_steps=eval_steps,
        eval_limit=eval_limit,
        resample_rounds=resample_rounds,
        off_policy_steps=off_policy_steps,
        resume=resume,
        wandb=wandb,
        verbose=verbose,
    )


@train_app.command("magic-number")
def train_magic_number(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    vllm_base_url: str = "http://localhost:8000",
    output_dir: str = "irl_output",
    max_steps: int = 100,
    per_device_train_batch_size: int = 8,
    num_generations: int = 8,
    dataset_limit: int = 256,
    eval_steps: int = 10,
    eval_limit: int = 16,
    resample_rounds: int = 3,
    off_policy_steps: int = 0,
    resume: str | None = None,
    wandb: bool = True,
    verbose: bool = False,
) -> None:
    """Magic number — minimal multi-turn agent sanity test."""
    from inspect_rl.example.magic_number import train

    train(
        model=model,
        vllm_base_url=vllm_base_url,
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        num_generations=num_generations,
        dataset_limit=dataset_limit,
        eval_steps=eval_steps,
        eval_limit=eval_limit,
        resample_rounds=resample_rounds,
        off_policy_steps=off_policy_steps,
        resume=resume,
        wandb=wandb,
        verbose=verbose,
    )


@train_app.command("math-agent")
def train_math_agent(
    model: str = "Qwen/Qwen2.5-3B-Instruct",
    vllm_base_url: str = "http://localhost:8000",
    output_dir: str = "irl_output",
    max_steps: int = 150,
    per_device_train_batch_size: int = 8,
    num_generations: int = 4,
    dataset_limit: int = 1000,
    eval_steps: int = 10,
    eval_limit: int = 32,
    learning_rate: float = 5e-6,
    beta: float = 0.05,
    save_steps: int = 10,
    resample_rounds: int = 3,
    off_policy_steps: int = 0,
    resume: str | None = None,
    wandb: bool = False,
    verbose: bool = False,
) -> None:
    """Multi-turn math agent with calculator tool."""
    from inspect_rl.example.math_agent import train

    train(
        model=model,
        vllm_base_url=vllm_base_url,
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        num_generations=num_generations,
        dataset_limit=dataset_limit,
        eval_steps=eval_steps,
        eval_limit=eval_limit,
        learning_rate=learning_rate,
        beta=beta,
        save_steps=save_steps,
        resample_rounds=resample_rounds,
        off_policy_steps=off_policy_steps,
        resume=resume,
        wandb=wandb,
        verbose=verbose,
    )


@train_app.command("gsm8k")
def train_gsm8k(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    vllm_base_url: str = "http://localhost:8000",
    output_dir: str = "irl_output",
    max_steps: int = 200,
    per_device_train_batch_size: int = 16,
    num_generations: int = 8,
    dataset_limit: int = 5000,
    eval_steps: int = 5,
    eval_limit: int = 50,
    resample_rounds: int = 3,
    off_policy_steps: int = 0,
    resume: str | None = None,
    wandb: bool = False,
    verbose: bool = False,
) -> None:
    """GSM8K with structured XML output and LoRA."""
    from inspect_rl.example.gsm8k import train

    train(
        model=model,
        vllm_base_url=vllm_base_url,
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        num_generations=num_generations,
        dataset_limit=dataset_limit,
        eval_steps=eval_steps,
        eval_limit=eval_limit,
        resample_rounds=resample_rounds,
        off_policy_steps=off_policy_steps,
        resume=resume,
        wandb=wandb,
        verbose=verbose,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(app())
