"""Smoke tests: package surface imports cleanly with no side-effect crashes.

Kept deliberately cheap — the goal is to catch "moved a module" / "broke the
__init__" regressions without needing a GPU or network. Anything heavier
(inspect_eval, GRPOTrainer construction, vLLM contact) belongs elsewhere.
"""

from __future__ import annotations

import importlib
import os

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "inspect_rl",
        "inspect_rl.cli",
        "inspect_rl.core.rollout",
        "inspect_rl.core.trainer",
        "inspect_rl.core.trl_vllm_provider",
        "inspect_rl.util._cuda",
        "inspect_rl.util.display",
        "inspect_rl.util.heartbeat",
        "inspect_rl.util.logging_config",
        "inspect_rl.util.run_dir",
        "inspect_rl.perf.prefetch",
        "inspect_rl.example.tldr",
        "inspect_rl.example.magic_number",
        "inspect_rl.example.math_agent",
        "inspect_rl.example.gsm8k",
    ],
)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)


def test_public_entry_points_exposed() -> None:
    import inspect_rl
    from inspect_rl.cli import app, main

    assert callable(inspect_rl.inspect_rl_train)
    assert callable(main)
    assert app is not None


def test_cli_help_lists_subcommands() -> None:
    """Typer wires `rl serve` + `rl train <example>` subcommands at import time."""
    from typer.testing import CliRunner

    from inspect_rl.cli import app

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "serve" in result.output
    assert "train" in result.output

    train_help = CliRunner().invoke(app, ["train", "--help"])
    assert train_help.exit_code == 0, train_help.output
    for name in ("tldr", "magic-number", "math-agent", "gsm8k"):
        assert name in train_help.output, (name, train_help.output)


def test_run_dir_resolution_prefers_scratch(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("SCRATCH", str(scratch))
    monkeypatch.delenv("INSPECT_RL_OUTPUT_ROOT", raising=False)

    from inspect_rl.util.run_dir import resolve_output_root

    assert resolve_output_root() == scratch.resolve()


def test_run_dir_resolution_uses_override(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "custom"
    override.mkdir()
    monkeypatch.setenv("INSPECT_RL_OUTPUT_ROOT", str(override))
    monkeypatch.setenv("SCRATCH", str(tmp_path / "should_be_ignored"))

    from inspect_rl.util.run_dir import resolve_output_root

    assert resolve_output_root() == override.resolve()


def test_create_run_dir_absolute_bypasses_lookup(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SCRATCH", "/this/should/be/ignored/for/abs/paths")

    from inspect_rl.util.run_dir import create_run_dir

    run = create_run_dir(base=str(tmp_path / "abs"))
    assert run.is_dir()
    assert (run / "checkpoints").is_dir()
    assert (run / "eval_logs").is_dir()
    assert run.is_relative_to(tmp_path)


def test_bootstrap_respects_existing_cuda_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bootstrap_cuda_home is a no-op when CUDA_HOME is already set."""
    from inspect_rl.util._cuda import bootstrap_cuda_home

    monkeypatch.setenv("CUDA_HOME", "/user/override")
    bootstrap_cuda_home()
    assert os.environ.get("CUDA_HOME") == "/user/override"


def test_bootstrap_autodetects_when_nvhpc_present(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """autodetect_cuda_home finds `<root>/Linux_<arch>/<release>/cuda/bin/nvcc`."""
    fake_nvhpc = tmp_path / "hpc_sdk"
    cuda_root = fake_nvhpc / "Linux_aarch64" / "24.11" / "cuda"
    (cuda_root / "bin").mkdir(parents=True)
    (cuda_root / "bin" / "nvcc").write_text("")

    import inspect_rl.util._cuda as _cuda

    monkeypatch.setattr(
        _cuda,
        "autodetect_cuda_home",
        lambda: str(cuda_root) if (cuda_root / "bin" / "nvcc").is_file() else None,
    )
    monkeypatch.delenv("CUDA_HOME", raising=False)
    _cuda.bootstrap_cuda_home()
    assert os.environ.get("CUDA_HOME") == str(cuda_root)


def test_importing_inspect_rl_does_not_import_torch() -> None:
    """`import inspect_rl` must not pull torch — otherwise `irl --help` is
    slow and CUDA_HOME bootstraps applied later would be too late.
    """
    import subprocess
    import sys

    script = (
        "import sys; import inspect_rl; "
        "print('torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False", result.stdout + result.stderr


def test_lazy_import_inspect_rl_train_works() -> None:
    """`from inspect_rl import inspect_rl_train` resolves via __getattr__."""
    import inspect_rl

    assert callable(inspect_rl.inspect_rl_train)


def test_cli_num_processes_gt_1_redirects_to_accelerate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distributed runs must be started via `accelerate launch`; the CLI flag
    redirects the user there rather than silently re-execing."""
    from typer.testing import CliRunner

    from inspect_rl.cli import app

    # Don't let autodetect mutate the env under the test runner.
    monkeypatch.setenv("CUDA_HOME", "/tmp/ignored-for-this-test")

    result = CliRunner().invoke(
        app, ["train", "--num-processes", "2", "tldr", "--help"]
    )
    assert result.exit_code != 0
    assert "accelerate launch" in result.output


def test_cli_train_devices_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """--devices on `train` exports CUDA_VISIBLE_DEVICES before the subcommand."""
    from typer.testing import CliRunner

    from inspect_rl.cli import app

    monkeypatch.setenv("CUDA_HOME", "/tmp/ignored-for-this-test")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    # `tldr --help` is a cheap way to trigger the train callback without
    # actually constructing a trainer.
    result = CliRunner().invoke(app, ["train", "--devices", "2,3", "tldr", "--help"])
    assert result.exit_code == 0, result.output
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "2,3"
