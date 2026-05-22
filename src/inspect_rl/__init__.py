# Keep imports in this file LIGHT. Every downstream entry point — the `irl`
# CLI, `import inspect_rl`, a Jupyter `from inspect_rl import …` — loads this
# module first, and the CUDA_HOME bootstrap below has to run before torch is
# imported anywhere in the process (torch caches cpp_extension.CUDA_HOME at
# its own import time). So we avoid pulling in trainer/torch here and expose
# the public API lazily via `__getattr__`.

import os

# Clear deployment-specific INSPECT_* env vars that point at external provider
# packages. If the provider package isn't installed (the common case for
# external users), inspect_ai aborts at startup. Pops are harmless when the
# vars aren't set.
# - INSPECT_TELEMETRY: routes logs to a remote sink (e.g. CloudWatch); upload
#   failures dump a 50-line traceback per sample on shared infra.
# - INSPECT_API_KEY_OVERRIDE: key-rotation provider; PrerequisiteError on first
#   inspect_eval if the provider module isn't importable.
# - INSPECT_REQUIRED_HOOKS: hard-fails startup if the listed hook isn't an
#   installed entry point.
for _var in ("INSPECT_TELEMETRY", "INSPECT_API_KEY_OVERRIDE", "INSPECT_REQUIRED_HOOKS"):
    os.environ.pop(_var, None)

# Force rich to treat the environment as a terminal, not a Jupyter kernel.
# Otherwise every rich.Live() context (used by inspect_ai even with
# display="none") emits an empty Output() widget placeholder per call,
# filling the cell with `:: Output()` lines between real log lines.
import rich  # noqa: E402

rich.reconfigure(force_jupyter=False, force_terminal=False)

from inspect_rl.util.logging_config import configure_logging  # noqa: E402

configure_logging()

# Ensure CUDA_HOME is set *before* the first torch import (whether triggered
# by a CLI subcommand below or by a user `from inspect_rl import
# inspect_rl_train`). Must stay above the lazy trainer loader.
from inspect_rl.util._cuda import bootstrap_cuda_home  # noqa: E402

bootstrap_cuda_home()

__all__ = ["inspect_rl_train"]


def __getattr__(name: str):
    # Lazy-import the trainer (and thus torch/trl/transformers) only when the
    # public API is actually touched. `irl --help`, `import inspect_rl`, and
    # `import inspect_rl.cli` stay cheap.
    if name == "inspect_rl_train":
        from inspect_rl.core.trainer import inspect_rl_train as _fn

        return _fn
    raise AttributeError(f"module 'inspect_rl' has no attribute {name!r}")
