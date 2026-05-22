"""Minimal consumer of `inspect-rl` — proves the package installs and its
public surface is reachable from a downstream project. Not a training run.

Run with:

    uv run example-pkg
"""

from __future__ import annotations


def main() -> None:
    # Import inside main() so --help and any import errors surface here
    # rather than at entry-point discovery time.
    import inspect_rl
    from inspect_rl.cli import app
    from inspect_rl.util.run_dir import resolve_output_root

    print("inspect_rl imported OK")
    print(f"  inspect_rl_train:     {inspect_rl.inspect_rl_train!r}")
    print(f"  cli app:              {app!r}")
    print(f"  resolved output root: {resolve_output_root()}")
