"""Logging setup for inspect_rl.

Single call to `configure_logging()` wires handlers onto the `inspect_rl`
namespace. The notebook stream is filtered to show only compact progress
lines (from `inspect_rl.util.display`) and warnings/errors. Everything at
INFO+ goes to `train.log` — `tail -f run_dir/train.log` for full detail.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_FMT = logging.Formatter(
    fmt="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

_CONSOLE_FMT = logging.Formatter(fmt="%(message)s")


class _ConsoleFilter(logging.Filter):
    """Pass all inspect_rl records and WARNING+ to the console stream.

    In notebook contexts this is intentionally broad — suppress rollout noise
    there by setting ``logging.getLogger("inspect_rl.core.rollout").setLevel(WARNING)``
    after calling configure_logging.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return record.name.startswith("inspect_rl.")


def configure_logging(
    log_file: Path | str | None = None,
    level: int = logging.INFO,
    is_main_process: bool = True,
) -> None:
    """Configure the `inspect_rl` logger (idempotent).

    - Always attaches a stdout StreamHandler if none is present.
    - If `log_file` is given and ``is_main_process`` is True, attaches a
      FileHandler for that path (dedup'd). Non-main ranks under
      ``accelerate launch`` share the run dir but skip the file handler so
      ``train.log`` is not written N times per line.
    - Silences noisy third-party loggers (inspect_ai, transformers) to WARNING.
    """
    logger = logging.getLogger("inspect_rl")
    logger.setLevel(level)
    logger.propagate = False

    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_stream:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(_CONSOLE_FMT)
        stream.addFilter(_ConsoleFilter())
        logger.addHandler(stream)

    if log_file is not None and is_main_process:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        target = str(log_path.resolve())
        has_file = any(
            isinstance(h, logging.FileHandler) and h.baseFilename == target
            for h in logger.handlers
        )
        if not has_file:
            fh = logging.FileHandler(log_path)
            fh.setFormatter(_FMT)
            logger.addHandler(fh)

    logging.getLogger("inspect_ai").setLevel(logging.WARNING)

    # Transformers is chatty at INFO (PAD/BOS aligns, deprecation warnings); pin WARN.
    for name in ("transformers", "transformers.tokenization_utils_base"):
        logging.getLogger(name).setLevel(logging.WARNING)


@contextmanager
def capture_inspect_log(log_dir: Path | str | None) -> Iterator[None]:
    """Temporarily capture inspect_ai INFO logs into ``eval.log`` within *log_dir*.

    No-op when *log_dir* is None, so callers don't need a conditional.
    Restores the inspect_ai logger to WARNING on exit.
    """
    if log_dir is None:
        yield
        return

    log_path = Path(log_dir) / "eval.log"
    ai_logger = logging.getLogger("inspect_ai")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(_FMT)
    fh.setLevel(logging.INFO)
    ai_logger.addHandler(fh)
    ai_logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        ai_logger.removeHandler(fh)
        fh.close()
        ai_logger.setLevel(logging.WARNING)
