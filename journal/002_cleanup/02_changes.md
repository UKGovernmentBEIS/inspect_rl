# 002 Cleanup — Changes

## What changed

1. **Old code → `_deprecated/`**: `grpo.py`, old `tldr.py`, `gsm8k.py`, `bfcl.py` moved to `src/inspect_rl/_deprecated/`. Can be deleted once we're confident nothing references them.

2. **Examples renamed**: `tldr_v2.py` → `tldr.py`, `gsm8k_v2.py` → `gsm8k.py`. They're the canonical examples now. `main.py` dispatcher updated to match.

3. **README rewritten**: Explains the new architecture (Inspect-owned rollout), shows the serve+train workflow, has a "write your own task" template. Old tutorial content removed.

4. **Artifact dirs**: `.gitignore` updated with clear section headers. `outputs/`, `logs/`, `wandb/`, and journal subdirs all excluded.

5. **Training display**: New `display.py` prints a compact per-rollout summary (scores, sample completion, timing) via `print()` + `logging` — works in terminal, Jupyter, and wandb streams. Wired into `rollout.py`.

## Still TODO from the brief

- Multi-turn agent example (no sandbox)
- Agent with coding environment example
- LoRA vs full-FT documentation (gsm8k uses LoRA, tldr uses full-FT — could document the pattern more explicitly)
