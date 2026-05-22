# 002 — Tier 1 implementation: GRPO mods + active sampling

Incremental changes — no architecture rework, backwards compatible (resample_rounds=0 gives old behaviour).

## What changed

### Active sampling (rollout.py)

Refactored `make_inspect_rollout_func` to extract eval+scoring into a reusable `_run_eval` helper, then wrapped the main `rollout_func` with a resample loop.

New parameter: `max_resample_rounds` (default 0, examples set it to 3).

Algorithm per step:
1. Run the normal rollout (inspect_eval on all samples)
2. Group completions by prompt (stride = `num_generations`)
3. For each group, check if all completions got identical scores → zero advantage → zero gradient
4. If zero-gradient groups exist, regenerate just those groups (new inspect_eval, different random completions due to temperature=1.0)
5. Splice new results back into the batch
6. Repeat up to `max_resample_rounds` times

`_find_zero_gradient_groups` compares score dicts across all completions in a group. If `scores[0] == scores[1] == ... == scores[num_gen-1]`, the group is flagged.

Even with `max_resample_rounds=0`, zero-gradient groups are now detected and logged:
```
[rollout 12] 3/8 groups have zero gradient (all identical scores)
```

This gives visibility into wasted compute before deciding to enable resampling.

### Asymmetric clipping (example configs)

Added `epsilon_high=0.28` to all four example GRPOConfigs. TRL's default is symmetric (epsilon=0.2 for both bounds). The asymmetric setting allows the policy to take larger steps toward high-reward completions while still constraining regression on negative ones.

The value 0.28 is from OLMo 3 / DAPO conventions — the upper bound is ~40% wider than the lower bound.

### Token-level loss

Already the TRL 1.2.0 default (`loss_type="dapo"`). No change needed. This normalizes by total active tokens in the batch rather than per-sequence, which eliminates length bias (short completions no longer contribute disproportionate per-token gradient).

### What we did NOT change

- **beta (KL penalty)**: OLMo 3 uses beta=0 but our math_agent depends on beta=0.05 to prevent mode collapse at small batch sizes. Left unchanged.
- **scale_rewards**: OLMo 3 drops std-dev normalization. TRL default is `scale_rewards="group"`. Leaving as-is until we can A/B test — the OLMo 3 recommendation assumes much larger batch sizes.
- **No offline difficulty filtering**: OLMo 3 pre-filters prompts with >62.5% majority pass rate. Could add this later but it requires a pre-training eval sweep.

## Files changed

| File | Change |
|------|--------|
| `rollout.py` | Extracted `_run_eval` helper, added resample loop + `_find_zero_gradient_groups`, zero-gradient logging |
| `trainer.py` | New `max_resample_rounds` param on `inspect_rl_train`, passed to rollout factory |
| `cli.py` | Added `--resample-rounds` to all four train subcommands |
| `example/math_agent.py` | `epsilon_high=0.28`, `resample_rounds=3` |
| `example/gsm8k.py` | `epsilon_high=0.28`, `resample_rounds=3` |
| `example/tldr.py` | `epsilon_high=0.28`, `resample_rounds=3` |
| `example/magic_number.py` | `epsilon_high=0.28`, `resample_rounds=3` |

## Expected behaviour during training

With `resample_rounds=3` and `num_generations=4`:

- **Easy dataset (late training)**: Many groups all-correct → first round regenerates ~50% of groups → some flip to mixed → second round catches stragglers → net: 2-4x fewer wasted steps.
- **Hard dataset (early training)**: Most groups all-wrong → regeneration unlikely to help → caps at 3 rounds, moves on. Small overhead from the extra evals (~1 inspect_eval per round on the subset), but these are fast since they're smaller batches.
- **Boundary (typical mid-training)**: ~20-40% zero-gradient groups → 1-2 resample rounds catches most of them.

The overhead per resample round is one `inspect_eval` call on the zero-gradient subset. For a batch of 8 samples with 2 zero-gradient groups (4 samples to regenerate), that's ~50% of a normal rollout's inference cost per round. Worst case (3 rounds, all groups zero-gradient every time) is 4x the inference cost — but this only happens when the dataset is trivially easy and you'd want to switch datasets anyway.

## Test plan

- `just lint` passes (ruff format + check)
- `just test` passes (23/23)
- Run `irl train math-agent --max-steps 20` and verify:
  - Zero-gradient logging appears in train.log
  - Resample rounds fire when zero-gradient groups exist
  - Training curve looks at least as good as before
