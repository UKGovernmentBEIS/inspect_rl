# Run report — `2026-05-15-14-59-45_slurm_202881`

- model: `google/gemma-3-270m-it`
- planned steps: 5; completed: 5
- runtime: 5 steps in 12.4s · rollout avg 1.0s · train avg 0.4s · off-policy: sync

## Did the model learn?

Before/after window means (first-N vs last-N), held-out val first:

- eval/tldr_reward/accuracy -190.250→-225.750 (Δ-35.500)
- rollout/tldr_reward first5=-244.750→last5=-244.750 (Δ+0.000)

## Artifacts

- [manifest.json](manifest.json)
- [train.log](train.log)
- latest checkpoint: [`checkpoint-5`](checkpoints/checkpoint-5) (of 1)
- val eval logs: `eval_logs/*_val/` (4 runs; baseline=[`000_val`](eval_logs/000_val), final=[`005_val`](eval_logs/005_val))
- rollout logs: `eval_logs/*_rollout/` (5 steps; final=[`005_rollout`](eval_logs/005_rollout))
