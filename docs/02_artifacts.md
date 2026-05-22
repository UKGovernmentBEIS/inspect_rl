# 02 — Artifacts and logging

*Last updated: 2026-05-15*

What a training run leaves on disk.

## Layout

Each training run creates a timestamped directory:

- **Outside Slurm**: `irl_output/<YYYY-MM-DD-HH-MM-SS>_irl/`
- **Inside Slurm** (`$SLURM_JOB_ID` set): `irl_output_slurm/<YYYY-MM-DD-HH-MM-SS>_slurm_<jobid>/`

Both bases keep a single `latest` symlink at the top so you can `cd irl_output/latest` (or `irl_output_slurm/latest`) regardless of timestamp.

```
irl_output/2026-05-15-14-59-45_irl/
├── manifest.json          # snapshot of run config (model, lr, batch size, …)
├── train.log              # INFO+ log for inspect_rl (tail -f friendly)
├── report.md              # human-readable run summary + artifact-tree links
├── report.html            # same, rendered with a 1-screen stylesheet
├── checkpoints/
│   ├── checkpoint-10/     # saved every `save_steps`
│   │   ├── model.safetensors
│   │   ├── optimizer.pt
│   │   ├── trainer_state.json
│   │   └── ...
│   └── checkpoint-20/
└── eval_logs/
    ├── 000_val/                                  # baseline held-out eval
    │   ├── 2026-05-15T14-59-46_task_….eval     # Inspect structured log
    │   └── eval.log                              # inspect_ai INFO log
    ├── 001_rollout/                              # per-step rollout eval
    │   ├── 2026-05-15T14-59-50_task_….eval
    │   └── eval.log
    ├── 002_val/
    └── …
```

| Path | Contents |
|------|----------|
| `manifest.json` | Run hyperparameters — model, batch size, learning rate, etc. |
| `train.log` | `inspect_rl` INFO+ log — step timings, eval summaries, heartbeats, errors |
| `report.md` / `report.html` | Run summary — before/after deltas on val + rollout, runtime, links into the rest of the tree |
| `checkpoints/` | Model weights, optimizer state, scheduler (saved every `save_steps`) |
| `eval_logs/NNN_rollout/` | Inspect logs for the rollout that produced gradients at step `NNN` |
| `eval_logs/NNN_val/` | Inspect logs for the held-out validation eval at step `NNN` (when an `eval_task` is configured) |
| `eval_logs/*/*.eval` | Inspect structured eval log — open with `inspect view` for full traces |
| `eval_logs/*/eval.log` | `inspect_ai` INFO-level text log for that eval (solver steps, scoring, HTTP) |
| `wandb/` (top-level) | W&B run data (local cache, only if `wandb=True`) |

All output directories are gitignored.

## Example run

[`docs/example_run/`](example_run/) is a checked-in 5-step `tldr` smoke (checkpoint stripped to keep the tree small). Use it as a reference for what `eval_logs/` and `report.{md,html}` look like in practice without having to run training first.

- [example report.md](example_run/report.md)
- [example report.html](example_run/report.html)

## Logging

Two layers: `inspect_rl` (the trainer) and `inspect_ai` (the eval engine).

**`train.log`** captures `inspect_rl` at INFO+ — step timings, eval metric summaries, heartbeat ticks, and errors. The console only shows compact progress lines from `inspect_rl.util.display` and WARNING+, so `tail -f train.log` is the way to watch a run in detail.

**`eval.log`** (one per step directory) captures `inspect_ai` at INFO for the duration of that single eval — solver steps, scorer output, HTTP calls, retries. Outside of eval windows `inspect_ai` is pinned to WARNING to keep `train.log` clean. These per-eval logs are useful for debugging scorer behaviour or sample-level failures without wading through the structured `.eval` file.

**Heartbeat.** A daemon thread emits `[heartbeat] step=N phase=… elapsed_in_phase=Xs` every 30 s while training runs (see `README.md` for the full list of phases). Absent heartbeats for >2 ticks are an unambiguous "wedged" signal.

## Slurm log layout

When launched via the bundled sbatch templates in `examples/configs/`, the sbatch script and the trainer share one directory under `irl_output_slurm/`:

```
irl_output_slurm/
├── latest -> 2026-05-15-15-30-00_slurm_202900   # single rolling symlink
└── 2026-05-15-15-30-00_slurm_202900/
    ├── master.out                                # sbatch driver stdout
    ├── vllm.out                                  # vLLM server stdout
    ├── trainer-0.out                             # trainer node 0 stdout
    ├── trainer-N.out                             # …additional nodes
    ├── manifest.json                             # written by the trainer
    ├── train.log
    ├── report.md / report.html
    ├── checkpoints/
    └── eval_logs/
```

`irl_output_slurm/` is gitignored. The sbatch scripts pre-create the run directory and export `INSPECT_RL_RUN_DIR=<absolute path>` so `create_run_dir` skips its own timestamping and the trainer artifacts land alongside slurm stdouts.
