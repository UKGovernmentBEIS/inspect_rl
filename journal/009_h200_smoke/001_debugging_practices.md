# 009/001 — Debugging practices from the H200 single-node smoke

Notes from getting `examples/configs/hpc2/h200/slurm_2gpu.sbatch` to run end-to-end
with W&B on a shared, non-cgroup-constrained Slurm cluster. Captures the
practices worth keeping; nothing site-specific.

## Practices

### 1. Live-tail the three logs that matter

The split-job sbatch writes three independent streams. Watching them
together makes "is it stuck on vLLM warmup or on weight-sync?" instantly
obvious:

```
tail -F irl_output_slurm/latest/{master,vllm,trainer}.out
```

`master.out` = orchestrator + watchdog. `vllm.out` = server. `trainer.out`
= timestamped trainer step. Don't tail just one — silence in `trainer.out`
during vLLM warmup is expected, not a hang.

### 2. The watchdog's "idle" warnings are not errors

The sbatch watchdog complains every 30 s when `trainer.out` hasn't grown.
That fires both during the genuine pre-train phase (imports → tokenizer →
model load → NCCL handshake; legitimately silent for 60–120 s) **and**
after the trainer exits cleanly while vLLM stays warm. Treat watchdog
lines as informational and correlate with the trainer's own
`[heartbeat] step=N phase=…` (emitted every 30 s once
`trainer.train()` is running) before concluding anything is wedged.

### 3. Always reset the NCCL communicator before re-attaching to vLLM

If a trainer step crashes inside `init_communicator`, vLLM's HTTP
endpoint stays healthy but its `StatelessProcessGroup` is poisoned —
the next attempt will fail with `NCCL error: remote process exited or
there was a network error` at `ncclCommInitRank`. The HTTP-level reset
is the first thing to try:

```bash
curl -sf -X POST http://<vllm-node>:8000/close_communicator/ \
    -H 'Content-Type: application/json' -d '{}'
```

If a second trainer attach still fails, the process group is unrecoverable
from outside and the whole sbatch job must be cancelled and resubmitted.
This is now documented in the h200 README's "known gotchas".

### 4. W&B vars must be in the submitter's shell at sbatch time

`sbatch --export=ALL` snapshots the env at submission. Adding
`WANDB_API_KEY` / `WANDB_ENTITY` / `WANDB_BASE_URL` to `~/.zshrc`
**after** a job has been submitted does nothing for that job — the
running srun steps inherit the submission-time env. Workflow:

```bash
source ~/.zshrc   # or `set -a; source .env; set +a`
env -u MODEL WANDB=1 sbatch examples/configs/hpc2/h200/slurm_2gpu.sbatch
```

The `WANDB=1` gate was added so the trainer step picks up `--wandb`
conditionally; you don't pay W&B sign-in latency on every smoke.

### 5. `env -u MODEL` is a real footgun guard, not paranoia

`sbatch --export=ALL` propagates the submitter's `MODEL` if any
parent process (tmux session, sourced rc file, agent harness) has
exported one for unrelated reasons. The sbatch's own default then
silently loses to it. `env -u MODEL` unsets `MODEL` for the sbatch
invocation only; the rest of the env still propagates.

Generalises: any sbatch with `${VAR:-default}` env-driven knobs is one
exported `VAR` in the parent shell away from being overridden without
warning. When debugging "why didn't my flag take effect?", check
`env | grep -i <var>` in the submitter before blaming the script.

### 6. `--wandb` failures are independent of NCCL failures

Don't infer "wandb broke the run" from a `NCCL error` traceback right
after enabling `--wandb`. The two are independent — wandb init runs
inside `inspect_rl_train` before `GRPOTrainer.__init__`, NCCL failures
happen later in `init_communicator`. If a wandb-enabled run crashes on
NCCL after a previous wandb-less run crashed on NCCL, it's the same
poisoned-vLLM bug, not a wandb regression.

## Infosec / hygiene reminders

- **Never put real `WANDB_API_KEY` / `HF_TOKEN` values into commit
  messages, journal entries, or chat transcripts.** Rotate immediately
  if a key leaks into any system whose logs you don't fully control
  (W&B at <https://wandb.ai/authorize>, HF at
  <https://huggingface.co/settings/tokens>).
- **Don't pass secrets on the `srun --export=...` command line** if
  there's any chance someone else can `ps`/`squeue -o %k` on the node.
  Prefer sourcing them into the submitter shell + `--export=ALL`, or
  reading them from a 0600 file inside the job.
- **Scrub before journaling.** Cluster node names, partition/QoS names,
  shared filesystem paths, and IP addresses can drift into log excerpts
  pasted into journal entries. Existing entries in `008_multinode/`
  use `<redacted-nid>` / `<redacted-scratch>` placeholders — follow
  that pattern.
- **`/close_communicator/` accepts an empty body and returns 200 even
  for a poisoned group.** Don't read a 200 as evidence the underlying
  TCPStore is healthy; only a successful next `init_communicator`
  proves that.

## Repo changes from this session

- `examples/configs/hpc2/h200/slurm_2gpu.sbatch`: added `WANDB=1`-gated
  `--wandb` flag passthrough.
- `examples/configs/hpc2/h200/hf_load_sanity.sbatch`: removed. The plain-HF
  load test wasn't pulling its weight as a separate sbatch — when vLLM
  is slow, the right move is to read `vllm.out` rather than spin up
  another job. Kept the practice in mind for future ad-hoc debugging.
- `README.md`: distilled the "shared Slurm tenant" guidance to a single
  bullet under the agent-guidance list.
- `justfile`: added `just cluster-status` for at-a-glance partition /
  squeue / per-node GRES view before submitting.
