# 005 — Multi-node trainer hangs in TRL's `init_communicator` AllReduce

## TL;DR

1-node trainer + 1-node vLLM **works** end-to-end (validated 2026-05-13, 2-step smoke `[summary] 2 steps in 213.0s` with 3B). 2+ node trainer **hangs** at `GRPOTrainer.__init__` regardless of FSDP vs DDP. Root cause is in TRL's vLLM↔trainer NCCL communicator setup, not FSDP. Need standalone NCCL test to isolate the cluster vs library factor before deciding the fix.

## Pinpointed location (py-spy on hung process)

```
Thread N (idle): "MainThread"
    ncclAllReduce (vllm/distributed/device_communicators/pynccl_wrapper.py:431)
    all_reduce (vllm/distributed/device_communicators/pynccl.py:170)
    __init__ (vllm/distributed/device_communicators/pynccl.py:144)   ← verification AllReduce
    init_communicator (trl/generation/vllm_client.py:484)
    _init_vllm (trl/generation/vllm_generation.py:304)
    __init__ (trl/generation/vllm_generation.py:282)
    __init__ (trl/trainer/grpo_trainer.py:781)
    inspect_rl_train (inspect_rl/trainer.py:235)
```

The 3-rank vLLM↔trainer NCCL communicator (TP=2 vLLM workers + 1 trainer rank 0) reports `ncclCommInitRank ... rank 2 nranks 3 - Init COMPLETE`. Then the AllReduce that PyNcclCommunicator runs in `__init__` to verify the group either hangs forever or fails with `NCCL error: remote process exited or there was a network error`.

## What works vs what doesn't

| Topology                              | Result                            |
|---------------------------------------|-----------------------------------|
| 1 vLLM (TP=4, 3B) + 1 trainer node DDP | ✅ 213s, 2 steps, checkpoint     |
| 1 vLLM (TP=2, 0.5B) + 1 trainer DDP    | ✅ Same path works               |
| 1 vLLM (TP=2-4) + **2 trainer nodes** DDP | ❌ Hang in init AllReduce      |
| 1 vLLM (TP=4) + 3 trainer nodes FSDP   | ❌ Same hang (originally diagnosed as FSDP) |

The 8-rank intra-trainer NCCL group (DDP cross-node) initializes cleanly on the `cxi` libfabric provider — that part is fine. Only the secondary 3- or 5-rank vLLM↔trainer NCCL group hangs.

## Things tried that did NOT fix it

- `NCCL_P2P_DISABLE=1` — no change, still hangs at AllReduce
- `NCCL_CROSS_NIC=1` (alone) — still hangs
- `NCCL_CROSS_NIC=1` + `NCCL_NET=Socket` — fails fast with `NCCL error: remote process exited` instead of hanging
- Restarting vLLM with completely fresh state — same hang, so not "stale communicator" debt
- Switching from FSDP (3-node) to plain DDP (2-node) — same hang, so not FSDP-specific

Important: the Slurm HPC module pre-sets `NCCL_SOCKET_IFNAME=hsn` (Slingshot HSN), so `NCCL_NET=Socket` still runs over the Slingshot fabric. There may be no actually independent fallback path.

## Working hypothesis

When the trainer has ≥2 nodes:

1. `accelerate launch` initializes the intra-trainer NCCL group across nodes; this consumes the libfabric/CXI endpoint(s) on the trainer's `hsn` interface.
2. TRL then asks rank 0 to create a *second*, *separate* NCCL communicator that includes vLLM's TP workers (on a different physical node).
3. The verification AllReduce on this second communicator either deadlocks because no free CXI endpoint pairs exist between rank-0 trainer and the vLLM workers, or completes a half-connect and the remote side never receives the AllReduce buffer.

Notably, this does *not* show up with a 1-node trainer because in that case the intra-trainer group is intra-NVLink (no libfabric endpoint consumption), leaving CXI fully available for the vLLM↔trainer group.

## Master log layout / iteration improvements made this session

- `slurm_logs/` is now a symlink to `$SCRATCH/slurm_logs/`
- Each sbatch refreshes `slurm_logs/latest_{trainer,vllm}{,_master.out,.out}` symlinks on start → `tail -F` panes survive job restarts
- `--output=/dev/null` + `exec 1>>$LOGDIR/master.out` keeps everything under the job dir
- Inspect/eval logs land at `$LOGDIR/run/<ts>/` via `--output-dir $LOGDIR/run` flag passed through to `irl train`
- Persistent vLLM job (`slurm_vllm_1node.sbatch`) + separate trainer job → `~2 min` warmup avoided per trainer iteration
- py-spy sidecar baked into trainer srun so any future hang gets a fresh stack dump every 60s

## Next step

Standalone NCCL multi-node test (NVIDIA's nccl-tests on `workq`) to determine whether the cluster supports two concurrent NCCL communicators that share the same `hsn` interface on rank 0. That isolates "TRL is misusing NCCL" from "this cluster can't do what TRL asks." Branch from there:

- Tests pass → bug is in TRL/vLLM; explore `vllm_server_use_ipv6`, alternate group_port, newer TRL master, or move weight sync off NCCL entirely
- Tests fail → ask `<redacted: HPC vendor support>` for libfabric/CXI provider settings to allow concurrent communicator groups on the same NIC

## Update: cluster is innocent

[`debug/dual_nccl_repro.py`](../../debug/dual_nccl_repro.py) (2-node, 8 ranks, torch.distributed) confirms cross-node NCCL with two concurrent communicators **works**:

```
[rank 0] test1 (default 8-rank cross-node AllReduce)  → 3.19s ✅
[rank 0] test2 (secondary 8-rank cross-node AllReduce) → 1.08s ✅
[rank 0] test3 (subset 2-rank AllReduce)              → 0.08s ✅
```

So the cluster supports concurrent NCCL communicators. The hang is specifically inside vLLM's `PyNcclCommunicator` / `StatelessProcessGroup` code path used by TRL — *not* torch.distributed.

Likely culprits (in order of suspicion):

1. `StatelessProcessGroup` uses a raw `TCPStore` on `(host=vllm_node, port=group_port=51216 default)`. The rendezvous works (we see `Init COMPLETE`), but the resulting NCCL communicator runs over `hsn` and may not pick up the right local interface on rank 0 when torch.distributed has already claimed it.
2. `PyNcclCommunicator` may not honor `NCCL_SOCKET_IFNAME=hsn` the way torch.distributed does, leading to a different network selection that can't actually move bytes.
3. TRL 1.2.0's `init_communicator` HTTP call may race with vLLM's internal NCCL setup if the trainer had a partially-closed previous communicator.

Concrete next experiments:

- Write a vLLM-specific reproducer that uses `StatelessProcessGroup` + `PyNcclCommunicator` directly cross-node. If this hangs without TRL, the bug is purely in vLLM's wrapper.
- Try newer TRL/vLLM versions to see if upstream fixed this.
- Patch TRL's `init_communicator` to use `torch.distributed.new_group` (with a TCPStore on vLLM host) instead of `PyNcclCommunicator`.

## Update 2: even the mixed pattern works in isolation

[`debug/vllm_pynccl_repro.py`](../../debug/vllm_pynccl_repro.py) and [`debug/mixed_nccl_repro.py`](../../debug/mixed_nccl_repro.py) both pass:

- Cross-node `StatelessProcessGroup` + `PyNcclCommunicator` AllReduce → works fine (0.01s).
- `torch.distributed` 8-rank intra-trainer + `PyNcclCommunicator` 3-rank vLLM↔trainer concurrently on the same rank 0 → both AllReduces complete cleanly.

So the failure is **not** the multi-group NCCL pattern itself. The hypothesis pivots to: **the hang is on the vLLM server side**. The vLLM workers run their own internal TP NCCL group; when `init_communicator` arrives, they try to *also* join a second NCCL group (vLLM↔trainer) via `PyNcclCommunicator` while the TP NCCL group is still active. Two concurrent NCCL groups on the same process where one was *not* opened via torch.distributed may not work in vLLM's wiring.

Test plan: rerun the 2-node trainer with **`TP=1` vLLM** (single GPU vLLM, no internal TP NCCL group). If that works, we've isolated the bug to vLLM's TP-worker multi-group handling.

## Update 3: the failure is on the vLLM Worker side

With a vLLM 202602 (TP=2) + 2-node trainer 202603, capturing vLLM's `vllm.out` shows that **vLLM's `Worker_TP0` subprocess itself raises**:

```
(Worker_TP0 pid=227742) RuntimeError: NCCL error: internal error - please report this issue to the NCCL developers
  at trl/scripts/vllm_serve.py:111 → PyNcclCommunicator(pg, device=self.device)
  at vllm/distributed/device_communicators/pynccl.py:144 → self.all_reduce(data)
  at vllm/distributed/device_communicators/pynccl_wrapper.py:429 → ncclAllReduce
```

The trainer side then receives `Error encountered progressing operation=Connect, res=3, closing connection` because the remote (vLLM Worker) has crashed. So the "hang" we previously saw is actually the trainer waiting forever for a vLLM worker that has already died — except in some configurations (TP=4 3B) the error gets eaten and only the hang manifests.

Reproducer summary table:

| Setup                                                    | Outcome                |
|----------------------------------------------------------|------------------------|
| 1-node trainer + 1-node vLLM (any TP)                    | ✅ works               |
| 2-node trainer + 1-node vLLM (any TP)                    | ❌ vLLM Worker NCCL "internal error" → trainer connect fails |
| Pure-Python dual NCCL group via torch.distributed (2-node) | ✅ works               |
| Pure-Python PyNcclCommunicator-only cross-node (2-node)  | ✅ works               |
| torch.distributed + PyNcclCommunicator on same rank 0 (3-node) | ✅ works         |

Conclusion: the bug is **specific to vLLM Worker subprocess + `init_communicator` from a multi-node trainer**. The cluster, libfabric/CXI, NCCL, torch.distributed, and PyNcclCommunicator all work fine in every controlled variant — the failure mode only manifests inside vLLM's `multiproc_executor.py` worker_busy_loop path. Most likely cause: vLLM Worker's CUDA context / NCCL state setup conflicts with creating a new NCCL communicator when the peer is multi-node.

## Decision

Park multi-node trainer for now and stay on **1-node trainer + 1-node vLLM** (Layout A from `003_summary_and_test_plan.md`) for productive iteration. Multi-node trainer remains blocked on a vLLM/TRL upstream fix or a meaningful workaround (likely options: switch vLLM weight-sync from NCCL to HTTP file-based, or wait for a newer TRL/vLLM release that addresses this).

Artefacts left for the next person:

- [`debug/dual_nccl_repro.py`](../../debug/dual_nccl_repro.py) + [`debug/slurm_dual_nccl_repro.sbatch`](../../debug/slurm_dual_nccl_repro.sbatch)
- [`debug/vllm_pynccl_repro.py`](../../debug/vllm_pynccl_repro.py) + [`debug/slurm_vllm_pynccl_repro.sbatch`](../../debug/slurm_vllm_pynccl_repro.sbatch)
- [`debug/mixed_nccl_repro.py`](../../debug/mixed_nccl_repro.py) + [`debug/slurm_mixed_nccl_repro.sbatch`](../../debug/slurm_mixed_nccl_repro.sbatch)
- [`debug/slurm_build_nccl_tests.sbatch`](../../debug/slurm_build_nccl_tests.sbatch) — builds NVIDIA's nccl-tests binaries at `~/src/nccl-tests/build/`
