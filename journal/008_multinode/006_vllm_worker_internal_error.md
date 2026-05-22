# 006 — vLLM Worker-side `NCCL internal error` from a multi-node trainer

## TL;DR

A multi-node trainer + a separate single-node vLLM hangs / crashes inside
vLLM's `Worker_TPx` subprocess at `PyNcclCommunicator.__init__`'s warmup
AllReduce. None of the four standalone NCCL reproducers — including a new
one that puts **two PyNcclCommunicator instances on the same CUDA device in
the same process** — fail. The failure is specific to a real vLLM Worker
subprocess receiving `init_communicator` from a trainer with `world_size > 1`
in its intra-trainer NCCL group.

This page picks up where [005](./005_trainer_nccl_hang.md) left off and
walks methodically toward bottoming out the bug.

## How TRL uses vLLM for weight sync

Source: [`trl/scripts/vllm_serve.py:49-114`](../../.venv/lib/python3.12/site-packages/trl/scripts/vllm_serve.py).

```python
class WeightSyncWorkerExtension:
    def init_communicator(self, host, port, world_size, client_device_uuid):
        ...
        rank = get_world_group().rank
        pg = StatelessProcessGroup.create(host=host, port=port,
                                          rank=rank, world_size=world_size)
        self.communicator = PyNcclCommunicator(pg, device=self.device)
        self.client_rank = world_size - 1
```

When the trainer calls `POST /init_communicator/`, TRL `collective_rpc`s this
method into every vLLM Worker subprocess. Each Worker then constructs a
`PyNcclCommunicator` over a `StatelessProcessGroup` rendezvoused via TCP on
the vLLM host. The trainer's client rank is the last rank
(`world_size - 1`) and joins via the same rendezvous.

`PyNcclCommunicator.__init__`
([`vllm/distributed/device_communicators/pynccl.py:137-146`](../../.venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/pynccl.py))
runs a warmup AllReduce on the new communicator. **That is where the
failure surfaces.** With a single-node trainer the AllReduce completes; with
a multi-node trainer (any `world_size > 1` intra-trainer group) the Worker
either hangs or raises `NCCL error: internal error - please report this issue
to the NCCL developers`.

## NCCL state inside one vLLM Worker subprocess

Three concurrent NCCL states coexist in one Worker process:

1. **`torch.distributed` NCCL backend** (vLLM's `_WORLD` group), set up by
   `init_distributed_environment`
   ([`parallel_state.py:1417`](../../.venv/lib/python3.12/site-packages/vllm/distributed/parallel_state.py)).
2. **vLLM-TP `PyNcclCommunicator`**, built by `CudaCommunicator` from a
   per-group gloo subgroup
   ([`cuda_communicator.py:75-78`](../../.venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/cuda_communicator.py)).
   Used for TP allreduces during model forward passes.
3. **TRL weight-sync `PyNcclCommunicator`**, built from a
   `StatelessProcessGroup` ([above](#how-trl-uses-vllm-for-weight-sync)).
   Created lazily on the first `init_communicator/` call.

(2) and (3) share the same CUDA device on the Worker. The PyNccl class
docstring warns: *"It is the caller's responsibility to make sure each
communicator is bind to a unique device."* But this is violated in *every*
TRL+vLLM run, including the 1-node trainer case that does work — so binding
two communicators to one device cannot itself be the bug.

## Reproducer landscape

| Reproducer | Trainer-side state | vLLM-side state | Result |
|---|---|---|---|
| `debug/dual_nccl_repro.py` | `torch.distributed` × 2 groups | (none on a different process) | ✅ passes |
| `debug/vllm_pynccl_repro.py` | none | `StatelessProcessGroup` + `PyNcclCommunicator` only | ✅ passes |
| `debug/mixed_nccl_repro.py` | `torch.distributed` 8-rank + secondary `PyNcclCommunicator` on rank 0 | secondary `PyNcclCommunicator` only | ✅ passes |
| `debug/double_pynccl_repro.py` (NEW) | `torch.distributed` 8-rank + secondary | **`torch.distributed` + gloo subgroup + TP `PyNcclCommunicator` + secondary `PyNcclCommunicator`** | ✅ passes (2026-05-13, job 202604) |
| Real TRL + vLLM, 2-node trainer | accelerate-launched torch.distributed 8-rank, secondary | real vLLM Worker subprocess with model loaded | ❌ `NCCL error: internal error` (job 202602) |

The new `double_pynccl_repro.py` is the closest synthetic mirror of the real
case: vllm-side processes have all the same NCCL state a vLLM Worker has
(torch.distributed NCCL + a gloo subgroup + a PyNcclCommunicator built from
it), and additionally hold the secondary PyNcclCommunicator from the
StatelessProcessGroup. It passes. Therefore the bug is **not** any of:

- The cluster
- aws-ofi-nccl / CXI libfabric
- The number of PyNcclCommunicator instances per process
- `torch.distributed` NCCL + `PyNcclCommunicator` cohabitation
- Multi-node trainer participation in the secondary group

## Working hypotheses left standing

What's unique to real vLLM Workers that the repro doesn't cover:

1. **`VLLM_WORKER_MULTIPROC_METHOD=spawn` subprocess context.** vLLM Workers
   are spawned children of the vLLM API server. They inherit a CUDA / NCCL
   library state. Bootstrap-time DNS or interface resolution may differ
   between a spawned subprocess and a fresh srun task.
2. **Model loaded into CUDA memory.** ~0.5–7 GiB of weights + KV cache pool
   + custom kernels — different memory pressure, but more importantly
   different CUDA streams (vLLM uses non-default streams via `current_stream`
   in PyNcclCommunicator).
3. **CUDA streams.** The Worker process has CUDA graphs / compile state /
   model-execution streams active when `init_communicator` arrives. The
   warmup AllReduce inside `PyNcclCommunicator.__init__` uses
   `current_stream()`. The stream context inside a Worker may be different
   from a vanilla python process.
4. **NCCL options inherited from vLLM init.** Things like
   `NCCL_LAUNCH_MODE`, `NCCL_BLOCKING_WAIT`, `NCCL_ASYNC_ERROR_HANDLING`
   are sometimes set globally by vLLM and may not be set in our repros.

## Next experiments

- [ ] **TP=1 vLLM + 2-node trainer.** If this works, the bug is specifically
  the *combination* of vLLM-TP NCCL state with a cross-node secondary group.
  If it still fails, the bug lives elsewhere (subprocess context / streams).
- [ ] Capture **full `NCCL_DEBUG=INFO,NCCL_DEBUG_SUBSYS=INIT,COLL,NET,ENV`
  trace from the failing real Worker subprocess** and compare side-by-side
  with the working 1-node trainer + 1-node vLLM trace. Find the first
  divergent NCCL call.
- [ ] If TP=1 works, escalate-or-patch: patch `WeightSyncWorkerExtension`
  to destroy/recreate any conflicting NCCL state before constructing the
  secondary communicator.
- [ ] If TP=1 fails too, write a repro that loads a real vLLM engine and
  *then* runs the secondary PyNcclCommunicator from inside the Worker —
  smaller than full GRPO but bigger than today's repros.

## Update 1 (2026-05-13) — TP=1 vLLM ALSO fails, but with a different signature

Job 202605 (vLLM TP=1, 0.5B, devices=0) + job 202608 (2-node trainer):
`init_communicator` succeeds, `ncclCommInitRank` reports Init COMPLETE in
0.88s, channels set up via `NET/AWS Libfabric/0/GDRDMA`. Then on the first
warmup AllReduce, the OFI provider returns **`Inappropriate ioctl for
device`** on every SEND request:

```
<redacted-nid>:229905:230588 [0] NCCL INFO NET/OFI Initializing aws-ofi-nccl 1.8.1-aws
<redacted-nid>:229905:229905 [0] NCCL INFO NET/OFI Using Libfabric version 1.22
<redacted-nid>:229905:229905 [0] NCCL INFO NET/OFI Selected Provider is cxi (found 4 nics)
…
<redacted-nid>:229905:229905 [0] NCCL INFO ncclCommInitRank … rank 0 nranks 2 … Init COMPLETE
<redacted-nid>:229905:229905 [0] NCCL INFO AllReduce: opCount 0 … [nranks=2] stream (nil)
…
[2026-05-13 18:01:41] <redacted-nid>:229905:230588 [0] ofi_process_cq:196 NCCL WARN
    NET/OFI Request 0x400aa7dd3b98 completed with error.
    RC: 107. Error: Inappropriate ioctl for device.
    Completed length: 0, Request: { dev: 0, size: 0, state: CREATED, direction: SEND }
```

40+ identical SEND failures fire in a burst. Trainer side hangs at
`GRPOTrainer.__init__` because the vLLM Worker never returns from the warmup
AllReduce. (Note: the earlier "NCCL error: internal error" we saw in job
202602 with TP=2 was the *consequence* — NCCL wraps the underlying OFI
failure into a generic "internal error" once it has accumulated enough bad
completions.)

**Implication:** the failure is at the **libfabric / CXI / aws-ofi-nccl**
layer, not in NCCL, not in vLLM TP, not in any of the
`PyNcclCommunicator`-with-coexisting-NCCL-state scenarios. The hypotheses
list above moves down accordingly — the "TP=1 vLLM" experiment proves the
bug is **not** about vLLM-TP NCCL × secondary NCCL interaction. The failure
also does not need vLLM at all in the sense that it would also fail in any
process that runs the secondary `PyNcclCommunicator` AllReduce — *if* the
peer is a multi-node trainer.

## Update 2 — refined hypothesis (CXI MR/endpoint exhaustion)

`Inappropriate ioctl for device` (`ENOTTY`) returned from a CXI ioctl is
documented as the kernel module's response when an ioctl call references a
resource that doesn't exist or doesn't support the requested operation —
e.g. a memory-region key that wasn't allocated, an EP that isn't bound, or
an MR table slot that's exhausted.

The bug only manifests when the trainer's intra-trainer NCCL group is
multi-node, i.e. when that group has already established cross-node CXI EPs
+ MRs on the trainer rank 0 process. Adding a SECOND NCCL group (the
vLLM↔trainer secondary) then attempts a second round of EP / MR
allocations on the same `cxi0..cxi3` HCAs and runs out of something.

The `double_pynccl_repro.py` doesn't trigger this because its trainer-side
ranks only do an 8-rank intra-trainer AllReduce *once* and then move on;
NCCL releases the per-channel CXI EPs eagerly. The real trainer keeps the
intra-trainer NCCL group active across the entire training loop, so the
vLLM↔trainer secondary group has to coexist with held CXI resources.

If correct, the fix space is:

1. **Limit NCCL resource usage on trainer rank 0** so the secondary group
   can also fit. `NCCL_NCHANNELS_PER_NET_PEER=1`, lower `NCCL_BUFFSIZE`, or
   set `NCCL_MIN_NCHANNELS=1` may reduce the resource footprint of the
   intra-trainer group.
2. **Disable GDRDMA for the secondary group** with `NCCL_NET_GDR_LEVEL=0` so
   it falls back to host-staged buffers — fewer CXI MR slots required.
3. **Force the secondary group onto a different transport** with
   `NCCL_NET=Socket` (or aws-ofi-nccl's socket path). Slower but uses
   sockets, not CXI.
4. **Restart the trainer's NCCL group with a freed-MR config** before
   `init_communicator`. Tricky to implement cleanly inside accelerate.

## Next experiments (revised)

- [x] TP=1 vLLM + 2-node trainer — bug recurs at OFI level.
- [x] 1-node trainer + TP=1 vLLM on a fresh vLLM job — **works**
  (`[summary] 2 steps in 8.2s`, job 202611).
- [x] `NCCL_NET_GDR_LEVEL=0` on the 2-node trainer — does **not** fix
  (job 202612).
- [x] `NCCL_MAX_NCHANNELS=2,NCCL_MIN_NCHANNELS=1` — inconclusive on
  polluted vLLM, did not appear to help.
- [x] `NCCL_NET=Socket` — env propagation issue at first (NCCL_NET overwritten
  by the cluster aws-ofi-nccl module); fix in progress.
- [x] `FI_LOG_LEVEL=info FI_LOG_PROV=cxi` captured the libfabric/CXI
  trace — bottomed out (next section).

## Update 3 — CXI VNI_NOT_FOUND is the root cause

Job 202631 (2-node trainer, `FI_LOG_LEVEL=info FI_LOG_PROV=cxi`) trips the
following libfabric/cxi-provider warnings on the trainer side at the moment
of the failure:

```
libfabric:242959::cxi:av:cxip_av_insert_addr():114<warn>
    <redacted-nid>: nid=0x862 pid=0 inserted multiple times
libfabric:242959::cxi:ep_data:cxip_report_send_completion():675<warn>
    <redacted-nid>: TXC (0x8f0:4): Request dest_addr: 1 caddr.nic: 0X862 caddr.pid: 0
    error: 0x4008886603b0 (err: 107, VNI_NOT_FOUND)
```

(Full excerpt: [`006_fi_log_excerpt.txt`](006_fi_log_excerpt.txt) and
[`006_fi_log_vllm_excerpt.txt`](006_fi_log_vllm_excerpt.txt).)

`err: 107, VNI_NOT_FOUND` means the **CXI/Slingshot Virtual Network
Identifier** for the destination NIC is unknown to the source endpoint.
CXI VNIs are how Slurm partitions Slingshot traffic between jobs: each
Slurm job is allocated its own VNI namespace, and an endpoint in one job
cannot address a NIC in another job's namespace.

vLLM and the trainer run as **separate Slurm jobs** with separate VNI
allocations. Their NCCL secondary-group bootstrap exchanges nid/pid
addresses successfully (bootstrap uses TCP), but the actual SEND from
trainer→vLLM is rejected by the CXI driver: vLLM's NIC is not in the
trainer job's VNI map.

### Why 1-node trainer works in the same separate-jobs topology

With a single trainer node, the trainer's intra-DDP NCCL group is entirely
intra-node (NVLink-class transport) and never opens a cross-node CXI
endpoint. When `init_communicator` arrives, the secondary group's CXI EP is
the *first* one in this process — and aws-ofi-nccl's address-vector setup
inserts the vLLM NIC cleanly (no prior collision).

With ≥2 trainer nodes, the intra-DDP NCCL group has already opened CXI
endpoints with the *trainer* job's VNI. The secondary group then tries to
insert vLLM's NIC into the same AV (`cxip_av_insert_addr inserted multiple
times` warning), and the resulting EP doesn't have a route to vLLM's NIC
in the trainer job's VNI scope → SEND fails with `VNI_NOT_FOUND` → NCCL
surfaces as `Inappropriate ioctl for device`.

### Fix paths (in order of preference)

1. **Co-located Slurm job.** Put vLLM and the trainer in a *single* sbatch
   job, started as two `srun` steps. They share the job's VNI namespace,
   so the secondary group's SEND is permitted. Most surgical; doesn't
   change NCCL or vLLM code. (Costs one job-script restructure but keeps
   the "persistent vLLM" benefit if the script holds the vLLM step running
   while the trainer step iterates.)
2. **Cross-job VNI sharing via Slurm `--network`.** The Slingshot
   plugin may accept `--network=def_tles=X,def_cqs=Y,def_eqs=Z,def_vnis=N`
   or a shared-namespace flag — needs to be confirmed with
   `<redacted: HPC vendor support>` / docs.
3. **`NCCL_NET=Socket` on both sides.** Forces NCCL to use the built-in
   TCP transport, bypassing CXI/libfabric. Slow (no GDRDMA, no
   Slingshot) but should sidestep the VNI issue entirely. Currently
   blocked by the cluster aws-ofi-nccl module baking
   `NCCL_NET="AWS Libfabric"` into the env; needs us to `unset` it after
   module load or use `NCCL_NET_PLUGIN` override.

Recommended next step: prototype fix path 1 — a single sbatch that runs
vLLM and the 2-node trainer as concurrent `srun` steps within one job
allocation.

## Update 4 — same-job alone is NOT enough; `NCCL_NET=Socket` is needed too

`examples/configs/slurm_combined_vllm_trainer_3node.sbatch` runs vLLM and
the 2-node trainer as concurrent srun steps within one Slurm job. Even
so, the VNI_NOT_FOUND failure recurred:

```
<redacted-nid>: nid=0x862 pid=0 inserted multiple times
<redacted-nid>: TXC (0x8f0:4): Request dest_addr: 1 caddr.nic: 0X862 caddr.pid: 0
    error (err: 107, VNI_NOT_FOUND)
```

The damning detail is `caddr.pid: 0`. **aws-ofi-nccl always advertises pid=0
for every peer address**, so the libfabric AV table (which deduplicates by
(nid, pid)) collapses peer entries. With 1-node trainer there is only one
peer in the AV → no collision → works. With ≥2 trainer nodes the AV gets
multiple (nid, pid=0) tuples that collide; the EP's VNI lookup misses; the
SEND fails. Same-job VNI namespace doesn't help because the collision is
internal to a single process's libfabric AV.

## Update 5 — working recipe (validated 2026-05-13, job 202638)

**Bypass aws-ofi-nccl entirely by forcing NCCL's built-in TCP socket
transport.** Even though the cluster NCCL modules set `NCCL_NET="AWS
Libfabric"`, you can override it AFTER the module
load in the srun bash body — but the override must use *outer-shell
single-quoted expansion* so the value survives module-load reset:

```bash
# Inside srun's bash -c body, AFTER module load:
[ -n '${NCCL_NET:-}' ] && export NCCL_NET='${NCCL_NET:-}'
[ -n '${NCCL_DEBUG:-}' ] && export NCCL_DEBUG='${NCCL_DEBUG:-}'
# ... etc for every knob you want to forward.
```

Then submit with `NCCL_NET=Socket` (plus the same-job pattern):

```bash
env -u MODEL sbatch \
    --export=ALL,NCCL_NET=Socket,MODEL=Qwen/Qwen2.5-0.5B-Instruct,EXAMPLE=tldr,MAX_STEPS=2 \
    examples/configs/slurm_combined_vllm_trainer_3node.sbatch
```

Result (job 202638): **2-node trainer + 1-node vLLM completed 2 training
steps in 1m46s end-to-end**, weight sync `0.5–0.7s/step`, valid rewards
emitted, checkpoint saved. No VNI_NOT_FOUND, no Inappropriate ioctl, no
hang.

```
[step 1/2] tldr_reward=-178.81 | --
[step 1] weight update done in 0.7s (step total 9.7s)
[step 2/2] tldr_reward=-129.88 | 6.4s
[step 2] weight update done in 0.5s (step total 6.2s)
[step 2] checkpoint saved
```

### Costs and follow-ups

- **TCP socket transport is slower than Slingshot RDMA.** For 0.5B at 2-step smoke it's
  invisible (0.5–0.7s weight sync); for 7B+ it will be a bottleneck on
  every step's weight broadcast.
- The intra-trainer NCCL group ALSO uses Socket now (NCCL_NET is global
  per-process). For 8-rank intra-trainer this is a real cost.
- The pid=0-AV-collision underlying bug is in aws-ofi-nccl (or its CXI
  provider config). Worth filing with `<redacted: HPC vendor support>` /
  the aws-ofi-nccl project so RDMA can be re-enabled.
- Possible follow-ups to keep RDMA on the intra-trainer group only:
  per-process NCCL_NET wouldn't help (NCCL has only one transport at a
  time). Would need separate process groups via torch.distributed
  subgroups configured differently — non-trivial.

## Artifacts

- [`debug/double_pynccl_repro.py`](../../debug/double_pynccl_repro.py)
- [`debug/slurm_double_pynccl_repro.sbatch`](../../debug/slurm_double_pynccl_repro.sbatch)
- Job 202604: passing run, `debug/runs/latest_double_pynccl/`
- Job 202602: failing real vLLM run from 005, `slurm_logs/2026-05-13_17-32-54_202602/vllm.out`
