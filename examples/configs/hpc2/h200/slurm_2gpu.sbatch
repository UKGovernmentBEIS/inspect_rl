#!/bin/bash
#
# Combined 1-node sbatch for the h200 cluster:
#   - Reserves the whole node (--exclusive --gpus-per-node=8) so we don't
#     collide with other tenants on this cluster (Slurm here does not
#     enforce single-tenant GPU allocation; partial allocations OOM).
#   - Launches vLLM (GPU 0) and a single-process trainer (GPU 1) as two
#     concurrent srun steps with --overlap, so they share the allocation.
#   - After trainer exits, vLLM stays alive until the job's wall-clock
#     limit or `scancel`. You can `scancel` just the trainer step or
#     relaunch it with `srun --jobid=... --overlap ...` while keeping
#     vLLM warm — no ~7 min cold start on retry.
#
# Usage:
#   sbatch examples/configs/hpc2/h200/slurm_2gpu.sbatch
#   tail -F irl_output_slurm/latest/{master,vllm,trainer}.out
#
#   # Re-run the trainer against the same vLLM (vLLM stays warm):
#   JOB=<sbatch_job_id>
#   RD=irl_output_slurm/latest
#   srun --jobid=$JOB --overlap --exact --ntasks=1 --gpus=1 --cpus-per-task=8 \
#        --mem=32G --output=$RD/trainer_retry.out \
#        bash -c 'export CUDA_VISIBLE_DEVICES=1; cd '"$PWD"'; \
#                 uv run --no-sync irl train --devices 0 tldr \
#                     --model Qwen/Qwen2.5-0.5B-Instruct \
#                     --vllm-base-url http://$(hostname):8000 \
#                     --max-steps 2 --per-device-train-batch-size 2 --num-generations 4'
#
# Overrides:
#   env -u MODEL sbatch \
#       --export=ALL,MODEL=Qwen/Qwen2.5-0.5B-Instruct,EXAMPLE=tldr,MAX_STEPS=2 \
#       examples/configs/hpc2/h200/slurm_2gpu.sbatch
#
#SBATCH --job-name=irl-h200
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null

set -xeuo pipefail

STAMP=$(date +%Y-%m-%d-%H-%M-%S)
RUN_DIR="irl_output_slurm/${STAMP}_slurm_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
exec 1>>"$RUN_DIR/master.out" 2>&1
ln -sfn "$(basename $RUN_DIR)" irl_output_slurm/latest
touch "$RUN_DIR/vllm.out" "$RUN_DIR/trainer.out"

NODE=$(hostname)
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_URL="http://${NODE}:${VLLM_PORT}"
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
EXAMPLE=${EXAMPLE:-tldr}
MAX_STEPS=${MAX_STEPS:-2}
# WANDB=1 → pass --wandb to the trainer. Needs WANDB_API_KEY (and
# typically WANDB_ENTITY, WANDB_PROJECT, WANDB_BASE_URL) in the submitter's
# shell so they propagate via sbatch --export=ALL.
WANDB=${WANDB:-}
WANDB_FLAG=$([ -n "$WANDB" ] && echo "--wandb" || echo "")

echo "rundir:    $RUN_DIR/"
echo "node:      $NODE  (exclusive, 8 GPUs)"
echo "url:       $VLLM_URL"
echo "model:     $MODEL"
echo "example:   $EXAMPLE"
echo "max-steps: $MAX_STEPS"
echo "job-id:    $SLURM_JOB_ID"

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TRANSFORMERS_OFFLINE=1

# ── vLLM step: GPU 0, background, persistent ────────────────────────────────
srun --overlap --exact --ntasks=1 --cpus-per-task=8 --mem=64G \
     --output="$RUN_DIR/vllm.out" \
     bash -c "
        export CUDA_VISIBLE_DEVICES=0
        export HF_HOME=$HF_HOME
        export TRANSFORMERS_OFFLINE=1
        cd $SLURM_SUBMIT_DIR
        uv run --no-sync irl serve '$MODEL' \
            --tensor-parallel-size 1 \
            --devices 0 \
            --gpu-memory-utilization 0.9
     " &
VLLM_SRUN_PID=$!
echo "vllm srun pid: $VLLM_SRUN_PID"

# Watchdog: warn in master.out whenever trainer.out is idle > IDLE_WARN_S.
# Useful during the long pre-train phase (imports → tokenizer load → model
# load → NCCL handshake) which has no built-in heartbeat. The trainer
# itself emits `[heartbeat]` lines every 30 s once training starts.
# Started AFTER the trainer srun launches (otherwise it fires on the
# 0-byte placeholder file during vllm warmup).
IDLE_WARN_S=${IDLE_WARN_S:-60}
WATCHDOG_PID=""
start_watchdog() {
    local log=$1 warn=$2
    (
        local last_warn=0 mtime now elapsed
        # Wait until the log has at least one byte (trainer has started writing)
        while [ ! -s "$log" ]; do sleep 5; done
        while true; do
            sleep 15
            mtime=$(stat -c %Y "$log" 2>/dev/null || echo 0)
            now=$(date +%s)
            elapsed=$(( now - mtime ))
            if [ "$elapsed" -ge "$warn" ] && [ "$elapsed" -ge "$(( last_warn + 30 ))" ]; then
                echo "[watchdog] trainer.out idle ${elapsed}s — process still alive, no new output"
                last_warn=$elapsed
            fi
        done
    ) &
    WATCHDOG_PID=$!
}

trap 'echo "[cleanup] killing vllm srun pid=$VLLM_SRUN_PID, watchdog pid=$WATCHDOG_PID"; [ -n "$WATCHDOG_PID" ] && kill $WATCHDOG_PID 2>/dev/null || true; kill $VLLM_SRUN_PID 2>/dev/null || true' EXIT

echo "Waiting for vLLM at $VLLM_URL ..."
for i in $(seq 1 240); do
    if curl -sf --max-time 2 "$VLLM_URL/health/" >/dev/null 2>&1; then
        echo "vLLM ready after ${i}x5s: $VLLM_URL"
        break
    fi
    if ! kill -0 "$VLLM_SRUN_PID" 2>/dev/null; then
        echo "vLLM srun exited prematurely — tail of $RUN_DIR/vllm.out:"
        tail -40 "$RUN_DIR/vllm.out" || true
        exit 1
    fi
    sleep 5
done
curl -sf --max-time 2 "$VLLM_URL/health/" >/dev/null 2>&1 || {
    echo "vLLM not ready after 20 min — see $RUN_DIR/vllm.out"
    exit 1
}

# ── Trainer step: GPU 1, foreground (so the sbatch waits on it) ─────────────
# `irl train --devices 0` is correct here: CUDA_VISIBLE_DEVICES=1 narrows
# the trainer's view to a single GPU which it then sees as index 0.
echo "Launching trainer step (GPU 1) ..."
start_watchdog "$RUN_DIR/trainer.out" "$IDLE_WARN_S"
# Pipe trainer stdout/stderr through `ts` (or awk) to prefix every line
# with a timestamp — makes long gaps obvious when scanning the log.
srun --overlap --exact --ntasks=1 --cpus-per-task=8 --mem=32G \
     --output="$RUN_DIR/trainer.out" \
     bash -c "
        export CUDA_VISIBLE_DEVICES=1
        export HF_HOME=$HF_HOME
        export TRANSFORMERS_OFFLINE=1
        export INSPECT_RL_RUN_DIR=$SLURM_SUBMIT_DIR/$RUN_DIR
        # PYTHONUNBUFFERED=1 + TRL_EXPERIMENTAL_SILENCE=1 keep trainer.out
        # streaming live (Python defaults to block-buffer when stdout is a
        # pipe, which makes the output appear silent until the step ends).
        export PYTHONUNBUFFERED=1
        export TRL_EXPERIMENTAL_SILENCE=1
        cd $SLURM_SUBMIT_DIR
        stamp() { awk '{ print strftime(\"[%H:%M:%S]\"), \$0; fflush() }'; }
        echo \"[\$(date +%H:%M:%S)] trainer step starting on \$(hostname)\"
        # --devices 1 = physical GPU 1 (vllm has GPU 0). DO NOT change to 0;
        # the irl CLI overwrites CUDA_VISIBLE_DEVICES with this value, so
        # 0 would land the trainer on vllm's GPU and trip
        # \"same CUDA device for multiple ranks within same communicator\".
        uv run --no-sync irl train --devices 1 \
            $EXAMPLE \
            --model '$MODEL' \
            --vllm-base-url '$VLLM_URL' \
            --max-steps $MAX_STEPS \
            --per-device-train-batch-size 4 \
            --num-generations 4 \
            $WANDB_FLAG \
            2>&1 | stamp
        echo \"[\$(date +%H:%M:%S)] trainer step exited with rc=\${PIPESTATUS[0]}\"
     " || echo "trainer step exited non-zero (see $RUN_DIR/trainer.out)"

echo "Trainer step finished. vLLM stays running until job time-limit or scancel."
echo "To re-run trainer without restarting vLLM:"
echo "  srun --jobid=$SLURM_JOB_ID --overlap --exact --ntasks=1 --gpus=1 \\"
echo "       --cpus-per-task=8 --mem=32G --output=$RUN_DIR/trainer_retry.out \\"
echo "       bash -c 'export CUDA_VISIBLE_DEVICES=1; cd $SLURM_SUBMIT_DIR; \\"
echo "                uv run --no-sync irl train --devices 0 $EXAMPLE \\"
echo "                    --model $MODEL --vllm-base-url $VLLM_URL \\"
echo "                    --max-steps $MAX_STEPS --per-device-train-batch-size 2 --num-generations 4'"

wait "$VLLM_SRUN_PID"
