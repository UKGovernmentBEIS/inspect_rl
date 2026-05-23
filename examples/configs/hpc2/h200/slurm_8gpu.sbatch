#!/bin/bash
#
# Combined 1-node sbatch for the h200 cluster, full-node variant:
#   - Reserves the whole node (--exclusive --gpus-per-node=8).
#   - vLLM on GPU 0 (TP=1), 7-rank DDP trainer on GPUs 1-7 via accelerate.
#   - Saturates the exclusive reservation (the slurm_2gpu.sbatch sibling
#     leaves 6 GPUs idle); use this when you actually want the parallelism.
#
# Trainer is launched as `accelerate launch --num_processes 7 -m
# inspect_rl.cli train …` rather than `irl train`, because `irl train
# --num-processes N` is rejected with a redirect (see top-level README
# "Multi-GPU and multi-node"). All 7 ranks attach to the single vLLM
# server; only rank 0 issues requests to it (see journal/008_multinode/
# 002_rank_gating.md).
#
# Usage:
#   env -u MODEL WANDB=1 sbatch examples/configs/hpc2/h200/slurm_8gpu.sbatch
#   tail -F irl_output_slurm/latest/{master,vllm,trainer}.out
#
# Overrides via --export=ALL,MODEL=...,EXAMPLE=...,MAX_STEPS=...,WANDB=1
# (same env knobs as slurm.sbatch).
#
#SBATCH --job-name=irl-h200-8gpu
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
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
NUM_TRAIN_PROCESSES=${NUM_TRAIN_PROCESSES:-7}
WANDB=${WANDB:-}
WANDB_FLAG=$([ -n "$WANDB" ] && echo "--wandb" || echo "")

echo "rundir:    $RUN_DIR/"
echo "node:      $NODE  (exclusive, 8 GPUs)"
echo "topology:  vLLM on GPU 0 (TP=1), trainer DDP on GPUs 1-$NUM_TRAIN_PROCESSES"
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

IDLE_WARN_S=${IDLE_WARN_S:-90}
WATCHDOG_PID=""
start_watchdog() {
    local log=$1 warn=$2
    (
        local last_warn=0 mtime now elapsed
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

# ── Trainer step: GPUs 1..NUM_TRAIN_PROCESSES, DDP via accelerate ───────────
echo "Launching trainer step (GPUs 1-$NUM_TRAIN_PROCESSES, $NUM_TRAIN_PROCESSES-rank DDP)..."
start_watchdog "$RUN_DIR/trainer.out" "$IDLE_WARN_S"

# Build "1,2,...,N" device list
TRAIN_DEVICES=$(seq -s, 1 "$NUM_TRAIN_PROCESSES")

srun --overlap --exact --ntasks=1 --cpus-per-task=8 --mem=256G \
     --output="$RUN_DIR/trainer.out" \
     bash -c "
        export CUDA_VISIBLE_DEVICES=$TRAIN_DEVICES
        export HF_HOME=$HF_HOME
        export TRANSFORMERS_OFFLINE=1
        export INSPECT_RL_RUN_DIR=$SLURM_SUBMIT_DIR/$RUN_DIR
        export PYTHONUNBUFFERED=1
        export TRL_EXPERIMENTAL_SILENCE=1
        cd $SLURM_SUBMIT_DIR
        stamp() { awk '{ print strftime(\"[%H:%M:%S]\"), \$0; fflush() }'; }
        echo \"[\$(date +%H:%M:%S)] trainer step starting on \$(hostname) (CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES)\"
        # accelerate launch wraps the irl CLI for multi-rank DDP. irl train
        # --num-processes is rejected — multi-process must go via accelerate.
        uv run --no-sync accelerate launch --num_processes $NUM_TRAIN_PROCESSES \
            -m inspect_rl.cli train \
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
wait "$VLLM_SRUN_PID"
