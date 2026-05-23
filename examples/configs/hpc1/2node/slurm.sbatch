#!/bin/bash
#
# 2-node combined run: 1 vLLM node (1 GPU, TP=1) + 1 trainer node (4 GPUs DDP).
#
# Smallest topology that exercises the "vLLM on separate node + multi-GPU
# trainer" path. Use this as the entry-point smoke test for any change to
# the trainer↔vLLM weight-sync plumbing.
#
# vLLM and trainer run as concurrent srun steps within ONE Slurm job — this
# is required on Slingshot/CXI clusters to share the job's VNI namespace.
# NCCL_NET=Socket forces NCCL to bypass aws-ofi-nccl (which trips a
# VNI_NOT_FOUND failure with ≥2 trainer peers in the address-vector — see
# journal/008_multinode/006).
#
# Usage:
#   env -u MODEL sbatch --export=ALL,NCCL_NET=Socket,MODEL=Qwen/Qwen2.5-0.5B-Instruct,EXAMPLE=tldr,MAX_STEPS=2 \
#       examples/configs/hpc1/2node/slurm.sbatch
#
# Debug mode (NCCL_DEBUG=TRACE + libfabric trace + py-spy sidecar):
#   env -u MODEL sbatch --export=ALL,IRL_DEBUG=1,NCCL_NET=Socket,MODEL=...,EXAMPLE=tldr,MAX_STEPS=2 \
#       examples/configs/hpc1/2node/slurm.sbatch
#
#SBATCH --job-name=irl-combined-2node
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null

set -euo pipefail

STAMP=$(date +%Y-%m-%d-%H-%M-%S)
RUN_DIR="irl_output_slurm/${STAMP}_slurm_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
exec 1>>"$RUN_DIR/master.out" 2>&1
ln -sfn "$(basename $RUN_DIR)" irl_output_slurm/latest
touch "$RUN_DIR/vllm.out" "$RUN_DIR/trainer-0.out"

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
VLLM_NODE="${NODES[0]}"
TRAINER_NODE="${NODES[1]}"
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
TRAINER_RANKS=$GPUS_PER_NODE

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
EXAMPLE=${EXAMPLE:-tldr}
MAX_STEPS=${MAX_STEPS:-}
OFF_POLICY_STEPS=${OFF_POLICY_STEPS:-0}
WANDB=${WANDB:-0}
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_URL="http://${VLLM_NODE}:${VLLM_PORT}"

# Debug mode (opt-in): turns on NCCL/libfabric trace + py-spy sidecar.
if [ "${IRL_DEBUG:-${DEBUG:-0}}" = "1" ]; then
    export NCCL_DEBUG=TRACE
    export NCCL_DEBUG_SUBSYS=ALL
    export NCCL_ASYNC_ERROR_HANDLING=1
    export FI_LOG_LEVEL=debug
    export FI_LOG_PROV=cxi
    export TORCH_DISTRIBUTED_DEBUG=DETAIL
    export TORCH_NCCL_TRACE_BUFFER_SIZE=65536
    echo "[debug] IRL_DEBUG=1 → NCCL_DEBUG=TRACE, NCCL_DEBUG_SUBSYS=ALL, FI_LOG_LEVEL=debug, FI_LOG_PROV=cxi"
fi

TRAIN_EXTRA="--model $MODEL"
[ -n "$MAX_STEPS" ] && TRAIN_EXTRA="$TRAIN_EXTRA --max-steps $MAX_STEPS"
[ -n "$OFF_POLICY_STEPS" ] && TRAIN_EXTRA="$TRAIN_EXTRA --off-policy-steps $OFF_POLICY_STEPS"
[ "$WANDB" = "1" ] && TRAIN_EXTRA="$TRAIN_EXTRA --wandb"

echo "rundir:        $RUN_DIR/"
echo "vllm node:     $VLLM_NODE  (TP=1, 1 GPU)"
echo "trainer node:  $TRAINER_NODE  ($GPUS_PER_NODE GPUs DDP)"
echo "model:         $MODEL"
echo "example:       $EXAMPLE"
echo "vllm url:      $VLLM_URL"

# py-spy sidecar — only used when IRL_DEBUG=1. Defined before either srun step.
PYSPY_SCRIPT="$RUN_DIR/pyspy_sidecar.sh"
cat > "$PYSPY_SCRIPT" <<'PYSPY_EOF'
#!/bin/bash
RUN_DIR="$1"; CWD="$2"; PATTERN="$3"
cd "$CWD"
sleep 30
while true; do
    PID=$(pgrep -f "$PATTERN" | head -1)
    if [ -z "$PID" ]; then sleep 10; continue; fi
    {
        printf '\n--- %s pid=%s ---\n' "$(date +%H:%M:%S)" "$PID"
        uv run --no-sync py-spy dump --pid "$PID" --nonblocking 2>&1 || true
    } >> "$RUN_DIR/pyspy-$(hostname -s).txt"
    sleep 30
done
PYSPY_EOF
chmod +x "$PYSPY_SCRIPT"

# -----------------------------------------------------------------------
# Step 1: vLLM on node 0, 1 GPU, TP=1.
# -----------------------------------------------------------------------
srun --nodes=1 --ntasks=1 --nodelist="$VLLM_NODE" --gpus-per-node="$GPUS_PER_NODE" \
    --output="$RUN_DIR/vllm.out" --export=ALL \
    bash -c "
        cd $SLURM_SUBMIT_DIR
        echo "[sbatch] NOTE: add 'module load' for your cluster's NCCL/OFI modules here if cross-node NCCL requires them"
        export CUDA_VISIBLE_DEVICES=0
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        export CUDA_HOME=\${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda}
        # Re-apply env knobs AFTER module load (some NCCL modules reset
        # NCCL_NET and NCCL_DEBUG — override them back here).
        # Outer-shell single-quoted expansion baked in here.
        [ -n '${NCCL_DEBUG:-}' ] && export NCCL_DEBUG='${NCCL_DEBUG:-}'
        [ -n '${NCCL_DEBUG_SUBSYS:-}' ] && export NCCL_DEBUG_SUBSYS='${NCCL_DEBUG_SUBSYS:-}'
        [ -n '${NCCL_NET:-}' ] && export NCCL_NET='${NCCL_NET:-}'
        [ -n '${NCCL_ASYNC_ERROR_HANDLING:-}' ] && export NCCL_ASYNC_ERROR_HANDLING='${NCCL_ASYNC_ERROR_HANDLING:-}'
        [ -n '${FI_LOG_LEVEL:-}' ] && export FI_LOG_LEVEL='${FI_LOG_LEVEL:-}'
        [ -n '${FI_LOG_PROV:-}' ] && export FI_LOG_PROV='${FI_LOG_PROV:-}'
        echo \"[vllm \$(hostname)] starting; knobs:\" \$(env | grep -E '^(NCCL_DEBUG|NCCL_NET|FI_LOG)' | tr '\\n' ' ')
        if [ \"\${IRL_DEBUG:-0}\" = \"1\" ]; then
            bash $PYSPY_SCRIPT $RUN_DIR $SLURM_SUBMIT_DIR 'trl.scripts.vllm_serve|vllm-serve|EngineCore' &
        fi
        uv run --no-sync irl serve '$MODEL' \
            --tensor-parallel-size 1 \
            --devices 0 \
            --gpu-memory-utilization 0.9
    " &
VLLM_PID=$!
echo "vllm srun PID: $VLLM_PID"

cleanup() {
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

# -----------------------------------------------------------------------
# Step 2: wait for vLLM to be ready.
# -----------------------------------------------------------------------
echo "Waiting for vLLM at $VLLM_URL ..."
for i in $(seq 1 240); do
    if curl -sf "$VLLM_URL/health/" >/dev/null 2>&1; then
        echo "vLLM ready after ${i}s: $VLLM_URL"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM srun exited prematurely — see $RUN_DIR/vllm.out"
        exit 1
    fi
    sleep 5
done
if ! curl -sf "$VLLM_URL/health/" >/dev/null 2>&1; then
    echo "vLLM not ready after 20 min — aborting"
    exit 1
fi

# -----------------------------------------------------------------------
# Step 3: trainer on node 1, 4 GPUs DDP (accelerate single-node config).
# -----------------------------------------------------------------------
srun --nodes=1 --ntasks=1 --nodelist="$TRAINER_NODE" --gpus-per-node="$GPUS_PER_NODE" \
    --output="$RUN_DIR/trainer-0.out" --export=ALL \
    bash -c "
        cd $SLURM_SUBMIT_DIR
        echo "[sbatch] NOTE: add 'module load' for your cluster's NCCL/OFI modules here if cross-node NCCL requires them"
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        export CUDA_HOME=\${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda}
        [ -n '${NCCL_DEBUG:-}' ] && export NCCL_DEBUG='${NCCL_DEBUG:-}'
        [ -n '${NCCL_DEBUG_SUBSYS:-}' ] && export NCCL_DEBUG_SUBSYS='${NCCL_DEBUG_SUBSYS:-}'
        [ -n '${NCCL_NET:-}' ] && export NCCL_NET='${NCCL_NET:-}'
        [ -n '${NCCL_ASYNC_ERROR_HANDLING:-}' ] && export NCCL_ASYNC_ERROR_HANDLING='${NCCL_ASYNC_ERROR_HANDLING:-}'
        [ -n '${FI_LOG_LEVEL:-}' ] && export FI_LOG_LEVEL='${FI_LOG_LEVEL:-}'
        [ -n '${FI_LOG_PROV:-}' ] && export FI_LOG_PROV='${FI_LOG_PROV:-}'
        [ -n '${TORCH_DISTRIBUTED_DEBUG:-}' ] && export TORCH_DISTRIBUTED_DEBUG='${TORCH_DISTRIBUTED_DEBUG:-}'
        [ -n '${TORCH_NCCL_TRACE_BUFFER_SIZE:-}' ] && export TORCH_NCCL_TRACE_BUFFER_SIZE='${TORCH_NCCL_TRACE_BUFFER_SIZE:-}'
        export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300
        export WANDB_API_KEY=\${WANDB_API_KEY:-}
        export WANDB_BASE_URL=\${WANDB_BASE_URL:-}
        # Tell create_run_dir to reuse this directory (skip its own timestamping)
        # so trainer artifacts land alongside slurm stdouts.
        export INSPECT_RL_RUN_DIR=$SLURM_SUBMIT_DIR/$RUN_DIR
        echo \"[trainer \$(hostname)] starting accelerate; knobs:\" \$(env | grep -E '^(NCCL_DEBUG|NCCL_NET|FI_LOG)' | tr '\\n' ' ')
        if [ \"\${IRL_DEBUG:-0}\" = \"1\" ]; then
            bash $PYSPY_SCRIPT $RUN_DIR $SLURM_SUBMIT_DIR 'inspect_rl.cli' &
        fi
        uv run --no-sync accelerate launch \
            --config-file examples/configs/hpc1/2node/accelerate.yaml \
            -m inspect_rl.cli train $EXAMPLE \
            --vllm-base-url $VLLM_URL \
            --per-device-train-batch-size 2 \
            --num-generations 4 $TRAIN_EXTRA
    "

echo "Trainer step done."
