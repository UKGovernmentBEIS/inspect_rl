#!/bin/bash
#
# 8-node combined run: 1 vLLM node (4 GPUs, TP=4) + 7 trainer nodes
# (4 GPUs × 7 = 28 ranks, accelerate DeepSpeed ZeRO-3).
#
# Sized for Qwen2.5-72B-Instruct, but the topology auto-scales: --nodes=8
# computes TRAINER_NODE_COUNT=7 and TRAINER_RANKS=28 at job start. For 32B
# the 4-node config is fine; jump to this when ZeRO-3 per-rank memory under
# 12 ranks exceeds ~50 GB.
#
# Same NCCL_NET=Socket workaround as 4node/: aws-ofi-nccl's CXI provider
# trips VNI_NOT_FOUND with ≥2 trainer peers (journal/008_multinode/006).
# vLLM + trainer in one Slurm job to share the job's VNI namespace.
#
# Usage:
#   env -u MODEL sbatch --export=ALL,NCCL_NET=Socket,MODEL=Qwen/Qwen2.5-72B-Instruct,EXAMPLE=math-agent,MAX_STEPS=3 \
#       examples/configs/hpc1/8node/slurm.sbatch
#
#SBATCH --job-name=irl-combined-8node
#SBATCH --partition=workq
#SBATCH --nodes=8
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
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
TRAINER_NODES=("${NODES[@]:1}")
TRAINER_NODELIST=$(IFS=,; echo "${TRAINER_NODES[*]}")
HEAD_TRAINER="${TRAINER_NODES[0]}"
TRAINER_NODE_COUNT=${#TRAINER_NODES[@]}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
TRAINER_RANKS=$((TRAINER_NODE_COUNT * GPUS_PER_NODE))

MODEL=${MODEL:-Qwen/Qwen2.5-72B-Instruct}
EXAMPLE=${EXAMPLE:-tldr}
TP=${TP:-4}
DEVICES=${DEVICES:-0,1,2,3}
MAX_STEPS=${MAX_STEPS:-}
OFF_POLICY_STEPS=${OFF_POLICY_STEPS:-0}
RESAMPLE_ROUNDS=${RESAMPLE_ROUNDS:-}
WANDB=${WANDB:-0}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-examples/configs/hpc1/8node/accelerate_zero3.yaml}
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_URL="http://${VLLM_NODE}:${VLLM_PORT}"

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
[ -n "$RESAMPLE_ROUNDS" ] && TRAIN_EXTRA="$TRAIN_EXTRA --resample-rounds $RESAMPLE_ROUNDS"
[ "$WANDB" = "1" ] && TRAIN_EXTRA="$TRAIN_EXTRA --wandb"

echo "rundir:        $RUN_DIR/"
echo "vllm node:     $VLLM_NODE  (TP=$TP, devices=$DEVICES)"
echo "trainer nodes: $TRAINER_NODELIST  ($TRAINER_NODE_COUNT × $GPUS_PER_NODE GPUs = $TRAINER_RANKS ranks)"
echo "model:         $MODEL"
echo "example:       $EXAMPLE"
echo "vllm url:      $VLLM_URL"
echo "accelerate:    $ACCELERATE_CONFIG"

PYSPY_SCRIPT="$RUN_DIR/pyspy_sidecar.sh"
cat > "$PYSPY_SCRIPT" <<'PYSPY_EOF'
#!/bin/bash
LOGDIR="$1"; CWD="$2"; PATTERN="$3"
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
# Step 1: vLLM on node 0, TP=4 across all 4 GPUs.
# -----------------------------------------------------------------------
srun --nodes=1 --ntasks=1 --nodelist="$VLLM_NODE" --gpus-per-node="$GPUS_PER_NODE" \
    --output="$RUN_DIR/vllm.out" --export=ALL \
    bash -c "
        cd $SLURM_SUBMIT_DIR
        echo "[sbatch] NOTE: add 'module load' for your cluster's NCCL/OFI modules here if cross-node NCCL requires them"
        export CUDA_VISIBLE_DEVICES=$DEVICES
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        export CUDA_HOME=\${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda}
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
            --tensor-parallel-size $TP \
            --devices $DEVICES \
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
# Step 2: wait for vLLM to be ready. 72B TP=4 warmup ≈ 5-8 min.
# -----------------------------------------------------------------------
echo "Waiting for vLLM at $VLLM_URL ..."
for i in $(seq 1 360); do
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
    echo "vLLM not ready after 30 min — aborting"
    exit 1
fi

# -----------------------------------------------------------------------
# Step 3: trainer across 7 nodes (28 ranks) via accelerate + DeepSpeed.
# -----------------------------------------------------------------------
srun --nodes="$TRAINER_NODE_COUNT" --ntasks="$TRAINER_NODE_COUNT" --ntasks-per-node=1 \
    --nodelist="$TRAINER_NODELIST" --gpus-per-node="$GPUS_PER_NODE" \
    --output="$RUN_DIR/trainer-%n.out" --export=ALL \
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
        export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
        export WANDB_API_KEY=\${WANDB_API_KEY:-}
        export WANDB_BASE_URL=\${WANDB_BASE_URL:-}
        export MACHINE_RANK=\$SLURM_NODEID
        # Tell create_run_dir to reuse this directory (skip its own timestamping).
        export INSPECT_RL_RUN_DIR=$SLURM_SUBMIT_DIR/$RUN_DIR
        echo \"[trainer \$(hostname) machine_rank=\$MACHINE_RANK] starting accelerate; knobs:\" \$(env | grep -E '^(NCCL_DEBUG|NCCL_NET|FI_LOG)' | tr '\\n' ' ')
        if [ \"\${IRL_DEBUG:-0}\" = \"1\" ]; then
            bash $PYSPY_SCRIPT $RUN_DIR $SLURM_SUBMIT_DIR 'inspect_rl.cli' &
        fi
        uv run --no-sync accelerate launch \
            --config-file $ACCELERATE_CONFIG \
            --num_machines $TRAINER_NODE_COUNT \
            --num_processes $TRAINER_RANKS \
            --machine_rank \$MACHINE_RANK \
            --main_process_ip $HEAD_TRAINER \
            -m inspect_rl.cli train $EXAMPLE \
            --vllm-base-url $VLLM_URL \
            --per-device-train-batch-size 1 \
            --num-generations 4 $TRAIN_EXTRA
    "

echo "Trainer step done."
