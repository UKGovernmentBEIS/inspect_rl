#!/bin/bash
#
# Simplest 1-node trainer (4 GPUs, plain DDP) — connects to a separate vLLM job.
#
# This is the "known good" baseline: matches the passing 2-node smoke from
# journal/008_multinode/004 (single trainer node, no sharding). Use it to
# verify the split vLLM/trainer plumbing works before adding FSDP/multi-node.
#
# Usage:
#   sbatch examples/configs/hpc1/1node/slurm_vllm.sbatch        # wait until ready
#   VLLM_URL=$(cat irl_output_slurm/latest/vllm_url.txt)
#   env -u MODEL sbatch --export=ALL,VLLM_URL=$VLLM_URL,MAX_STEPS=2 \
#       examples/configs/hpc1/1node/slurm_trainer.sbatch
#
#SBATCH --job-name=irl-trainer-1node
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null

set -euo pipefail

STAMP=$(date +%Y-%m-%d-%H-%M-%S)
RUN_DIR="irl_output_slurm/${STAMP}_slurm_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
exec 1>>"$RUN_DIR/master.out" 2>&1
# Single `latest` symlink in the base dir — tmux panes tail
# irl_output_slurm/latest/{master,trainer}.out across jobs.
ln -sfn "$(basename $RUN_DIR)" irl_output_slurm/latest
touch "$RUN_DIR/trainer.out"

if [ -z "${VLLM_URL:-}" ]; then
    echo "ERROR: VLLM_URL not set."
    echo "  VLLM_URL=\$(cat irl_output_slurm/latest/vllm_url.txt)"
    echo "  env -u MODEL sbatch --export=ALL,VLLM_URL=\$VLLM_URL,... examples/configs/hpc1/1node/slurm_trainer.sbatch"
    exit 1
fi

MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
EXAMPLE=${EXAMPLE:-math-agent}
MAX_STEPS=${MAX_STEPS:-}
OFF_POLICY_STEPS=${OFF_POLICY_STEPS:-0}
WANDB=${WANDB:-0}

TRAIN_EXTRA="--model $MODEL"
[ -n "$MAX_STEPS" ] && TRAIN_EXTRA="$TRAIN_EXTRA --max-steps $MAX_STEPS"
[ -n "$OFF_POLICY_STEPS" ] && TRAIN_EXTRA="$TRAIN_EXTRA --off-policy-steps $OFF_POLICY_STEPS"
[ "$WANDB" = "1" ] && TRAIN_EXTRA="$TRAIN_EXTRA --wandb"

echo "rundir:   $RUN_DIR/"
echo "node:     $(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)"
echo "vLLM:     $VLLM_URL"
echo "model:    $MODEL"
echo "example:  $EXAMPLE"

if ! curl -sf "$VLLM_URL/health/" >/dev/null 2>&1; then
    echo "ERROR: vLLM not reachable at $VLLM_URL"
    exit 1
fi
echo "vLLM reachable."

# Optional py-spy sidecar for hang diagnosis
PYSPY_SCRIPT="$RUN_DIR/pyspy_sidecar.sh"
cat > "$PYSPY_SCRIPT" << 'PYSPY_EOF'
#!/bin/bash
RUN_DIR="$1"
cd "$2"
sleep 30
while true; do
    PID=$(pgrep -f 'inspect_rl.cli' | head -1)
    [ -z "$PID" ] && break
    printf '\n--- %s ---\n' "$(date +%H:%M:%S)" >> "$RUN_DIR/trainer-pyspy-$(hostname -s).txt"
    uv run --no-sync py-spy dump --pid "$PID" --nonblocking \
        >> "$RUN_DIR/trainer-pyspy-$(hostname -s).txt" 2>&1 || true
    sleep 60
done
PYSPY_EOF
chmod +x "$PYSPY_SCRIPT"

srun --nodes=1 --ntasks=1 --gpus-per-node=4 \
    --output="$RUN_DIR/trainer.out" \
    bash -c "
        cd $SLURM_SUBMIT_DIR
        echo "[sbatch] NOTE: add 'module load' for your cluster's NCCL/OFI modules here if cross-node NCCL requires them"
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        export CUDA_HOME=\${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda}
        export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300
        export WANDB_API_KEY=\${WANDB_API_KEY:-}
        export WANDB_BASE_URL=\${WANDB_BASE_URL:-}
        # Tell create_run_dir to use this exact path (skip its own timestamping) —
        # keeps slurm stdouts + trainer artifacts under the same directory.
        export INSPECT_RL_RUN_DIR=$SLURM_SUBMIT_DIR/$RUN_DIR
        echo \"[node \$(hostname)] starting accelerate\"
        bash $PYSPY_SCRIPT $RUN_DIR $SLURM_SUBMIT_DIR &
        uv run --no-sync accelerate launch \
            --config-file examples/configs/hpc1/1node/accelerate.yaml \
            -m inspect_rl.cli train $EXAMPLE \
            --vllm-base-url $VLLM_URL \
            --per-device-train-batch-size 2 \
            --num-generations 4 $TRAIN_EXTRA
    "

echo "Done."
