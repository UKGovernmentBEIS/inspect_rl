#!/bin/bash
#
# Persistent 1-node vLLM server for Slurm HPC (developed against a GH200
# Slingshot cluster). Runs until cancelled — pair with 1node/slurm_trainer.sbatch.
#
# Only safe with a 1-node trainer. Multi-node trainer + separate-job vLLM
# trips a Slingshot CXI VNI bug (see docs/04-known-issues.md). For
# multi-node trainers, use 2node/slurm.sbatch or 4node/slurm.sbatch.
#
# Usage:
#   sbatch examples/configs/1node/slurm_vllm.sbatch
#   # Wait for "vLLM ready" line, then submit the trainer job pointing at the URL.
#
#SBATCH --job-name=irl-vllm
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00
#SBATCH --output=/dev/null

set -euo pipefail

STAMP=$(date +%Y-%m-%d-%H-%M-%S)
# All slurm runs land under irl_output_slurm/ — same convention as the trainer's
# create_run_dir. Tmux panes can tail irl_output_slurm/latest/{master,vllm}.out
# across jobs via the single `latest` symlink in the parent.
RUN_DIR="irl_output_slurm/${STAMP}_slurm_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
exec 1>>"$RUN_DIR/master.out" 2>&1
ln -sfn "$(basename $RUN_DIR)" irl_output_slurm/latest
touch "$RUN_DIR/vllm.out"

VLLM_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
VLLM_PORT=8000
VLLM_URL="http://${VLLM_NODE}:${VLLM_PORT}"
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
TP=${TP:-4}
DEVICES=${DEVICES:-0,1,2,3}
ENFORCE_EAGER=${ENFORCE_EAGER:-}

SERVE_EXTRA=""
[ -n "$ENFORCE_EAGER" ] && SERVE_EXTRA="--enforce-eager"

echo "rundir:  $RUN_DIR/"
echo "node:    $VLLM_NODE"
echo "url:     $VLLM_URL"
echo "model:   $MODEL"
echo "tp:      $TP"
echo "devices: $DEVICES"

srun --nodes=1 --ntasks=1 --output="$RUN_DIR/vllm.out" --export=ALL \
    bash -c "
        cd $SLURM_SUBMIT_DIR
        echo "[sbatch] NOTE: add 'module load' for your cluster's NCCL/OFI modules here if cross-node NCCL requires them"
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        export CUDA_HOME=\${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda}
        # Surface the secondary NCCL group's setup when init_communicator
        # arrives from a multi-node trainer (see journal/008_multinode/005,006).
        export NCCL_DEBUG=INFO
        export NCCL_DEBUG_SUBSYS=INIT,COLL,NET,ENV
        # Opt-in workaround knobs (only export when non-empty —
        # NCCL_NET_PLUGIN="" / FI_LOG_LEVEL="" break things).
        for _v in NCCL_CROSS_NIC NCCL_NET NCCL_NET_GDR_LEVEL NCCL_MAX_NCHANNELS NCCL_MIN_NCHANNELS NCCL_NET_PLUGIN FI_LOG_LEVEL FI_LOG_PROV; do
            v=\$(eval echo \\\${\$_v:-})
            if [ -n \"\$v\" ]; then export \$_v=\"\$v\"; fi
        done
        echo \"[node \$(hostname)] env knobs:\" \$(env | grep -E '^(NCCL_NET|NCCL_CROSS|NCCL_MAX|NCCL_MIN|FI_LOG)' | tr '\\n' ' ')
        uv run --no-sync irl serve '$MODEL' \
            --tensor-parallel-size $TP \
            --devices $DEVICES \
            --gpu-memory-utilization 0.9 $SERVE_EXTRA
    " &
VLLM_PID=$!

echo "Waiting for vLLM…"
until curl -sf "$VLLM_URL/health/" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM exited — see $RUN_DIR/vllm.out"; exit 1
    fi
    sleep 5
done

# Write URL to a well-known path so the trainer job can pick it up without
# coordination. MUST happen BEFORE the "vLLM ready" echo — downstream tools
# tail master.out for that string and immediately read irl_output_slurm/latest/vllm_url.txt.
echo "$VLLM_URL" > "$RUN_DIR/vllm_url.txt"
echo "vLLM ready: $VLLM_URL"

wait "$VLLM_PID"
