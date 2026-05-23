lint:
    uvx ruff format src/inspect_rl
    uvx ruff check --fix --unsafe-fixes src/inspect_rl

test:
    uv run --no-sync pytest tests

# Start vLLM server for inference (defaults to GPU 0).
# CUDA_HOME is auto-detected from the NVHPC toolchain on aarch64 Slurm HPC when unset.
serve model="Qwen/Qwen2.5-3B-Instruct" devices="0":
    uv run --no-sync irl serve {{model}} --devices {{devices}}

# Run GRPO training on a single GPU (NCCL weight sync requires all trainer
# params on one device until the distributed path lands). Defaults to GPU 1
# so it pairs with `just serve` on GPU 0.
train example="tldr" devices="1":
    # cleanup prev runs
    curl -sf --max-time 5 -X POST http://localhost:8000/close_communicator/ -H "Content-Type: application/json" -d '{}' || true
    uv run irl train --devices {{devices}} {{example}}

# End-to-end smoke test on a 2-GPU machine: start vLLM on GPU 0, run one
# trainer step on GPU 1, tear vLLM down. Uses the tldr example with
# google/gemma-3-270m-it so cold start is fast (~2-3 min wall time on H100).
# Override the GPUs if 0/1 are busy: `just smoke 2 3`.
smoke vllm_gpu="0" train_gpu="1":
    #!/usr/bin/env bash
    set -euo pipefail
    if curl -sf --max-time 2 http://localhost:8000/health/ >/dev/null 2>&1; then
        echo "[smoke] aborting: something is already serving on http://localhost:8000."
        echo "[smoke] kill the existing vllm before running this recipe — the trainer"
        echo "[smoke] would otherwise NCCL-sync against the wrong process."
        exit 1
    fi
    log=$(mktemp -t smoke-vllm.XXXX.log)
    cleanup() {
        echo "[smoke] tearing down vllm (log: $log)"
        if [ -n "${vllm_pgid:-}" ]; then
            kill -TERM -"$vllm_pgid" 2>/dev/null || true
            sleep 2
            kill -KILL -"$vllm_pgid" 2>/dev/null || true
        fi
    }
    trap cleanup EXIT
    echo "[smoke] starting vllm (google/gemma-3-270m-it on GPU {{vllm_gpu}})…"
    setsid uv run --no-sync irl serve google/gemma-3-270m-it --devices {{vllm_gpu}} >"$log" 2>&1 &
    vllm_pgid=$!
    for i in $(seq 1 60); do
        if curl -sf --max-time 2 http://localhost:8000/health/ >/dev/null 2>&1; then
            echo "[smoke] vllm ready after ${i}x5s"
            break
        fi
        if ! kill -0 "$vllm_pgid" 2>/dev/null; then
            echo "[smoke] vllm exited early; tail of log:"
            tail -40 "$log"
            exit 1
        fi
        sleep 5
    done
    curl -sf --max-time 2 http://localhost:8000/health/ >/dev/null 2>&1 || { \
        echo "[smoke] vllm did not become healthy in 5min; tail of log:"; \
        tail -40 "$log"; \
        exit 1; \
    }
    echo "[smoke] running one trainer step on GPU {{train_gpu}}…"
    uv run irl train --devices {{train_gpu}} tldr --max-steps 1
    echo "[smoke] PASSED"

# Clean up old irl_output_slurm run directories, keeping the most recent N.
# Targets dirs only (e.g. 2026-05-23-13-23-26_slurm_<jobid>/) — leaves
# `latest` / `latest_vllm` symlinks alone, and re-points `latest` if it
# pointed at a now-deleted dir.
clean-irl-output keep="5":
    #!/usr/bin/env bash
    set -euo pipefail
    shopt -s nullglob
    # Real dirs only (not symlinks), newest first.
    mapfile -t dirs < <(find irl_output_slurm -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
        | sort -rn | awk '{print $2}')
    count=${#dirs[@]}
    echo "found $count irl_output_slurm dirs, keeping {{keep}}"
    if [ "$count" -le {{keep}} ]; then echo "nothing to remove"; exit 0; fi
    # Don't delete whatever `latest` currently resolves to (active run).
    active=""
    [ -L irl_output_slurm/latest ] && active=$(readlink -f irl_output_slurm/latest 2>/dev/null || true)
    removed=0
    for d in "${dirs[@]:{{keep}}}"; do
        if [ -n "$active" ] && [ "$(readlink -f "$d")" = "$active" ]; then
            echo "skipping active: $d"; continue
        fi
        if rm -rf "$d" 2>/dev/null; then
            removed=$((removed+1))
        else
            echo "skipped (busy/open files): $d"
        fi
    done
    echo "removed $removed dirs"
    if [ -L irl_output_slurm/latest ] && [ ! -e irl_output_slurm/latest ]; then
        newest=$(find irl_output_slurm -mindepth 1 -maxdepth 1 -type d -printf '%T@ %f\n' \
            | sort -rn | awk '{print $2; exit}')
        if [ -n "$newest" ]; then
            ln -sfn "$newest" irl_output_slurm/latest
            echo "repointed latest -> $newest"
        else
            rm irl_output_slurm/latest && echo "removed dangling latest symlink"
        fi
    fi

# Clean up old slurm log directories, keeping the most recent 10.
# Targets slurm_logs/<datetime>_<jobid>/ — leaves the symlink + any other files alone.
clean-slurm-logs keep="10":
    @count=$(ls -1dt slurm_logs/*_[0-9]*/ 2>/dev/null | wc -l); \
        echo "found $count slurm log dirs, keeping {{keep}}"; \
        if [ $count -gt {{keep}} ]; then \
            ls -1dt slurm_logs/*_[0-9]*/ | tail -n +$(({{keep}}+1)) | xargs rm -rf && \
            echo "removed $((count-{{keep}})) old dirs"; \
        else \
            echo "nothing to remove"; \
        fi

# At-a-glance Slurm cluster utilisation: partitions, your jobs, per-node
# GPU allocation. Useful before submitting to avoid piling onto a saturated
# node (see "Shared Slurm clusters" in README).
cluster-status:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "── partitions ──────────────────────────────────────────────"
    sinfo -o "%20P %5a %10l %6D %10t %N"
    echo
    echo "── your jobs ───────────────────────────────────────────────"
    squeue --me -o "%.10i %.12P %.20j %.8T %.10M %.6D %R"
    echo
    echo "── GPU allocation per node (alloc/total) ───────────────────"
    sinfo -N -o "%N %G %C %t" -h | awk '{printf "  %-20s gres=%-20s cpus=%-15s state=%s\n", $1, $2, $3, $4}' | sort -u
    echo
    echo "── tip: 'squeue -p <partition> -o \"%.10i %.8u %.6D %R\"' to see neighbours ──"

# Slurm HPC (arm64): fetch arm64 kubectl and configure EKS.
# Reads AWS_EKS_CLUSTER and K8S_NAMESPACE_USER from the environment — see
# .env.example.
setup-slurm-hpc:
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/arm64/kubectl"
    chmod +x kubectl
    mv kubectl ~/.local/bin

    aws eks update-kubeconfig --name ${AWS_EKS_CLUSTER:?set AWS_EKS_CLUSTER (see .env.example)}
    kubectl config set-context --current --namespace=${K8S_NAMESPACE_USER:?set K8S_NAMESPACE_USER (see .env.example)}-default
