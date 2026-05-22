# Smoke-test findings — 2026-04-21

End-to-end GRPO training verified on Slurm HPC (GH200 120GB × 4). Both GSM8K and TLDR tasks ran through the full pipeline: Inspect rollout → scoring → reward merge → GRPO loss → weight update → NCCL weight sync to vLLM.

## Setup

- **vLLM server**: `trl vllm-serve` on GPUs 0,1 (tp=2)
- **Trainer**: single GPU (GPU 2)
- **Model**: `google/gemma-3-270m-it` (text-only, avoids Gemma 3 multimodal quirks)

## GSM8K (xmlcount + correctness scorers)

LoRA r=16, 5 steps, batch=4, num_generations=4. xmlcount accuracy peaked at 0.147, correctness stayed at 0.0 — expected for 270m on math. Confirms multi-scorer reward aggregation works.

## TLDR (length-shaping reward) — the main result

Full fine-tuning (no LoRA), default GRPO hyperparams (batch=8, num_generations=8, lr=1e-6), 50 steps, dataset_limit=1000. Matches v1 `tldr.py` config as closely as possible — only difference is v2 uses vLLM server for generation.

Single scorer (`tldr_reward`): penalizes deviation from 100-char ideal, +100 bonus for first-person lead.

| Steps | Avg reward | Range |
|-------|-----------|-------|
| 1–5   | -245      | -272 to -206 |
| 10–15 | -192      | -169 to -211 |
| 20–25 | -103      | -47 to -145 |
| 35–40 | -93       | -48 to -157 |
| 45–50 | -40       | -28 to -67 |

Clear learning signal. Reward improved from ~-245 to ~-40 over 50 steps — summaries converging toward the 100-char target length.

### LoRA vs full fine-tuning

Earlier runs with LoRA r=16 (batch=4, gen=4, lr=1e-5) showed no learning over 15 steps — rewards bounced randomly between -190 and -430. Switching to full fine-tuning with v1 defaults immediately showed a clear trend. The likely causes: (a) LoRA with r=16 on a 270m model doesn't have enough effective capacity, (b) num_generations=4 gives noisy GRPO advantage estimates, (c) the default lr=1e-6 is better calibrated for full FT than the 1e-5 we used with LoRA.

## Weight sync verified

NCCL communicator confirmed active after training — `POST /close_communicator/` returned 200. Separate diagnostic: greedy-decoded output for a fixed prompt changed after training (pre: "The summary is that a fox jumped over a dog." → post: "The fox jumped over the dog, which was a simple and straightforward action."). Training is on-policy.

## Environment issues hit during testing

1. **`CUDA_HOME` must be set before importing torch.** DeepSpeed (via accelerate) probes `torch.utils.cpp_extension.CUDA_HOME` at import. Fix: `export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6`.

2. **Single-GPU training required.** Multi-GPU spreads params across devices but TRL's NCCL weight sync asserts tensors on `cuda:0`. Fix: `CUDA_VISIBLE_DEVICES=2` for trainer.

3. **Gemma-3 `token_type_ids` required during training.** `Gemma3Model.forward` raises `ValueError` without it. Fix: monkey-patch to default `torch.zeros_like(input_ids)`. Text-only models (Llama) don't need this.

4. **Stale NCCL communicator on kernel restart.** If a trainer dies without cleanup, the vLLM server holds the old communicator and the next `init_communicator/` call crashes. Fix: `POST /close_communicator/` before each run.

5. **asyncio in Jupyter.** `rollout.py` now detects a running event loop and falls back to `ThreadPoolExecutor` + `asyncio.run()` in a worker thread.

## Artifacts

- `debug.ipynb` — cells 0–3 (GSM8K), cells 4–6 (TLDR)
- `src/inspect_rl/example/gsm8k_v2.py` — v2 GSM8K example
- `src/inspect_rl/example/tldr_v2.py` — v2 TLDR example

## Open questions

- ~~**Longer TLDR run**: does the reward curve clearly trend up over 50–200 steps with 270m?~~ **Answered**: yes, clear improvement from -245 to -40 over 50 steps with full FT.
- **Llama-3.1-8B**: the original target model. Needs the Gemma-3 monkey-patch removed and a vLLM server switch. Should train faster in reward-per-step terms.
- **Multi-GPU training**: blocked by the NCCL device assertion. Likely needs TRL to support non-zero device ordinals, or a wrapper that remaps.
- **LoRA tuning**: LoRA didn't work out of the box on 270m. Worth revisiting with higher rank, different LR schedule, or a larger base model where LoRA has more room to maneuver.
