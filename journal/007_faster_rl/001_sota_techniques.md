# 001 — Frontier RL speed techniques: OLMo 3 + VCPO

Research note on how OLMo 3 (Allen AI, Dec 2025) and VCPO (MIT Han Lab, Feb 2026) make GRPO-based RL training faster, and what's applicable to inspect-rl.

**Sources:** [OLMo 3 paper](https://arxiv.org/abs/2512.13961), [OLMo 3 deep dive](https://cameronrwolfe.substack.com/p/olmo-3), [VCPO paper](https://arxiv.org/abs/2602.17616), [VCPO code](https://github.com/mit-han-lab/vcpo)

## The bottleneck

Same problem everywhere: the trainer is idle during generation, and inference is idle during weight updates. OLMo 3 measured 75% of wall-clock time as the learner waiting for rollout data (32B model, 20 inference nodes vs 8 training nodes). inspect-rl has the same structure — `rollout_func` blocks `trainer.train()` synchronously.

## Tier 1 — GRPO algorithmic mods (no architecture change)

OLMo 3 lists six modifications to vanilla GRPO. Several are free or near-free to implement on top of TRL's `GRPOTrainer`.

### 1a. Zero-gradient filtering + active sampling

**What:** Drop prompts where every completion in the group gets the same reward (zero advantage → zero gradient). Backfill the batch by generating more until you have enough non-trivial examples.

**Why it matters:** On easy datasets or late in training, most prompts get all-correct or all-wrong completions. You're paying full generation cost for zero learning signal. OLMo 3 reports **4x speedup** from active sampling alone.

**Cost:** Medium. Zero-gradient detection is trivial (check if all rewards in a group are equal). Active sampling — re-requesting generations until the batch is full — requires a loop around `rollout_func` or a wrapper that calls inspect_eval multiple times per step. Maybe 100–200 lines, mostly in `rollout.py` and `trainer.py`.

**Expected benefit:** 1.5–4x fewer wasted steps depending on dataset difficulty. Biggest wins on GSM8K-style tasks where 3B models already get many problems right.

### 1b. Token-level loss normalization

**What:** Normalize the policy gradient loss by total tokens in the batch, not per-sequence. Short completions currently contribute disproportionately high per-token gradient.

**Cost:** Low. Check if TRL's `GRPOConfig` has a `loss_type` or `token_level_loss` flag. If not, a small monkey-patch to `GRPOTrainer._compute_loss`. ~20 lines.

**Expected benefit:** Removes length bias. Prevents the policy from learning to write shorter answers to maximize per-token reward. Stabilises training more than speeds it up.

### 1c. Asymmetric clipping

**What:** Set the PPO-style ratio clip bounds asymmetrically — higher upper bound than lower bound (e.g. clip_low=0.8, clip_high=1.28 instead of symmetric 0.8/1.2). Allows larger positive updates while still constraining negative ones.

**Cost:** Low. Config change or small patch if TRL doesn't expose separate bounds. ~10 lines.

**Expected benefit:** Faster learning from high-reward completions. OLMo 3 found this helps especially early in training when the policy needs to move quickly toward tool-use patterns.

### 1d. Drop KL penalty and std-dev normalization

**What:** Remove the KL divergence term (`beta=0`) and remove reward std-dev from the advantage calculation. OLMo 3 found both unnecessary with their other modifications in place.

**Cost:** Trivial. `beta=0` in config. Std-dev removal may need a TRL patch.

**Expected benefit:** More policy flexibility. **Caution:** our CLAUDE.md notes say `beta=0` drifts policies into noise and `beta=0.05` is a reasonable floor. This conflicts with OLMo 3's recommendation. Likely depends on batch size / num_generations — OLMo 3 uses much larger batches. Test before committing.

**Tier 1 total cost:** ~1–2 days of implementation + testing.
**Tier 1 total benefit:** 2–4x fewer wasted gradient steps, more stable training curves. No wall-clock improvement on the generation side — still synchronous.

---

## Tier 2 — Async generation with importance sampling correction

### The idea

Break the synchronous coupling between generation and training. Instead of:

```
[generate] → block → [train] → block → [generate] → ...
```

Run a continuous generation loop in the background. The trainer pulls completed batches from a queue and trains immediately. Rollouts are "1 step off-policy" — generated under the previous step's weights. Correct for the distribution shift using Truncated Importance Sampling (TIS):

```
ratio = exp(log_prob_current_policy - log_prob_generation_policy)
capped_ratio = min(ratio, ρ)  # ρ ≈ 10
loss = capped_ratio * advantage * log_prob
```

OLMo 3 also does **inflight model updates** — pushing new weights to vLLM without pausing generation or clearing the KV cache. vLLM supports this via its NCCL weight-sync endpoint (which inspect-rl already uses).

### What changes in inspect-rl

1. **New async rollout manager** — a background thread/process that continuously calls `inspect_eval` and pushes `(rollout_data, generation_logprobs)` into a queue. ~200–300 lines, new file.
2. **Modified `rollout_func`** — instead of running eval inline, pop from the queue. Return the pre-generated data. The generation_logprobs from the rollout become `log_prob_old`; TRL computes `log_prob_new` during the forward pass.
3. **TIS correction in the loss** — TRL's GRPO already computes a ratio for PPO-style clipping. We'd need to multiply in the off-policy importance weight. This likely requires subclassing `GRPOTrainer` or patching `_compute_loss`. ~50–100 lines.
4. **Weight sync scheduling** — after each training step, push weights to vLLM. Currently this is already done by TRL. The change is that we no longer wait for new rollouts to finish before starting the next training step.

### Cost

~1–2 weeks. The hard parts are:
- Getting the async queue + eval lifecycle right (model creation/teardown per batch, error handling).
- Ensuring the TIS ratio is computed correctly (need to store generation-time logprobs alongside the rollout and align them token-for-token with the new policy's logprobs).
- Testing that training stability holds with 1-step staleness.

### Expected benefit

Near-elimination of trainer idle time. If generation takes 3x longer than weight update (typical), async gives ~3x wall-clock speedup. Combined with Tier 1 active sampling, could be **4–6x** total.

### Risk

TIS with a simple cap can be unstable if the policy moves fast (small model, high learning rate). This is exactly what VCPO addresses.

---

## Tier 3 — VCPO: variance-controlled off-policy RL

### The problem with TIS alone

When the policy drifts far from the generation policy (high staleness), importance-sampling ratios become heavy-tailed. A few rollouts get enormous weight, dominating the gradient. OLMo 3 uses simple ratio capping (truncation), which throws away information. Under high asynchrony (k>2 steps stale), TIS can cause gradient spikes and training collapse.

### VCPO's solution

Two mechanisms that plug into any policy-gradient method (GRPO, REINFORCE, RLOO):

**ESS-Guided Step Scaling:** Scale the effective learning rate by the square root of the effective sample size:

```
ρ_ess = (Σ w_i)² / (B · Σ w_i²)
η_eff ∝ √ρ_ess
```

When staleness makes importance weights unreliable (low ESS), updates automatically shrink. When weights are fresh (ESS ≈ B), learning rate is unchanged. Follows AdamW conventions.

**Off-Policy Optimal Baseline (OPOB):** A closed-form control variate that minimises gradient variance without a learned critic:

```
b*_OPOB = (Σ w_i² |∇log π(τ_i)|² R_i) / (Σ w_i² |∇log π(τ_i)|²)
```

Minimal overhead, compatible with tensor parallelism. Replaces the group-mean baseline in GRPO.

### Results

- 2.5x end-to-end speedup on AIME 2025 (multi-turn tool use, 7B model)
- Stable up to k=128 steps off-policy (though practical sweet spot is k=2–10)
- Tested on Qwen2-1.5B and Qwen2.5-7B, GSM8K and MATH-500
- Built on Megatron backend (not TRL)

### Cost for inspect-rl

~2–4 weeks. Requires:
- Everything from Tier 2 (async generation pipeline)
- Custom GRPO loss function implementing ESS scaling + OPOB baseline. This is probably a `GRPOTrainer` subclass that overrides `_compute_loss`. ~200–300 lines of math-heavy code.
- Careful testing across staleness levels (k=1, 2, 4, 8) to validate stability.
- VCPO's reference code is Megatron-based; translating to TRL/HF Trainer idioms is non-trivial but the math is self-contained.

### Expected benefit

Same async speedup as Tier 2 (~3x from eliminating idle time) but **stable at higher asynchrony** — you can push k=4–8 without gradient spikes. Combined with Tier 1, **4–8x total wall-clock improvement** with more robust training. The real win over Tier 2 is reliability: you don't need to babysit the training for divergence.

---

## Summary

| Tier | Technique | Wall-clock speedup | Impl cost | Risk |
|------|-----------|-------------------|-----------|------|
| 1 | GRPO mods (active sampling, token-level loss, asymmetric clip) | 2–4x fewer wasted steps | 1–2 days | Low — algorithmic, no infra change |
| 2 | Async generation + TIS | 3–4x (eliminates trainer idle) | 1–2 weeks | Medium — TIS can be unstable at high staleness |
| 3 | VCPO (ESS scaling + OPOB) | 4–8x combined with Tier 1 | 2–4 weeks | Low once implemented — designed for stability |

Tier 1 is the obvious first move. Active sampling alone could cut GSM8K training time in half by not wasting gradient steps on all-correct batches.

Tier 2 vs 3 depends on ambition. If we're only ever 1-step off-policy (which is the common case for single-node training where generation is ~3x slower than update), TIS is probably fine and Tier 2 suffices. VCPO becomes essential if we scale to multi-node inference where generation is fast enough to get 4+ steps ahead.

## Next steps

- Check TRL's `GRPOTrainer` source for existing support for token-level loss, asymmetric clipping, and zero-gradient filtering. Some of these may already be configurable.
- Prototype active sampling as a wrapper around `rollout_func` — generate, filter, re-generate until batch is full.
- Benchmark current step timing breakdown (generation vs weight update) on a real run to quantify the async opportunity.
