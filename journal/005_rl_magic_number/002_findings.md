# 005 Magic Number — Findings

Built `magic_number.py` as a minimum-viable RL sanity test after `math_agent` trained for 60 steps without learning. Same prompt every sample, target always `MAGIC_NUMBER`, only tool is `submit`, up to 3 attempts per episode. Uses `react` (agent API) rather than `basic_agent` (solver API). If *this* doesn't learn, the pipeline is broken — not the task difficulty.

It learned. Two real bugs on the way.

## Bug 1 — `AgentSubmit.answer_only=False` silently kills retry scoring

Default `AgentSubmit` *accumulates* each submit into `state.output.completion` with a `\n\n` delimiter (`_react.py:245-248`). After two submits "5" then "7", `completion == "5\n\n7"`. `int(completion)` raises, scorer returns 0.0, rollout gets no reward **even though the last guess was correct**. Same logic runs intermediate via `AgentAttempts`, so the retry mechanism itself can't detect a correct second attempt.

Fix: `submit=AgentSubmit(answer_only=True)` — replaces completion per submit instead of appending.

## Bug 2 — `react` strips tool_calls from the final state

`react` post-processes the finished state through `_remove_submit_tool` (`_react.py:331-335`) which:
- Strips `ChatMessageTool` entries where `function == submit_name`
- Strips submit calls from `ChatMessageAssistant.tool_calls`
- Synthesizes a text-only assistant message containing the submitted string

Our `valid_submit` and `tool_call_failures` scorers walk `ChatMessageAssistant.tool_calls` — they found nothing and silently always returned 0.0. Confirmed by inspecting `s.messages` in the notebook: `assistant text='2', tool_calls=False`.

Fix: `submit=AgentSubmit(answer_only=True, keep_in_messages=True)`.

This is specific to `react` — `basic_agent` doesn't do this rewrite. `math_agent` (which uses `basic_agent`) wasn't affected.

## Reward-hacking observation

Early iteration of `submitted()` scored 1.0 for *any* `ChatMessageTool` with `function == "submit"`. Training logs showed tool-calling quality **getting worse after briefly getting better** — model learned to spam JSON-shaped text that satisfied the floor credit. Replaced with `valid_submit()` which walks tool_calls and requires `arguments["answer"]` to parse as a 0-9 digit, and dropped weight 0.5 → 0.1 so correctness dominates 26:1. Added `tool_call_failures()` with weight -0.3 as an explicit penalty for parse errors / batched calls / hallucinated tool names.

Final reward matrix: clean+right=2.6, clean+wrong=0.1, failures+anything ∈ [-0.3, 2.3].

## Learning curve (15 steps, bs=8, num_gen=8, lr=5e-6)

```
step |correct|valid_sub|failures|tok_out
  1  | 0.12  |  0.62   |  0.25  | 1695
  3  | 0.50  |  1.00   |  0.12  |  725
  7  | 0.38  |  1.00   |  0.00  |  483
  8  | 0.88  |  1.00   |  0.00  |  355   ← escape
 10  | 1.00  |  1.00   |  0.00  |  228
 15  | 1.00  |  1.00   |  0.00  |  284
```

- `valid_submit` hits 1.0 by step 3 — tool-calling format learns almost immediately.
- `tool_call_failures` drops to 0 by step 3 and stays.
- `magic_correctness` plateaus 0–0.5 for 7 steps, then step-change to ≥0.88.
- **Trajectory length collapse**: output tokens 1700 → 280. Policy went from "0, 1, 2 across three attempts" to "2 on first attempt" — confirms the gradient updated the *first*-digit preference, not just aggregate hit rate.

## Caveat on this task

Base Qwen-2.5-0.5B-Instruct has a strong prior toward `0, 1, 2` as first guesses for "pick a digit" prompts. With `MAGIC_NUMBER=2` it almost always reaches 2 by attempt 3 even before training. For a starker demo of learning, use a non-adjacent digit (7 works) — forces the policy to actually move probability mass.

## Leftover for `math_agent`

`math_agent` uses `basic_agent`, so Bug 2 doesn't apply. But the reward-hacking diagnosis (loose "any submit" scorer + oversized submit weight) likely does. Worth tightening its `submitted()` to require a numeric arg and dropping its submit weight from the current 1.5.

## Files touched

- `src/inspect_rl/example/magic_number.py` — new example.
- `src/inspect_rl/example/main.py` — wire into CLI.
- `README.md` — examples table + "How it works" section rewrite.

## Takeaway

`react`'s post-loop state rewriting is hostile to scorers that inspect message structure. When writing an Inspect task for RL, always pass `AgentSubmit(answer_only=True, keep_in_messages=True)` unless you explicitly want the rewrite behavior.
