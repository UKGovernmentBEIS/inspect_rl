# inspect_rl.perf

Optional performance wrappers around the rollout path. The training loop is
correct without them — these only change *when* work happens.

| Module | Role |
|---|---|
| `prefetch.py` | `FreshestPrefetchRolloutFunc` (off-policy prefetch in a background thread) and `AutoCalibratingRolloutFunc` (measures rollout vs train time and flips prefetch on if it would help). |

**Does not belong here:** logic the loop can't run without → `inspect_rl.core`.
Generic, perf-neutral helpers → `inspect_rl.util`.
