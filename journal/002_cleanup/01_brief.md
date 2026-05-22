in the prv journal try we switched a new architecture which lets us rollout the model in an entire inspect agent environment with multiple turns and coding environments.

the issue is things are a bit of a mess

1. we still have old code/docs e.g. readme/examples and in source - pls can we move these to 1 self contained area which will probs be removed soon
2. we need new docs and a better UI see the debug.ipynb - it's very hard to understand what is going on
  - ideally we have a simple output format that works both in terminal and jupyter, and when sent to a wandb output stream
3. artifact management: inspect produces output .eval files in logs/
  - can we have one artifact directory that holds all of our stuff (model weights, eval logs, debug logs, etc.)
4. examples
  - tldr is great but we need (a) a multi turn agent without an environment, and (b) one with a coding environment
  - these should line in python scripts that are well documented though i can manually check these function properly with you
5. lora - by default let's fine tune the whole thing, but having docs/example with lora is cool