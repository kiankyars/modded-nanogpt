---
name: benchmark-closer
description: Drive metric-based optimizer benchmark loops for modded-nanogpt Track 3 until a candidate beats the current step-count target or is killed with evidence. Use when Codex needs to implement, tune, evaluate, parse logs, or repeatedly test optimizer changes in records/track_3_optimization/train_gpt_simple.py while preserving Track 3 validity.
---

# Benchmark Closer

Use this skill to close the loop between an optimizer idea and a measured Track 3 result. The job is to keep experiment state clean, preserve benchmark validity, and continue until the candidate wins or the evidence says to stop.

## Benchmark Contract

Read `records/track_3_optimization/README.md` and `records/track_3_optimization/train_gpt_simple.py` before changing anything.

Preserve:

- Dataset and token streams.
- Batch size.
- Model architecture.
- Validation computation and target loss.
- One forward-backward pass per step.

Allowed:

- Optimizer algorithms.
- Optimizer hyperparameters and schedules.
- Model initialization.

Default objective: reach validation loss `<= 3.28` in fewer than the current best local step count. Treat `3500` steps as the default best only if local records do not show a newer result.

## Loop

1. Establish the baseline.
   - Parse relevant logs with `scripts/parse_track3_log.py`.
   - Record current best step count, final loss, command, hardware, and commit if available.

2. Define the experiment.
   - State one hypothesis.
   - Change one meaningful variable or one coherent optimizer port at a time.
   - Keep changes concentrated in the optimizer and init/hyperparameter sections unless the user explicitly asks otherwise.

3. Run a cheap correctness pass.
   - Verify syntax and obvious import/device issues before launching a full run.
   - If a short smoke run is possible without invalidating the full evaluator, use it only to catch crashes.

4. Run the fixed evaluator.
   - Use the user-provided command if present.
   - Otherwise use the Track 3 README command shape:
     `torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) records/track_3_optimization/train_gpt_simple.py`
   - Do not alter the evaluator to make the metric easier.

5. Parse and decide.
   - Run `scripts/parse_track3_log.py` on the log.
   - Keep if it improves first step at or below target, final loss at equal steps, stability, or a clearly relevant per-optimizer SOTA.
   - Discard or revert the candidate if it crashes for a fundamental reason, violates rules, or underperforms after reasonable tuning.

6. Tune in this order.
   - Weight decay.
   - Learning rate.
   - Schedule/cooldown.
   - Momentum/betas.
   - Initialization.

7. Persist the trail.
   - Keep a compact table of every run: change, command, log path, target step, final loss, keep/discard/crash, and next action.
   - Update the strategy after repeated misses. Do not repeat failed settings.

## Kill Criteria

Stop spending runs on a candidate when any of these is true:

- It violates Track 3 rules.
- It requires a dependency or memory state that is impractical for the available hardware.
- It cannot complete a real run after two focused crash fixes.
- It remains worse than baseline after weight decay and learning rate have each received a real sweep.
- A simpler candidate has higher expected value.

## Parser

Use `scripts/parse_track3_log.py` to summarize logs:

```bash
python .agents/skills/benchmark-closer/scripts/parse_track3_log.py records/track_3_optimization/results/*.txt
```

The parser emits JSON with the final loss and first step at or below the target loss.

## Output Template

Return or maintain this concise status after each loop:

```markdown
## Current Best
[steps, loss, log]

## Runs
| Status | Change | First <=3.28 | Final loss | Log | Decision |
| --- | --- | ---: | ---: | --- | --- |

## Next Experiment
[One next change and why.]

## Stop Condition
[What would make this candidate done, killed, or promoted.]
```
