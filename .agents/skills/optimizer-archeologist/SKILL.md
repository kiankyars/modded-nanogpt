---
name: optimizer-archeologist
description: Find, inspect, and rank pre-existing neural-network optimizer implementations for modded-nanogpt Track 3 and adjacent fixed-evaluator optimizer benchmarks. Use when Codex needs to search arXiv/GitHub/reference repos for optimizer variants, decide whether an optimizer is valid under Track 3 rules, extract implementation details from source code, or produce a concrete porting plan for records/track_3_optimization/train_gpt_simple.py.
---

# Optimizer Archeologist

Use this skill to turn optimizer literature and repos into agent-ready Track 3 candidates. The output should help the next agent make or reject a port, not produce a polished human research report.

## Track 3 Frame

Anchor all judgment to `records/track_3_optimization/README.md` and `records/track_3_optimization/train_gpt_simple.py`.

Valid candidates must keep:

- The same dataset, batch size, and architecture.
- One forward-backward pass per step.
- The target of reaching `3.28` validation loss in as few steps as possible.

Track 3 allows optimizer algorithms, optimizer hyperparameters and schedules, and model initialization changes. Prefer changes concentrated in the `Optimizer` and `Init & Optim Hyperparams` sections.

## Workflow

1. Establish the local baseline.
   - Read the Track 3 README and current `train_gpt_simple.py`.
   - Parse local result logs if needed to confirm the best known step count and hyperparameters.
   - Treat Muon `3500` steps as the baseline only if no newer local record contradicts it.

2. Search for real implementations.
   - Prefer official repositories, author repositories, or well-used PyTorch packages.
   - Search for optimizer families likely to help matrix-heavy transformer training: PSGD/Kron, SOAP/Shampoo, schedule-free variants, Muon/NorMuon variants, lookahead/EMA wrappers, preconditioned momentum, adaptive spectral methods, and strong AdamW variants.
   - Prefer code with tests, example hyperparameters, or published transformer/LM results.

3. Reject invalid or low-signal candidates early.
   - Reject methods that depend on data, batch, model architecture, loss, tokenizer, or multi-pass training changes.
   - Reject methods whose memory state is obviously too large for Track 3 without a clear matrix-grouping compromise.
   - Reject repos that are too vague to port faithfully unless the paper gives the missing update rule.

4. Extract the porting facts.
   - Record exact source links, files, classes/functions, update equations, optimizer state, parameter grouping, dtype expectations, distributed assumptions, and dependency requirements.
   - Identify required hyperparameters and known good defaults.
   - Separate paper-specified facts from inferred implementation choices.
   - Do not paste large source files; write a minimal, attributed porting summary.

5. Rank candidates.
   - Score expected upside, Track 3 validity, implementation complexity, memory risk, dependency risk, and tuning burden.
   - Bias toward candidates that can be isolated to `train_gpt_simple.py` without changing benchmark semantics.

## Output Template

Return this structure:

```markdown
## Recommendation
[One candidate to port first and why.]

## Ranked Candidates
| Rank | Candidate | Source | Expected upside | Port surface | Initial hparams | Main risk |
| --- | --- | --- | --- | --- | --- | --- |

## Porting Notes
[For the top candidate: optimizer state, param groups, update rule, schedule/init interactions, and exact files/functions to inspect.]

## Kill Criteria
[What result or implementation blocker should make Benchmark Closer stop spending runs on this candidate.]
```

## Handoff

If the top candidate is ready to test, invoke or recommend `benchmark-closer` with:

- Candidate name and source.
- Minimal implementation surface.
- Initial hyperparameters.
- Fixed evaluator command.
- Kill criteria.
