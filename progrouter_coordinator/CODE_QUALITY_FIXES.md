# Code quality, public feedback and fresh-run settings

## Implemented

- Static candidate checks reject string placeholders (`null`, `None`, `...`),
  empty/stub bodies, syntax errors and missing expected HumanEval entry points.
  Checks parse Python only; they never execute candidate code on the host.
- Rejected proposals are stored separately, with concrete repair feedback. They
  do not overwrite the last accepted implementation or create fake progress.
  Rejected attempts still produce online exploration targets when applicable.
- New accepted code triggers public checks whenever visible examples and an
  enabled execution tool exist, even if the worker omits `run_tests=true`.
  With tests disabled, no host fallback is used and correctness stays unknown.
- Current-code test provenance is preserved across rejected proposals and
  invalidated when accepted code changes. Final verification metadata records
  current-code public-test status separately from hidden correctness.
- Structured predictor inputs include stagnation and invalid-code signals.
  Optional minimum model-role sample coverage prevents exploiting completely
  unsupported candidates. The original routing score and queue remain unchanged.
- Feature representation is version 3; start fresh rather than loading older
  learned checkpoints.

## Routing diagnosis

On the prior 100-task run, removing queue pressure changed only 27 of 516
exploitation selections. The median difference between predicted-progress terms
for 0.5B and 1.5B was zero; their median queue-penalty difference was about 42,412.
Flat early predictions plus calibrated costs therefore explain much of the 1.5B
concentration. Concentration itself is not a reason to override Eq. 12 or force a
uniform model mixture. Cost calibration and V sensitivity still need experimental
assessment; no quality/cost advantage is guaranteed.

## Fresh configuration

Use `examples/ollama_routed_checked.json`, which preserves the existing models,
calibrated costs, V=1000, energy/time budgets and average-energy target. Changes:

```json
"predictor": {
  "min_samples": 32,
  "update_every": 4,
  "min_candidate_samples": 2
},
"evaluation_protocol": {
  "min_samples": 64,
  "max_epsilon": 0.08
}
```

These are explicit reconstruction settings, not parameters supplied by the paper.
Candidate support is counted by both model and role. Cold selection remains
uniform over feasible candidates until the predictor is ready for them. Phase
labels remain a declared heuristic, not proof of stabilization. `configure_routing`
now also preserves a top-level predictor block supplied in a calibration file.

## Verification

- 82 regression tests passed.
- Static replay of the previous 100 artifacts identifies all 30 syntax/nonfunction
  artifacts as invalid; 70 have accepted function structure. This is not an
  independent correctness judgment.
- Five-question real adaptive pilot: `runs/progrouter-codechecks-pilot`.
  All five held-out EvalPlus judgments passed; no grading was unavailable.
  Four coordinator workflows finished and one stalled with retained code that
  nevertheless passed hidden tests. There were no protocol errors. Eleven public
  check runs were triggered automatically by accepted code changes.
- Report: `reports-progrouter-codechecks-pilot.json`.
- The pilot collected 28 samples, so it remained cold exploration under the new
  32-sample threshold. It does NOT demonstrate improved learned routing accuracy
  or predict the next 100-question pass rate. Independent grading did not train
  the predictor or change parameters.

## Run the same 100 questions again

Activate qwen-local in your Docker-enabled terminal. Use fresh output paths and
do not pass an old router/predictor checkpoint.

```bash
python benchmarks.py run --public prepared/humaneval-100 --config examples/ollama_routed_checked.json --enable-tests --output runs/progrouter-100-checked
python benchmarks.py evaluate --private evaluator-data/references-100.json --runs runs/progrouter-100-checked --image progrouter-coding-evaluator --output reports-progrouter-100-checked.json
python report_experiment.py --reference-stream runs/progrouter-100-checked/stream.json --reports reports-progrouter-100-checked.json --output analysis-progrouter-100-checked.json
```

The existing pilot config and previous 100-task results were not overwritten.
