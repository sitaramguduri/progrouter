# Coordinator and context fixes

The original 100-task run had 45 fatal duplicate-decision rejections, 12 missing
final answers, 3 missing/invalid roles, 6 character-limit stops and 4 worker-limit
stops. The old run remains untouched.

## Changes

- Coordinator output uses disjoint action-specific schema branches. Dispatch
  requires an eligible role and nonempty subtask; finish requires a nonempty final
  answer; abort requires an explicit blocker. Finished work is not an abort reason.
- Repetition is no longer a protocol error. Code/test-result changes are tracked.
  After repeated non-material steps, the coordinator receives recovery context;
  if analysis has not produced code, the recovery dispatch is restricted to the
  coder. The coder schema requires full code rather than null/another outline.
- Bounded recovery ends with explicit terminal decisions or `stalled`, not an
  automatic success. Useful repairs and changed test evidence reset stagnation.
- At the worker-step boundary, schema-enabled coordinator calls can only finish
  or abort. Hard energy/time/call budgets remain enforced.
- Transformers now receives `return_dict=False` for chat tokenization. The
  installed version otherwise returned BatchEncoding, whose length was 2 fields,
  not the input token count. A flat integer-ID list is validated before counting.
  This correction affects prefill estimation as well as context admission.
- Token-aware calls use the token budget, not the old character heuristic.
  When necessary, prompt preparation removes duplicate output, omits older
  history and shortens prose explicitly. Task/current code and current test
  results are preserved. A bounded excerpt of latest test output is retained.
  Compaction is logged; full states remain in artifacts. Mandatory inputs too
  large for 4K still stop explicitly rather than being silently truncated.
- The cost estimator uses the same prompt preparation as execution. Feedback
  from superseded code is not advertised as current-code verification.

The recovery restrictions are implementation safeguards for the smaller
coordinator, not additional ProgRouter equations. Candidate routing scores,
exploration learning targets and the virtual queue equations are unchanged.

## Validation

- 76 regression tests passed, including action-field requirements, repeated
  repairs, bounded stalls, tokenization return types, context compaction and
  current-code test provenance.
- Original six character-limit cases were replayed with matching local
  tokenizers; their coordinator/worker payloads fit the 4K budget after explicit
  compaction where required. See `runs/prompt-recovery-audit.json`.
- Real adaptive five-question run, with Docker public tests and measured energy:
  `runs/progrouter-recovery-v2/stream.json`.
- All five independent EvalPlus judgments were available: 3 passed (60%).
  Four coordinator workflows finished; one was explicitly stalled after repeated
  incorrect `same_chars` code and failed public examples. There were no protocol
  errors, prompt-limit failures or worker-limit failures in this pilot.
- Grading: `reports-progrouter-recovery-v2.json`.

This small pilot validates the corrected execution paths, not a 100-question
accuracy claim. Model coding mistakes and genuine exhausted budgets remain
possible. The old 100-task results used faulty token counts and should not be
treated as validation of the corrected implementation. Start a fresh stream;
do not resume its learned checkpoint for a clean comparison.

## Rerun the same 100 prepared questions

Run in your Docker-enabled terminal with qwen-local activated. No new calibration
or dataset preparation is required solely for these code fixes: calibration
samples used Ollama's actual token counts.

```bash
python benchmarks.py run --public prepared/humaneval-100 --config examples/ollama_routed_pilot.json --enable-tests --output runs/progrouter-100-fixed
python benchmarks.py evaluate --private evaluator-data/references-100.json --runs runs/progrouter-100-fixed --image progrouter-coding-evaluator --output reports-progrouter-100-fixed.json
python report_experiment.py --reference-stream runs/progrouter-100-fixed/stream.json --reports reports-progrouter-100-fixed.json --output analysis-progrouter-100-fixed.json
```

Use new output paths if these already exist. Pilot budgets/learning thresholds
remain your configured values; no parameters were silently tuned to hidden tests.
The interrupted `runs/progrouter-recovery-pilot` diagnostic is not the final pilot
and must not be used as a complete comparison stream.
