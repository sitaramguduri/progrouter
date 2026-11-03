# Missing-code and public-check fixes

Coder requests now use an implementation-only prompt with a non-null code example,
short notes, required imports/helpers, and no copied problem docstring. The coder
response schema requires a Python function definition. Invalid proposals still
retain the last accepted code and produce static repair feedback. The coordinator
cannot finish a coding task without accepted code; this is checked in Python as
well as in the transport schema. Recovery, call, energy and time limits remain
bounded. Models can still fail to produce correct code; no fallback answer or
forced stronger-model selection has been introduced.

Public examples of the form `>>> f(x) == expected` without an output line are
converted to assertions. Explicit output examples, setup and assertions are
preserved. Output-less expressions without a usable oracle are skipped, including
malformed prose such as `>>> f(x) [1, 2]`. Plain prose examples are not inferred.
Protocol identifier: prompt-doctests-v2. Existing prepared tasks contain old check
strings, so fresh preparation is required. Fresh private references are also
required because task hashes include public checks. Old reports stay unchanged.

Validation: 87 unit tests pass with scripted clients and authored fixtures.
No live inference or benchmark candidate execution was performed for this change.
Routing policy and model pool are unchanged. Public-check changes alter progress
feedback; use a fresh stream without an old router checkpoint.

Run from the coordinator directory in qwen-local:

```bash
python benchmarks.py prepare --benchmark humaneval_plus --dataset evaluator-data/HumanEvalPlus.jsonl --public prepared/humaneval-100-v2 --private evaluator-data/references-100-v2.json --seed 0 --limit 100
python benchmarks.py run --public prepared/humaneval-100-v2 --config examples/ollama_routed_checked.json --enable-tests --output runs/progrouter-100-v2
python benchmarks.py evaluate --private evaluator-data/references-100-v2.json --runs runs/progrouter-100-v2 --image progrouter-coding-evaluator --output reports-progrouter-100-v2.json
python report_experiment.py --reference-stream runs/progrouter-100-v2/stream.json --reports reports-progrouter-100-v2.json --output analysis-progrouter-100-v2.json
```

Use unused output paths. These commands are documented, not executed.
