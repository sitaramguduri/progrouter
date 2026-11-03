# HumanEval Plus reconstruction runbook

The code is an independent reconstruction for four Qwen2.5-Coder workers and a
shared 7B coordinator. Local rule weights, stabilization thresholds, and hardware
differ from the paper. Unit tests do not establish benchmark performance.
Nothing below is run automatically. Use fresh output paths for each experiment.

## 1. Fixed-model workflow

The coordinator now receives explicit test availability and instructions against
repeated unchanged reviews. This improves its instructions but does not guarantee
LLM termination. Limits still stop loops; no forced finish claims success.

```bash
conda activate qwen-local
python run.py --task examples/task.json --config examples/ollama_config.json --enable-progress --output runs/ollama-smoke-003
```

Add `--enable-tests` only after Docker access is resolved. Do not run the Python
workflow with sudo. Without Docker tests, correctness is unverified. The fixed
smoke-test config retains its character guard; the generated routed configs below
add local-tokenizer context admission. CPU MiniLM is already the default.

## 2. Provision tokenizers and coding evaluation

These commands download files/packages or build an image; run them explicitly.

```bash
python setup_tokenizers.py
python -m pip install 'evalplus>=0.3,<0.4'
docker pull python:3.12-slim
docker build -f Dockerfile.coding -t progrouter-coding-evaluator .
```

Tokenizers are cached locally, with remote code disabled. Routed calls reserve
1024 output tokens plus a 64-token template margin and reject oversized prompts.
They do not silently truncate history. Compare local counts with actual Ollama
prompt usage and pin tokenizer/model revisions for a formal experiment. A full
ledger may exhaust 4K on longer tasks; context compaction is not implemented.

## 3. Explicit energy/latency calibration

This is a real GPU workload: it unloads resident models, measures idle power, loads
all four models, and generates calibration text. It never executes generated code.
Close other GPU workloads, including inference clients. Desktop GPU use still
introduces measurement noise. This calibration requires the one-resident-model
Ollama service settings and local NVML access.

```bash
python calibrate_models.py --device 0 --idle-seconds 5 --output calibration.json
```

Inspect samples and fitted coefficients. The script fits nonnegative load,
input-token and generated-token coefficients for both energy and time. This is a
small initial calibration, not a validated universal estimator. Validate it on
representative workflow prompts, additional lengths, and repeated runs.

Edit `calibration.json`: choose positive task energy/time budgets and an average
energy target under `routing`. They deliberately start as null; do not blindly
copy the paper's 4800 J. `tradeoff` also requires sensitivity analysis because
queue penalties scale with energy units.

Optional explicit reconstruction protocol:

```json
"evaluation_protocol": {"min_samples": 100, "max_epsilon": 0.05}
```

Add this at the top level. These numbers are examples, not paper parameters.
Choose thresholds before main evaluation, using a separate pilot. A task that
first reaches maturity is still warm-up; subsequent tasks are steady-state. A
stream may contain no steady-state tasks, which is reported rather than hidden.

```bash
python configure_routing.py --calibration calibration.json --output examples/ollama_routed.json
python create_baselines.py --config examples/ollama_routed.json --output examples/baselines
```

Generated configs use all four models for each role, JSON mode, measured whole-task
queue updates, CPU MiniLM, and token admission. Dynamic worker admission costs use
actual locally tokenized worker payloads and maximum output length. They assume
the preceding coordinator is resident; 7B calls are warm, other models cold.
Repair attempts reuse the same conservative reserved profile. Measurements, not
reservations, determine final task energy and queue updates. Admission is not a
hardware energy cutoff: in-flight calls can overshoot. Constant calibrated profiles
remain supported as conservative floors or as a simpler approximation.

## 4. Prepare private/public task splits

```bash
python export_humaneval_plus.py --output evaluator-data/HumanEvalPlus.jsonl
python benchmarks.py prepare --benchmark humaneval_plus --dataset evaluator-data/HumanEvalPlus.jsonl --public prepared/humaneval --private evaluator-data/references.json --seed 0
```

For a separate small pilot, add `--limit 5` and use different public/private paths.
Preparation extracts only docstring examples from the public prompt into executable
doctests. Canonical solutions and hidden inputs remain private. Original examples
remain in the prompt. Docstring groups count as one visible requirement each.
Prompts with no supported examples still receive structural code/syntax/function
milestones, but cannot be marked complete by those checks alone. Mathematical and
QA domain adapters remain outside this coding implementation.

## 5. Run and independently grade

These commands run real inference/generated code; Docker access is required.

```bash
python benchmarks.py run --public prepared/humaneval --config examples/ollama_routed.json --enable-tests --output runs/progrouter
python benchmarks.py evaluate --private evaluator-data/references.json --runs runs/progrouter --image progrouter-coding-evaluator --output reports-progrouter.json
```

Repeat run/evaluate with the generated baseline configs and distinct directories:
`examples/baselines/fixed-small.json`, `fixed-7b.json`, and `random.json`. Use the
same prepared tasks. The fixed baselines share coordinator and measured task
budgets; random uses uniform feasible-candidate exploration. These are initial
comparison policies, not reproductions of MasRouter/CASCADIA/Educated Guessing.

```bash
python report_experiment.py --reference-stream runs/progrouter/stream.json --reports reports-progrouter.json --output comparison.json
```

Include baseline report paths after `--reports` to compare them on the same
ProgRouter warm-up/steady-state membership. No stabilization phase is inferred
unless thresholds were explicitly configured. Reports retain unknown correctness
and missing measured energy rather than treating them as success or zero cost.

## Validation and remaining experimental work

```bash
python -B -m unittest discover -v
```

Tests use authored fixtures and mocked inference/container calls. Actual Docker
EvalPlus execution, tokenizers, calibration fit, coordinator behavior, and the
full benchmark need real validation. Pin evaluator image digest and package/model
versions before reported experiments. CPU energy is excluded; whole local GPU
energy includes coordinator and worker activity. Paper-specific baseline ports,
ablations, uncertainty intervals and publication-quality comparisons require
additional experimental work. No claimed pass rate or energy savings is supplied.
