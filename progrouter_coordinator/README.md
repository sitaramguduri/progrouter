# ProgRouter reconstruction

An independent implementation of the coordinator workflow and Sections 3.2–3.4
of *ProgRouter: Online Progress-Guided Orchestration for Multi-Agent LLM Workflows
under Quality-Cost Tradeoffs* (arXiv:2608.25992v2). This is not author code or a
validated reproduction of the reported benchmark results. Defaults not specified
by the paper are reconstruction choices, documented below.

## Components

| Component | Implementation |
|---|---|
| Coordinator ledger, dynamic dispatch, role reuse, plan revision | `workflow.py` |
| Configurable HTTP model endpoints | `HTTPChatClient`; existing local servers required |
| Multi-view progress score, Eq. 2 | `progress.py` |
| Structured/semantic predictors and meta-learner, Eqs. 3–6 | `progress.py` |
| Candidate comparison and routing, Eqs. 9–12 | `routing.py` |
| Epsilon exploration, online refits, persistent queue, Eq. 7 | `routing.py` |
| NVML power integration, Eq. 13, task budgets | `energy.py`, workflow integration |
| Dataset adapters, stream runner, held-out evaluation | `benchmarks.py`, `grading.py`, `sandbox.py` |
| Context-dependent cost estimator, Eq. 14 | Not implemented; candidate admission uses calibrated per-call profiles |
| Framework | Standalone Python; paper uses AG2 |
| Benchmark reproduction, model comparisons, branching/cascading experiments | Not performed |

## Quick offline check

Python 3.10+ for the core. No third-party libraries are needed for the fixed-model
scripted demo or original workflow tests. Optional routing tests require NumPy and
scikit-learn; energy monitoring requires NVIDIA drivers and `nvidia-ml-py`.

```bash
python -m unittest -v test_workflow.py
python run.py --demo --output runs/scripted-demo
```

The demo uses scripted responses for the triangular-number task. It runs no LLM
inference or generated code and cannot be combined with routing or measurement.
Use a fresh output directory for each run.

## Real inference and optional visible tests

Start your model servers separately. Configure served model names and base URLs
ending in `/v1`; servers must implement `POST /chat/completions` and return string
message content. Model weights/servers are not downloaded or launched by this code.
Authentication comes from `api_key_env` (default `LOCAL_LLM_API_KEY`), never JSON
configuration. Example model names must be replaced.

```bash
python run.py --task examples/task.json --config examples/config.json --output runs/fixed-001
```

The coordinator chooses a role/subtask, the selector chooses its model, the worker
receives the shared ledger, and feedback returns to the coordinator. Roles, JSON
schemas, prompts, temperatures, token limits and retry limits are our choices.
Malformed JSON receives bounded repair attempts; HTTP failures are not retried.
Truncated responses are rejected. A coordinator finish decision is not a correctness
label. Steps, calls, prompt characters, request timeouts and elapsed time are bounded.
The input-size limit counts characters, not input tokens or dollars.

Generated Python runs only when explicitly enabled, in disposable Docker containers:

```bash
docker pull python:3.12-slim
python run.py --task examples/task.json --config examples/config.json --enable-tests --output runs/tested-001
```

Containers have no network, read-only roots and input mounts, a nonroot user,
dropped capabilities, resource/output/time limits and forced cleanup. There is no
host-execution fallback. Visible tests are fallible feedback, not held-out evaluation
or an adversarially hardened evaluator. Without the flag, requested tests report
`disabled`, never `passed`.

## Progress scorer and predictor

Install optional progress dependencies in your environment:

```bash
pip install -r requirements-progress.txt
```

`SentenceEncoder` uses locally cached MiniLM weights by default. Explicitly provision
weights once where downloads are available:

```python
from progress import SentenceEncoder
encoder = SentenceEncoder(local_files_only=False)
```

Add `--enable-progress` to fixed-model runs to record progress. Routing enables it
automatically. The coding adapter uses visible assertions as observable requirements;
new code invalidates earlier passes. `complete` means visible milestones satisfied,
never hidden-test correctness. Without visible requirements it cannot infer
completion. It does not infer fulfilled subtasks from worker prose. Other domains
should supply their own `Observation` adapter; the bundled coding adapter provides
little supervision for math/QA tasks with no visible milestones.

The scorer uses completion fraction, a three-transition mean completion delta,
embedding change and signed structural changes. Weights `(base,completion,trend,quality)`
are invalid=(0,0,0,0), recoverable=(.1,.3,.05,.05), partial_success=(.4,.4,.05,.05),
complete=(1,0,0,0). Weights/window are configurable. Scores are clipped to [0,1].
State quality is signed structural change times `(.5 + .5 * semantic_distance)`;
novelty alone gets no reward. These rules are reconstruction choices requiring
calibration. Coordinator `state_summary` is optional; a ledger summary is used when
absent. Code and latest worker output are included in the semantic representation.

The predictor has two random forests, one over structured scores/role/model/history
features and one over sentence embeddings including candidate/subtask context.
A decision tree combines their outputs. Meta-training uses out-of-fold predictions
with fold-local vectorizers. Defaults: 64 trees, depth 8, up to five folds,
meta-tree depth 3, minimum leaf size 2, seed 0. Cold predictions are explicitly null.

Only actual exploration transitions train the predictor, with target
`g(after)-g(before)`. Defaults: minimum 12 samples, refit every four collected samples,
bounded 2,000-sample buffer. Models/buffer persist across tasks. `ProgressPredictor`
also exposes `predict`, `observe`, `save`, and `load` for programmatic use. Only load
trusted joblib checkpoints, using the same encoder configuration.

## Routing and persistent task streams

`examples/routing_config.json` defines candidate pools with model configuration,
`energy` (estimated joules per attempt) and `seconds` (estimated seconds per attempt).
**Example costs, budgets, tradeoff and model names are placeholders.** Calibrate them
before running experiments. Optional `predictor` config sets predictor parameters.

```bash
python run.py --task examples/task.json --config examples/routing_config.json --enable-tests --output runs/routed-001
```

For each feasible candidate, routing evaluates:

```text
V * (1 - progress) * predicted_gain
- Q * (candidate_energy - average_energy)
- exp((used_seconds + candidate_seconds) / time_budget) * candidate_seconds
- exp((used_energy + candidate_energy) / energy_budget) * candidate_energy
```

Candidates whose estimates exceed remaining budgets are excluded. Maximum score
wins; ties follow configuration order. With probability epsilon, selection is
uniform among feasible candidates. Cold start uses uniform exploration until the
predictor is ready. No feasible candidate stops the run with `limit_reached`.
Every worker attempt, including protocol retries and HTTP failures, reserves its
profile cost before inference. These are estimates, not measured energy.

Epsilon after `n` collected exploration samples is
`max(epsilon_min, epsilon * epsilon_decay ** n)`. Defaults are decay=.99 and
minimum=0; the example sets minimum=.01. The paper does not supply this schedule.
Exploitation does not add training samples or decay epsilon.

At every terminal outcome (finish, abort, limit, error), Eq. 7 runs once:

```text
Q_next = max(0, Q_current + task_cost - average_energy)
```

`queue` initializes Q. Per-task costs reset; Q, predictors, training buffer,
exploration counters and RNG persist. `finish_task()` is idempotent. Tasks with
zero worker attempts count toward the task average. In estimated mode, task cost
is estimated worker energy. In measured mode, it is measured whole-task energy.
Measurement failures freeze Q and mark the queue update unavailable.

`router.joblib` atomically saves the entire stream state at task exit. Continue with:

```bash
python run.py --task next-task.json --config examples/routing_config.json --enable-tests --router-checkpoint runs/routed-001/router.joblib --output runs/routed-002
```

Use the same routing config, candidate pools and encoder. Measured-mode checkpoints
also validate measurement configuration. Predictor-only initialization remains
available via `--predictor-checkpoint`, but resets Q, epsilon counters and RNG.
Do not combine the two checkpoint flags. Reusing one router object in successive
workflow runs also preserves stream state. Mid-task recovery, process-kill recovery,
concurrent writers and exactly-once replay by task ID are not implemented.

## Measured energy and task budgets

Install `requirements-energy.txt` on the machine hosting inference. NVML reports
local board power; choose dedicated GPUs for **both workers and coordinator**.
It cannot attribute remote serving or separate concurrent users' GPU consumption.
Provide one independently measured idle baseline per selected GPU.

```json
{
  "energy_measurement": {"devices": [0], "idle_watts": [35.0], "interval": 0.1},
  "task_budget": {"energy_joules": 1000, "seconds": 300}
}
```

Numbers above illustrate the schema and are not calibrated values. `devices` are
NVML physical indices, not remapped CUDA-visible indices. The monitor samples
power in watts and integrates trapezoids at monotonic timestamps. Incremental
power is `max(0, power-idle)` per GPU; negative noise is clipped. It records both
incremental and gross joules. Zero idle baseline intentionally measures gross
energy. Model loading within a request is included; model preloading before the
workflow starts is outside the measurement window.

Measurement covers the workflow from initialization through final inference/tool
work, including coordinator calls, retries, embedding work and visible tools that
use the selected GPUs. CPU/other devices and independent final evaluation are outside
this cost boundary. `energy.jsonl` records raw power observations and cumulative
energy. `energy_boundary` events provide call-boundary totals, not precise component
attribution. Sampling failures produce null final energy, stop further inference,
and never silently substitute estimated costs.

Use `examples/measured_config.json` for measured routing. Replace model names,
profiles, GPUs, baselines and budgets. Its zero baseline is deliberately uncalibrated.
Set routing `cost_basis="measured_task"`; routing budgets then also define task
budgets if `task_budget` is omitted. Explicit budgets must match. Routing uses
measured consumption plus candidate estimates for admission; the queue uses final
measured energy. The default `estimated_worker` basis preserves earlier behavior.
A measured task budget may also be used with the fixed selector.

Energy budgets stop further scheduling when observed consumption reaches the limit.
They are **not a hardware energy cutoff**: sampling and an in-flight HTTP call can
overshoot, and a client timeout cannot guarantee cancellation of remote generation.
Measured overshoot remains in metrics and queue accounting. Task elapsed-time
budgets constrain request/tool timeouts independently of the existing workflow
limit. Final metrics report budget exceedances, including finalization overhead.

Read-only telemetry smoke test, without launching a workload:

```bash
python energy.py --devices 0 --idle-watts 0 --seconds 2 --output /tmp/power-samples.jsonl
```

This command does not establish inference energy or an idle baseline.

## Benchmark preparation, execution and independent grading

`benchmarks.py` reads **local exports**; it does not download datasets. Supported
formats are JSON lists, JSONL (optionally gzip), and the original ASQA split dictionary.
Use a reproducible dataset version and retain the reported source SHA256.

Preparation writes public task files/manifest and a separate private evaluator
file (mode 0600), rejecting a private path inside the public directory. Only
`task_id`, `problem` and `visible_tests` enter the workflow. Hidden tests, canonical
solutions, math answers and ASQA annotations never enter model requests or progress
training. The runner itself accepts no private-reference path.

```bash
python benchmarks.py prepare --benchmark humaneval_plus --dataset /data/HumanEvalPlus.jsonl --public prepared/humaneval --private evaluator-data/humaneval.json
python benchmarks.py run --public prepared/humaneval --config examples/measured_config.json --enable-tests --output runs/humaneval-stream
```

The stream runner executes each task in a fresh run directory and carries the
complete router checkpoint to the next task. `--router-checkpoint` starts from an
earlier stream. It preserves failures and stops if a routed task fails to produce
its checkpoint. Public hashes are validated before execution. `stream.json` records
task coverage and run directories. No automatic warm-up/steady-state split or
baseline comparison is claimed by the default summary.

| Adapter | Public input | Independent evaluator |
|---|---|---|
| `humaneval` | Function prompt/examples | Held-out `check(entry_point)` tests |
| `humaneval_plus` | Function prompt/examples | EvalPlus base and expanded input sets, official special oracles |
| `mbpp` | Description and explicitly selected visible tests | Remaining original and challenge assertions |
| `mbpp_plus` | EvalPlus prompt | EvalPlus base/expanded inputs, official MBPP deserialization/oracles |
| `math500` | Problem only | Math-Verify expression equivalence of final response against private answer |
| `asqa` | Ambiguous question plus separately supplied documents | NLI-based per-citation entailment precision |

For raw MBPP, `--visible-count 1` exposes the first test (including its function
interface) and holds the remaining tests out. Default visible count is zero.
At least one original test must remain held out. This split is an explicit local
protocol choice, not a claim of identical evaluation to the paper. HumanEval and
EvalPlus expose only examples already in the prompt.

ASQA requires `--documents corpus.json`, mapping task IDs to lists of
`{"id": 1, "title": "...", "text": "..."}`. Supply an independently retrieved
corpus; annotation contexts/long answers are never used as source retrieval.
The model is asked for one claim per line with numeric source citations. This is
a supplied-context QA variant; live retrieval is not implemented.
The evaluator counts each cited source separately, including invalid IDs in the
denominator; no citations scores zero. It reports uncited claims separately.
Its sentence/line splitting and single-source NLI judge are explicit reconstruction
choices, not the original ASQA ROUGE/QA metrics or a verified reproduction of the
paper's citation protocol. For math/QA, configure suitable worker roles in `roles`
and matching candidate pools rather than relying on the default coding roles.

Final evaluation happens in a separate command, after the task stream is complete:

```bash
docker build -f Dockerfile.evaluation -t progrouter-evaluator .
python benchmarks.py evaluate --private evaluator-data/humaneval.json --runs runs/humaneval-stream --image progrouter-evaluator --timeout 120 --output reports-humaneval.json
```

Build the image separately where dependency downloads are available; pin its digest
for reproducibility. The provided Dockerfile includes EvalPlus, Math-Verify and
Transformers. ASQA additionally requires a compatible NLI model/tokenizer bundled
in the evaluator image, then `--nli-model /models/nli --entailment-label entailment`
(or the actual entailment label from its `id2label`). No runtime model downloads
are permitted. You may use `python:3.12-slim` for raw HumanEval/MBPP assertions only.

Evaluation uses the terminal ledger's frozen `current_code`/`final_answer`, not a
potentially edited `final.py`. Task hashes and IDs must match the private references;
complete matching stream coverage is required. Tests, reference-program execution,
math parsing and NLI loading all occur in constrained evaluator containers. No
candidate or reference code executes on the host through the evaluation command.
Docker absence, container timeout, missing dependencies/weights and invalid results
produce unknown correctness, not a pass. Container isolation protects the host;
the Python test harness is not a proof against adversarial grader manipulation.

Reports contain per-task correctness or citation metrics, input/reference/artifact
hashes, evaluator image, workflow status and costs. Aggregate pass rates distinguish
unevaluated tasks; the all-task rate is null if any task lacks a boolean judgment.
Mean measured energy is null if any measurement is missing. Evaluation does not
modify the inference ledger, predictor or queue, and its feedback never trains the
online router. `final_correctness` in the independent report is the authoritative
held-out judgment; the original inference ledger intentionally retains null.

## Artifacts and validation

Each run writes `events.jsonl` (prompts/replies/decisions/usage/errors/transitions),
`ledger.json`, and `final.py` when code exists. Routing also writes `router.joblib`;
the CLI writes a predictor-only checkpoint. Measured runs add `energy.jsonl`.
Task streams add `stream.json`; separate evaluation writes its own report.
No automatic resume or overwrite of nonempty run directories is supported.

```bash
python -m unittest discover -v
```

Tests cover workflow protocol/state, progress, real tree-model fitting, routing,
checkpoint continuation, power integration, measurement failures, budget accounting,
private/public separation, adapter schemas, independent grading, report validation
and Docker invocation contracts. External model/judge calls are mocked in unit tests.
A real NVML power-sampling smoke test passed in this environment. Docker is absent,
so container execution/image builds and end-to-end benchmark results are unvalidated.
MiniLM/NLI inference and inference energy attribution have not been validated.

Sources used for adapter/evaluator integration:

- [ProgRouter paper](https://arxiv.org/abs/2608.25992)
- [HumanEval record schema](https://github.com/openai/human-eval/blob/master/data/example_problem.jsonl)
- [EvalPlus official evaluator](https://github.com/evalplus/evalplus)
- [Google MBPP dataset](https://github.com/google-research/google-research/tree/master/mbpp)
- [MATH-500 dataset](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
- [Math-Verify](https://github.com/huggingface/Math-Verify)
- [ASQA data and original evaluation](https://github.com/google-research/language/tree/master/language/asqa)
