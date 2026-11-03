# ProgRouter implementation audit — 2026-11-02

Scope: the supplied ProgRouter PDF, Sections 3.2–3.4 and Appendix D, versus this
independent coding-domain implementation. Smaller worker/coordinator models do
not invalidate the routing equations, but do affect workflow reliability and
the data the predictor can learn from. This is not an author-code reproduction.

## Findings and corrections

1. The fixed-7B smoke configuration was never an adaptive-router run. It is kept
   as a coordinator integration check; routed validation has its own config.
2. JSON-object mode alone allowed invalid field types from small models. Ollama
   models now opt into schema-constrained output, with ordinary validators and
   bounded retries retained. Repair prompts no longer demonstrate the rejected
   output back to the model. Official transport documentation:
   https://github.com/ollama/ollama/blob/main/docs/capabilities/structured-outputs.mdx
3. The coordinator repeatedly requested disabled tests or repeated completed
   reviews. Completed-attempt context, explicit review guidance and a bounded
   duplicate-subtask rejection now make this actionable. These are engineering
   safeguards, not additional paper equations; they never label a solution correct.
4. Structural requirements previously appeared only after code creation, changing
   the denominator. Their universe is now fixed before/after a transition.
   Syntax failures and failed worker execution are observable failure signals.
5. Candidate identity used to be appended after a long semantic state, where
   MiniLM truncation could remove it. Candidate/role/subtask now precede a compact
   dynamic state representation. This remains a locally chosen domain adapter.
6. Executed worker calls that exhausted protocol retries previously contributed
   cost but no exploration target. They now create failed transitions; selected
   actions with no call executed are not treated as observed model outcomes.
7. Retry cost estimates now use the actual repair messages and warm residency
   after the first attempt. The estimator still conservatively reserves maximum
   output length; it does not predict exact future generated-token counts.
8. Router diagnostics and duplicate historical source code no longer consume
   model context. Current code and all decision/feedback entries are retained in
   prompts; complete original states and diagnostics remain in audit artifacts.
9. Checkpoints now identify the feature representation and encoder configuration.
   Older prototype checkpoints may be incompatible and should not be reused.
10. A coordinator can still claim verification in natural language without having
    run tests. Ledger verification metadata and independent grading, rather than
    that prose, are authoritative. Hidden correctness remains null during inference.

## Equation and procedure mapping

| Paper | Implementation and assessment |
|---|---|
| Eq. 2, hierarchical progress | `ProgressScorer`: base + completion + trend + quality; clipped to [0,1]. Regime rules/weights are reconstruction choices. |
| Eqs. 3–5, conditional progress prediction | Separate structured and semantic random forests condition on state and candidate model. |
| Eq. 6, meta-gating | Decision tree over two path predictions; out-of-fold training avoids directly fitting the meta-tree to in-sample forest outputs. |
| Eq. 7, virtual queue | Once per terminal task; persistent Q; measured whole-task energy in measured mode. Failed measurement freezes Q explicitly. |
| Eq. 9, progress gap | V * (1-g) * predicted gain. |
| Eqs. 10–12, routing | Exponential time/energy penalties, queue pressure and argmax over feasible candidates. |
| Exploration/update | Uniform feasible exploration; only exploration transitions enter training; cold-start exploration, periodic refits and decaying epsilon persist across tasks. |
| Eq. 13, measurement | NVML board power sampled at 100 ms; trapezoidal idle-adjusted integration with negative noise clipped. |
| Eq. 14, call estimation | Calibrated load + input-token + generated-token terms; maximum output reservation and conservative constant floors are explicit additions. |
| Eq. 15, target | g(after)-g(before), including executed failed actions. Hidden evaluator outputs are never labels. |

The score/queue equations are implemented consistently with the PDF. The PDF does
not specify enough numeric scorer, predictor, epsilon-schedule, and stabilization
parameters to establish exact implementation equivalence. Those choices require
experimental validation, rather than merely an algebraic unit test.

## Real validation performed

- Four installed Q4_K_M models served through local Ollama. Matching tokenizer
  files were provisioned for local admission checks (not new model weights).
- 32 real cold/warm calibration generation calls with NVML sampling:
  `runs/audit-calibration.json`. These are pilot measurements, not final budgets.
- Corrected fixed-7B triangular workflow: `runs/audit-fixed-003/ledger.json`.
  It finished after coder and reviewer steps. Generated code was not executed.
- Final routed diagnostic stream: `runs/audit-routing-final/validation_summary.json`.
  Two synthetic tasks finished; four exploration transitions fitted the predictor;
  one subsequent step used learned-score exploitation. Both tasks measured GPU
  energy. Checkpoint save/restore preserved targets, queue, counters and RNG.
- `runs/audit-routing-final/equation_audit.json`: five independently replayed seeded
  selections, five realized gains, two queue updates and four saved training
  targets passed consistency checks. Eq. 2, Eq. 9 and Eqs. 10–12 were recomputed
  from recorded values. This verifies arithmetic/procedure, not predictor accuracy.
- The final stream used all four candidates in scoring, with 7B, 0.5B and 3B
  selected. 1.5B was exercised in calibration, not selected in that short stream.
- 66 local regression tests passed, covering scoring, candidate conditioning,
  actual tree fitting, failure samples, token admission, persistence and mocked
  container contracts. Ollama reported the 7B alias 100% on GPU with 4096 context.

The diagnostic stream deliberately uses min_samples=4 and epsilon=0 after cold
start, permissive 100000 J/120 s task budgets, and a 500 J queue target to exercise
the control paths. These are NOT proposed scientific evaluation settings. The
four training samples are insufficient evidence of a useful routing policy.

## What remains blocked or unvalidated

Docker is installed and running, but this process receives permission denied on
`/var/run/docker.sock`; noninteractive sudo requires a password. No socket
permissions or group memberships were changed and no unrestricted execution
fallback was used. Consequently:

- Actual public-test execution and independent EvalPlus grading were not run.
- No HumanEval Plus accuracy, benchmark energy saving, or quality/cost advantage
  is established by these diagnostics.
- Public-doctest extraction and the evaluator interface are locally tested, but
  the real EvalPlus image/API/container path still needs end-to-end verification.

The small pilot calibration has short prompts, initial-load variance and desktop
GPU interference. It is not a validated cost model across 4K contexts. Repeat
calibration with stable idle power, representative prompts and held-out calibration
calls before choosing budgets. The calibration utility now includes settling,
distinct prompt prefixes, server timings and in-sample fit diagnostics for future
runs. The saved pilot predates those additional calibration diagnostics.

For the main experiment, also predeclare stabilization thresholds, report warm-up
and common steady-state tasks, pin artifacts, and execute baseline/ablation runs.
MasRouter/CASCADIA/Educated Guessing ports are not implemented here; the generated
fixed and random configurations are initial comparison policies only.

## Reproduce the diagnostic checks

```bash
python -B -m unittest discover -v
python audit_trace.py runs/audit-routing-final/00000 runs/audit-routing-final/00001 --trusted-checkpoint runs/audit-routing-final/00001/router.joblib
```

Only load the checkpoint you generated and trust. To start another real diagnostic
stream (actual GPU inference; no candidate execution):

```bash
python validate_routing.py --calibration runs/audit-calibration.json --output runs/another-routing-validation
```

Use fresh output paths. Full provisioning/benchmark commands are in
`IMPLEMENTATION_RUNBOOK.md`.
