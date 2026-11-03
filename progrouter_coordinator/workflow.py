"""Ledger-driven coordinator workflow; no learned routing or hidden evaluator.

Python 3.10+, standard library only. Local HTTP chat endpoints are configured
per role. This is a documented reconstruction, not the authors' implementation.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Protocol
import urllib.error
import urllib.request
import uuid


class ProtocolError(ValueError):
    pass


class LimitReached(RuntimeError):
    pass


def prompt_ledger(ledger):
    """Keep decision history, but exclude router diagnostics from model prompts.

    Full history/code/diagnostics remain in the persisted ledger and events.
    The current code is supplied once; superseded code stays in audit artifacts.
    """
    result = dict(ledger)
    result['history'] = []
    for step in ledger.get('history', []):
        compact = {k: v for k, v in step.items() if k not in {'routing', 'code', 'progress_gain'}}
        compact['code_updated'] = step.get('code') is not None
        result['history'].append(compact)
    result['history_representation'] = 'all decisions and feedback; code supplied in current_code; router diagnostics in events'
    return result


def json_object(text: str) -> dict:
    text = text.strip()
    try:
        if text.startswith("```") and text.endswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        value = json.loads(text)
    except (ValueError, IndexError) as exc:
        raise ProtocolError("Return exactly one valid JSON object.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("Response must be a JSON object.")
    return value


@dataclasses.dataclass(frozen=True)
class Task:
    task_id: str
    problem: str
    visible_tests: tuple[str, ...] = ()

    @classmethod
    def load(cls, path: Path) -> "Task":
        data = json.loads(path.read_text(encoding="utf-8"))
        # Enforce separation from reference answers and held-out tests.
        if set(data) - {"task_id", "problem", "visible_tests"}:
            raise ValueError("Task accepts only task_id, problem, visible_tests. "
                             "Keep reference answers/hidden tests in a separate evaluator.")
        if not isinstance(data.get("task_id"), str) or not data["task_id"].strip():
            raise ValueError("task_id must be a nonempty string")
        if not isinstance(data.get("problem"), str) or not data["problem"].strip():
            raise ValueError("problem must be a nonempty string")
        tests = data.get("visible_tests", [])
        if not isinstance(tests, list) or not all(isinstance(t, str) for t in tests):
            raise ValueError("visible_tests must be a list of Python strings")
        return cls(data["task_id"], data["problem"], tuple(tests))


@dataclasses.dataclass(frozen=True)
class Model:
    base_url: str
    model: str
    api_key_env: str = "LOCAL_LLM_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 2048
    json_mode: bool = False
    context_tokens: int | None = None
    tokenizer_name: str | None = None
    schema_mode: bool = False
    response_schema: dict | None = None

    def __post_init__(self):
        if not self.model or self.max_tokens <= 0:
            raise ValueError("Each model needs a model name and positive max_tokens")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an HTTP(S) URL, normally ending in /v1")
        if self.context_tokens is not None and (self.context_tokens <= self.max_tokens or not self.tokenizer_name):
            raise ValueError('Context admission requires a tokenizer and space for input/output')


@dataclasses.dataclass(frozen=True)
class Limits:
    max_worker_steps: int = 10
    max_llm_calls: int = 30
    max_seconds: float = 600.0
    request_timeout: float = 120.0
    max_prompt_chars: int = 120_000
    protocol_retries: int = 1

    def __post_init__(self):
        if min(self.max_worker_steps, self.max_llm_calls, self.max_seconds,
               self.request_timeout, self.max_prompt_chars) <= 0:
            raise ValueError("Limits must be positive")
        if self.protocol_retries < 0:
            raise ValueError("protocol_retries must be nonnegative")


@dataclasses.dataclass
class Reply:
    content: str
    usage: dict = dataclasses.field(default_factory=dict)
    finish_reason: str | None = None


class ChatClient(Protocol):
    def complete(self, model: Model, messages: list[dict], timeout: float) -> Reply: ...


class HTTPChatClient:
    """Uses /chat/completions on an existing local vLLM-style endpoint.

    No automatic HTTP retries: failed calls may already have incurred inference.
    Authentication is read from environment and never written to configuration.
    """
    def complete(self, model: Model, messages: list[dict], timeout: float) -> Reply:
        if model.context_tokens is not None:
            from token_budget import check_context
            check_context(model, messages)
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(model.api_key_env)
        if key:
            headers["Authorization"] = "Bearer " + key
        payload = {"model": model.model, "messages": messages,
                           "temperature": model.temperature,
                           "max_tokens": model.max_tokens}
        if model.response_schema is not None:
            payload['response_format'] = {'type': 'json_schema',
                'json_schema': {'name': 'workflow_response', 'schema': model.response_schema}}
        elif model.json_mode:
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload).encode()
        request = urllib.request.Request(model.base_url.rstrip("/") + "/chat/completions",
                                         data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            # Avoid returning headers, keys, or arbitrary server error bodies.
            raise RuntimeError(f"Chat endpoint returned HTTP {exc.code}") from None
        choice = data["choices"][0]
        content = choice["message"].get("content")
        if not isinstance(content, str):
            raise RuntimeError("Chat endpoint did not return string message content")
        return Reply(content, data.get("usage") or {}, choice.get("finish_reason"))


class FixedModelSelector:
    """Explicit placeholder boundary for the paper's separate model router."""
    def __init__(self, workers: dict[str, Model]):
        if not workers:
            raise ValueError("Configure at least one worker role")
        self.workers = workers

    def select(self, role: str, ledger: dict, **context) -> Model:
        return self.workers[role]


class VisibleTestTool(Protocol):
    def run(self, code: str, tests: tuple[str, ...], timeout: float) -> dict: ...


class DisabledTests:
    def run(self, code: str, tests: tuple[str, ...], timeout: float) -> dict:
        return {"status": "disabled", "passed": None,
                "message": "Visible-test execution is disabled; no correctness observed."}


class DockerTests:
    """Run generated Python only in a disposable constrained Docker container.

    Requires a locally available Python Docker image. Never falls back to host
    execution. Tool stdout is an observed signal, not a trusted hidden evaluator.
    """
    def __init__(self, image: str = "python:3.12-slim", timeout: float = 10.0,
                 max_output_bytes: int = 65536):
        if timeout <= 0 or max_output_bytes <= 0:
            raise ValueError("Tool timeout and output limit must be positive")
        self.image, self.timeout, self.max_output_bytes = image, timeout, max_output_bytes

    def run(self, code: str, tests: tuple[str, ...], timeout: float) -> dict:
        if not tests:
            return {"status": "no_visible_tests", "passed": None}
        name = "progrouter-test-" + uuid.uuid4().hex
        script = '''import json, traceback
ns = {"__name__": "__candidate__"}
tests = json.load(open("/work/tests.json"))
passed, failures = 0, []
try:
    exec(compile(open("/work/solution.py").read(), "solution.py", "exec"), ns)
    for index, test in enumerate(tests):
        try:
            exec(compile(test, "visible_test", "exec"), ns)
            passed += 1
        except BaseException as error:
            failures.append({"index": index, "error": repr(error)[:1000]})
except BaseException as error:
    failures.append({"index": None, "error": repr(error)[:1000]})
print("__WORKFLOW_TEST_RESULT__=" + json.dumps({"passed": passed, "total": len(tests), "failures": failures}))
'''
        with tempfile.TemporaryDirectory(prefix="progrouter-tests-") as folder:
            root = Path(folder)
            root.chmod(0o755)
            for filename, contents in {"solution.py": code, "tests.json": json.dumps(tests),
                                       "run.py": script}.items():
                file = root / filename
                file.write_text(contents, encoding="utf-8")
                file.chmod(0o644)
            cmd = ["docker", "run", "--rm", "--pull=never", "--name", name,
                   "--network", "none", "--read-only", "--cpus", "1",
                   "--memory", "512m", "--memory-swap", "512m", "--pids-limit", "64",
                   "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                   "--user", "65534:65534", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
                   "--mount", f"type=bind,src={root},dst=/work,readonly",
                   self.image, "python", "-I", "-B", "/work/run.py"]
            status = "error"
            try:
                with (root / "output.log").open("wb") as output:
                    process = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT)
                    deadline = time.monotonic() + min(timeout, self.timeout)
                    while process.poll() is None:
                        if (root / "output.log").stat().st_size > self.max_output_bytes:
                            status = "output_limit"
                            break
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            status = "timeout"
                            break
                        try:
                            process.wait(timeout=min(0.05, remaining))
                        except subprocess.TimeoutExpired:
                            pass
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                    elif (root / "output.log").stat().st_size > self.max_output_bytes:
                        status = "output_limit"
                    else:
                        status = "completed" if process.returncode == 0 else "error"
                with (root / "output.log").open("rb") as output:
                    output.seek(0, 2)
                    output.seek(max(0, output.tell() - 16000))
                    text = output.read().decode("utf-8", errors="replace")
                observed = None
                for line in text.splitlines():
                    if line.startswith("__WORKFLOW_TEST_RESULT__="):
                        try:
                            observed = json.loads(line.split("=", 1)[1])
                        except ValueError:
                            pass
                if status == "completed" and observed is None:
                    status = "error"
                return {"status": status, "result": observed, "output_tail": text}
            except FileNotFoundError:
                return {"status": "unavailable", "passed": None,
                        "message": "Docker is unavailable. No code was executed."}
            finally:
                try:
                    subprocess.run(["docker", "rm", "-f", name], timeout=5,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   check=False)
                except (OSError, subprocess.TimeoutExpired):
                    pass


COORDINATOR_PROMPT = """You coordinate an iterative problem-solving workflow.
Choose the next worker role and subtask from the current ledger; revise your plan
when evidence reveals failures. You may reuse any role, change order, or request
recovery work. There is no prescribed sequence of roles. The model selector is
separate: choose a role, never a model. Treat worker outputs as fallible evidence.
Visible tests are partial feedback. Passing them does not prove final correctness.
Return exactly one JSON object, no Markdown or reasoning outside it.
Start with { and end with }. Never prefix it with Dispatch:, Finish:, or Abort:.
Choose one of these object shapes according to the action field:
{"action":"dispatch", "role":"one available role", "subtask":"specific work", "plan":["updated plan item"], "reason":"brief dispatch justification"}
{"action":"finish", "plan":["final plan"], "final_answer":"final response", "reason":"why ready to finish"}
{"action":"abort", "reason":"why unable to continue"}
You may include "state_summary": a compact summary of current progress and bottlenecks.
Dispatch at least one worker before finishing. A finish decision expresses your
judgment of readiness; it is not a correctness label. Preserve code in the ledger
unless a worker supplies an updated full implementation.
The payload reports execution_capabilities. Disabled or unavailable tests cannot
be made available by repeatedly asking a worker to run them. If code has been
reviewed and no actionable defect remains, finish with an explicit statement of
unverified correctness. Do not repeat an unchanged review or implementation task.
When code is needed, dispatch implementation rather than repeated solution outlines.
Every history entry is a completed attempt, not a future task. Use its output.
Once requirements are understood, dispatch a coder to WRITE FULL CODE. Do not
ask the analyst to restate the problem. You have no tool for enabling tests.
"""

WORKER_PROMPT = """You are a specialized worker in a ledger-driven workflow.
Perform only the coordinator's requested subtask, using the task and prior ledger.
Previous outputs can be wrong; use visible feedback where available.
Return exactly one JSON object with keys:
{"message":"your analysis or findings", "code":null, "run_tests":false}
Start with { and end with }; do not add labels or Markdown outside the object.
For a code update, replace null with the COMPLETE Python implementation, not a
patch or fenced block. For review/analysis without a code change, use null.
Set run_tests=true to request visible tests on the current or updated code.
Do not claim that hidden tests passed; you have no access to them.
"""

DEFAULT_ROLES = {
    "analyst": "Analyze the problem, requirements, and algorithmic approach.",
    "coder": "Implement or repair the full Python solution.",
    "reviewer": "Review the current solution, identify errors, and request visible checks.",
}


class CoordinatorWorkflow:
    def __init__(self, client: ChatClient, coordinator: Model,
                 selector: FixedModelSelector, role_descriptions: dict[str, str],
                 limits: Limits | None = None, tests: VisibleTestTool | None = None,
                 progress_scorer=None, progress_predictor=None, energy_monitor=None, task_budget=None):
        if set(selector.workers) != set(role_descriptions):
            raise ValueError("Worker models and role descriptions must have identical keys")
        self.client, self.coordinator, self.selector = client, coordinator, selector
        self.roles = role_descriptions
        self.limits = limits or Limits()
        self.tests = tests or DisabledTests()
        if hasattr(selector, "predictor"):
            if progress_scorer is None:
                raise ValueError("ProgressRouter requires progress_scorer")
            if progress_predictor is not None and progress_predictor is not selector.predictor:
                raise ValueError("Router and workflow must share one predictor")
            progress_predictor = selector.predictor
        from energy import TaskBudget
        self.energy_monitor = energy_monitor
        self.task_budget = task_budget or TaskBudget()
        measured_routing = getattr(getattr(selector, 'config', None), 'cost_basis', None) == 'measured_task'
        if (self.task_budget.energy_joules is not None or measured_routing) and energy_monitor is None:
            raise ValueError('Measured energy budgets/routing require an energy monitor')
        if measured_routing and (self.task_budget.energy_joules != selector.config.energy_budget
                                or self.task_budget.seconds != selector.config.time_budget):
            raise ValueError('Measured routing and task budgets must match')
        self.progress_scorer, self.progress_predictor = progress_scorer, progress_predictor

    def run(self, task: Task, output_dir: Path) -> dict:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if any(output_dir.iterdir()):
            raise ValueError("Use a new empty output directory; existing results are preserved")
        if hasattr(self.selector, "begin_task"):
            self.selector.begin_task()
        self.started = time.monotonic()
        self.calls = 0
        self.usage: dict[str, int] = {}
        self.output_dir = output_dir
        self.journal = (output_dir / "events.jsonl").open("x", encoding="utf-8")
        ledger = {"task": dataclasses.asdict(task), "plan": [], "history": [],
                  "execution_capabilities": {"visible_tests_enabled": not isinstance(self.tests, DisabledTests),
                                             "hidden_tests_available": False},
                  "current_code": None, "status": "running", "final_answer": None}
        observation = progress_score = None
        completion_deltas = []
        try:
            if self.energy_monitor is not None:
                self.energy_monitor.start(output_dir / "energy.jsonl")
            if self.progress_scorer is not None:
                from progress import coding_observation
                observation = coding_observation(ledger)
                progress_score = self.progress_scorer.score(observation)
                ledger["progress"] = dataclasses.asdict(progress_score)
            self._event("started", {"task_id": task.task_id, "roles": self.roles,
                                   "limits": dataclasses.asdict(self.limits),
                                   "model_selection": "progress_budget" if hasattr(self.selector, "pools") else "fixed_per_role",
                                   "model_names": {r: m.model for r, m in self.selector.workers.items()},
                                   "coordinator_model": self.coordinator.model})
            while True:
                self._check_limits()
                payload = {"available_roles": self.roles, "ledger": prompt_ledger(ledger),
                           "execution_capabilities": ledger["execution_capabilities"],
                           "completed_subtasks": [s['subtask'] for s in ledger['history']],
                           "worker_steps_remaining": self.limits.max_worker_steps - len(ledger["history"])}
                if ledger['current_code'] and ledger['history'] and ledger['history'][-1]['role'] == 'reviewer':
                    payload['decision_context'] = ('A review of current_code is already available. '
                        'Finish if no actionable defect was identified, otherwise dispatch a concrete repair. '
                        'Disabled tools cannot be enabled by a worker. Finishing is not proof of correctness.')
                decision = self._structured(self.coordinator, COORDINATOR_PROMPT, payload,
                                            lambda d: self._validate_next_decision(d, ledger), "coordinator")
                if "state_summary" in decision:
                    ledger["state_summary"] = decision["state_summary"]
                if "plan" in decision:
                    ledger["plan"] = decision["plan"]
                self._event("coordinator_decision", decision)
                if decision["action"] == "abort":
                    ledger["status"] = "coordinator_aborted"
                    ledger["stop_reason"] = decision["reason"]
                    break
                if decision["action"] == "finish":
                    if not ledger["history"]:
                        raise ProtocolError("Coordinator finished before dispatching a worker")
                    ledger["status"] = "coordinator_finished"
                    ledger["final_answer"] = decision["final_answer"]
                    ledger["stop_reason"] = decision["reason"]
                    break
                if len(ledger["history"]) >= self.limits.max_worker_steps:
                    raise LimitReached("max_worker_steps")
                role = decision["role"]
                try:
                    model = self.selector.select(role, ledger, subtask=decision["subtask"], score=progress_score)
                finally:
                    if getattr(self.selector, "last_decision", None) is not None:
                        self._event("routing_decision", self.selector.last_decision)
                before = json.loads(json.dumps(ledger))
                if self.progress_predictor is not None and progress_score is not None:
                    from progress import Candidate
                    prediction = self.progress_predictor.predict(before, progress_score,
                        Candidate(role, model.model, decision["subtask"]))
                    self._event("progress_prediction", prediction)
                calls_before_worker = self.calls
                try:
                    worker = self._structured(model, WORKER_PROMPT,
                        {"role": role, "role_description": self.roles[role],
                         "subtask": decision["subtask"], "ledger": prompt_ledger(before)},
                        self._validate_worker, "worker:" + role)
                except Exception as error:
                    # Failed executed actions are observations too. Omitting them
                    # would bias exploration training toward successful replies.
                    if self.calls > calls_before_worker:
                        entry = {"step": len(ledger['history'])+1, "role": role,
                            "model": model.model, "subtask": decision['subtask'],
                            "message": 'Worker execution failed', "code": None,
                            "worker_error": type(error).__name__,
                            "visible_feedback": None}
                        ledger['history'].append(entry)
                        if self.progress_scorer is not None:
                            failed_observation = coding_observation(ledger)
                            failed_score = self.progress_scorer.score(failed_observation, observation, completion_deltas)
                            entry['progress_gain'] = failed_score.value-progress_score.value
                            ledger['progress'] = dataclasses.asdict(failed_score)
                            routing = getattr(self.selector, 'last_decision', None)
                            if routing is not None:
                                from progress import Candidate
                                entry['routing'] = routing
                                fitted = self.selector.observe_transition(before, progress_score,
                                    Candidate(role, model.model, decision['subtask']), failed_score,
                                    exploration=routing['exploration'])
                                self._event('predictor_observation', {'exploration': routing['exploration'], 'refitted': fitted})
                        self._event('worker_transition', {'state_before': before, 'entry': entry, 'state_after': ledger})
                    raise
                if worker["code"] is not None:
                    ledger["current_code"] = worker["code"]
                pending_error = None
                try:
                    self._check_time()
                except (LimitReached, RuntimeError) as error:
                    pending_error = error
                feedback = {"status": "skipped_budget_or_measurement", "passed": None} if worker["run_tests"] and pending_error else None
                if worker["run_tests"] and pending_error is None:
                    if ledger["current_code"] is None:
                        feedback = {"status": "no_code", "passed": None}
                    else:
                        feedback = self.tests.run(ledger["current_code"], task.visible_tests,
                                                  max(0.01, self._remaining_seconds()))
                entry = {"step": len(ledger["history"]) + 1, "role": role,
                         "model": model.model, "subtask": decision["subtask"],
                         "message": worker["message"], "code": worker["code"],
                         "visible_feedback": feedback}
                ledger["history"].append(entry)
                if self.progress_scorer is not None:
                    current_observation = coding_observation(ledger)
                    next_score = self.progress_scorer.score(current_observation, observation, completion_deltas)
                    completion_deltas.append(next_score.completion - progress_score.completion)
                    entry["progress_gain"] = next_score.value - progress_score.value
                    ledger["progress"] = dataclasses.asdict(next_score)
                    routing = getattr(self.selector, "last_decision", None)
                    if routing is not None:
                        entry["routing"] = routing
                        from progress import Candidate
                        fitted = self.selector.observe_transition(before, progress_score,
                            Candidate(role, model.model, decision["subtask"]), next_score,
                            exploration=routing["exploration"])
                        self._event("predictor_observation", {"exploration": routing["exploration"], "refitted": fitted})
                    observation, progress_score = current_observation, next_score
                self._event("worker_transition", {"state_before": before, "entry": entry,
                                                   "state_after": ledger})
                self._snapshot(ledger)
                if pending_error is not None:
                    raise pending_error
        except LimitReached as exc:
            ledger["status"], ledger["stop_reason"] = "limit_reached", str(exc)
        except Exception as exc:
            ledger["status"] = "error"
            # Don't serialize network error representations, which can contain URLs.
            ledger["stop_reason"] = type(exc).__name__
            if isinstance(exc, (ProtocolError, RuntimeError)):
                ledger["stop_reason"] += ": " + str(exc)[:300]
        finally:
            ledger['verification_status'] = {
                'visible_tests_executed': any((s.get('visible_feedback') or {}).get('status') == 'completed'
                                              for s in ledger['history']),
                'hidden_correctness': None,
                'note': 'Coordinator statements are not independent correctness judgments.'}
            energy_result = self.energy_monitor.stop() if self.energy_monitor is not None else {'status': 'disabled', 'energy_joules': None}
            if self.energy_monitor is not None and energy_result['status'] != 'measured' and ledger['status'] != 'error':
                ledger['status'], ledger['stop_reason'] = 'error', 'Energy measurement failed at task completion'
            elapsed = time.monotonic()-self.started
            energy_exceeded = (energy_result['energy_joules'] > self.task_budget.energy_joules
                if energy_result['energy_joules'] is not None and self.task_budget.energy_joules is not None else None)
            time_exceeded = elapsed > self.task_budget.seconds if self.task_budget.seconds is not None else None
            if (energy_exceeded or time_exceeded) and ledger['status'] in {'running', 'coordinator_finished'}:
                ledger['status'] = 'limit_reached'
                ledger['stop_reason'] = 'task_energy_joules' if energy_exceeded else 'task_seconds'
            if hasattr(self.selector, 'sync_measurement'):
                self.selector.sync_measurement(energy_result['energy_joules'], elapsed)
            if hasattr(self.selector, "finish_task"):
                update = self.selector.finish_task()
                self._event("virtual_queue_update", update)
            ledger["metrics"] = {"llm_calls": self.calls, "worker_steps": len(ledger["history"]),
                "elapsed_seconds": time.monotonic() - self.started, "token_usage": self.usage,
                "final_correctness": None, "energy_joules": energy_result["energy_joules"],
                "energy_measurement": energy_result, "task_budget": dataclasses.asdict(self.task_budget),
                "task_budget_exceeded": {"energy": energy_exceeded, "seconds": time_exceeded}}
            if hasattr(self.selector, "metrics"):
                ledger["metrics"]["routing"] = self.selector.metrics()
            self._event("finished", {"status": ledger["status"], "metrics": ledger["metrics"],
                                     "stop_reason": ledger.get("stop_reason")})
            self._snapshot(ledger)
            if ledger["current_code"] is not None:
                (output_dir / "final.py").write_text(ledger["current_code"], encoding="utf-8")
            try:
                if hasattr(self.selector, "finish_task"):
                    self.selector.save(output_dir / "router.joblib")
            finally:
                self.journal.close()
        return ledger

    def _structured(self, model, system, payload, validate, kind):
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        for attempt in range(self.limits.protocol_retries + 1):
            self._check_limits()
            if sum(len(m["content"]) for m in messages) > self.limits.max_prompt_chars:
                raise LimitReached("max_prompt_chars; ledger was not silently truncated")
            if model.context_tokens is not None:
                from token_budget import check_context
                check_context(model, messages)
            if kind.startswith("worker:") and hasattr(self.selector, "charge_attempt"):
                self.selector.charge_attempt(model, messages=messages)
            self.calls += 1
            start = time.monotonic()
            try:
                request_model = model
                if model.schema_mode:
                    from output_schemas import schema
                    request_model = dataclasses.replace(model, response_schema=schema(kind, self.roles))
                reply = self.client.complete(request_model, messages,
                                            min(self.limits.request_timeout, self._remaining_seconds()))
            except Exception as exc:
                self._event("llm_call_failed", {"kind": kind, "model": model.model,
                    "attempt": attempt, "error_type": type(exc).__name__,
                    "latency_seconds": time.monotonic() - start,
                    "usage": None})
                raise
            finally:
                if self.energy_monitor is not None:
                    measured = self.energy_monitor.snapshot()
                    if hasattr(self.selector, 'sync_measurement'):
                        self.selector.sync_measurement(measured['energy_joules'], time.monotonic()-self.started)
                    self._event('energy_boundary', {'phase': kind, 'model': model.model, 'attempt': attempt, 'measurement': measured})
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = reply.usage.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    self.usage[key] = self.usage.get(key, 0) + value
            self._event("llm_call", {"kind": kind, "model": model.model, "attempt": attempt,
                "messages": messages, "response": reply.content, "usage": reply.usage,
                "finish_reason": reply.finish_reason, "latency_seconds": time.monotonic() - start})
            if not kind.startswith("worker:"):
                self._check_time()
            try:
                if reply.finish_reason == "length":
                    raise ProtocolError("Response truncated; increase configured max_tokens")
                parsed = json_object(reply.content)
                validate(parsed)
                return parsed
            except ProtocolError as exc:
                self._event("protocol_error", {"kind": kind, "error": str(exc)})
                if attempt == self.limits.protocol_retries:
                    raise
                # Do not seed a repair with the rejected answer as another
                # assistant demonstration: small models often copy it verbatim.
                messages[0]['content'] += '\nYour previous response was rejected: '+str(exc)
                messages.append({"role": "user", "content": "Return a corrected JSON decision/output. " + str(exc)})
        raise AssertionError("Unreachable")

    def _validate_decision(self, d):
        if "state_summary" in d and not isinstance(d["state_summary"], str):
            raise ProtocolError("state_summary must be a string")
        action = d.get("action")
        if action not in {"dispatch", "finish", "abort"}:
            raise ProtocolError("action must be dispatch, finish, or abort")
        if not isinstance(d.get("reason"), str):
            raise ProtocolError("reason must be a string")
        if "plan" in d and (not isinstance(d["plan"], list) or
                             not all(isinstance(p, str) for p in d["plan"])):
            raise ProtocolError("plan must be a list of strings")
        if action == "dispatch":
            if d.get("role") not in self.roles:
                raise ProtocolError("role must be one of " + ", ".join(self.roles))
            if not isinstance(d.get("subtask"), str) or not d["subtask"].strip():
                raise ProtocolError("subtask must be a nonempty string")
        if action == "finish" and not isinstance(d.get("final_answer"), str):
            raise ProtocolError("finish requires string final_answer")

    def _validate_next_decision(self, decision, ledger):
        self._validate_decision(decision)
        if decision['action'] != 'dispatch':
            return
        history = ledger['history']
        if history:
            last = history[-1]
            if last['role'] == decision['role'] and last['subtask'].strip() == decision['subtask'].strip():
                feedback = last.get('visible_feedback') or {}
                previous_code = next((s['code'] for s in reversed(history[:-1]) if s.get('code') is not None), None)
                code_changed = last.get('code') is not None and last['code'] != previous_code
                if not code_changed and feedback.get('status') in {None, 'disabled', 'unavailable', 'no_visible_tests'}:
                    guidance = ('The review is complete. If no concrete defect remains, return '
                        '{"action":"finish","final_answer":"Implementation provided; correctness unverified.",'
                        '"reason":"Review completed; tests unavailable"}. Otherwise dispatch a concrete repair.'
                        if ledger.get('current_code') else
                        'Analysis is already available. Dispatch a coder to write the full Python implementation now.')
                    raise ProtocolError('Do not repeat the identical completed subtask without new feedback. '
                                        'You cannot install or enable test tools. '+guidance)

    @staticmethod
    def _validate_worker(d):
        if not isinstance(d.get("message"), str):
            raise ProtocolError("message must be a string")
        if "code" not in d or (d["code"] is not None and not isinstance(d["code"], str)):
            raise ProtocolError("code must be a string or null")
        if not isinstance(d.get("run_tests"), bool):
            raise ProtocolError("run_tests must be a JSON boolean")

    def _remaining_seconds(self):
        elapsed = time.monotonic() - self.started
        return min(self.limits.max_seconds, self.task_budget.seconds or self.limits.max_seconds) - elapsed

    def _check_time(self):
        if self.energy_monitor is not None:
            measured = self.energy_monitor.snapshot()
            if hasattr(self.selector, 'sync_measurement'):
                self.selector.sync_measurement(measured['energy_joules'], time.monotonic()-self.started)
            if measured['status'] != 'measured':
                raise RuntimeError('Energy measurement failed; no further inference permitted')
            if self.task_budget.energy_joules is not None and measured['energy_joules'] >= self.task_budget.energy_joules:
                raise LimitReached('task_energy_joules')
        if self._remaining_seconds() <= 0:
            raise LimitReached("task_seconds" if self.task_budget.seconds is not None and self.task_budget.seconds <= self.limits.max_seconds else "max_seconds")

    def _check_limits(self):
        self._check_time()
        if self.calls >= self.limits.max_llm_calls:
            raise LimitReached("max_llm_calls")

    def _event(self, event, payload):
        self.journal.write(json.dumps({"event": event, "payload": payload}, ensure_ascii=False) + "\n")
        self.journal.flush()

    def _snapshot(self, ledger):
        temp = self.output_dir / "ledger.tmp"
        temp.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(self.output_dir / "ledger.json")
