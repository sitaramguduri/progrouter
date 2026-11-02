from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from workflow import (CoordinatorWorkflow, DEFAULT_ROLES, DockerTests, FixedModelSelector,
                      HTTPChatClient, Limits, Model, ProtocolError, Reply, Task, json_object)


def dispatch(role, subtask="Do the requested work", **kwargs):
    return {"action": "dispatch", "role": role, "subtask": subtask,
            "reason": "next", **kwargs}


def worker(message="findings", code=None, run_tests=False):
    return {"message": message, "code": code, "run_tests": run_tests}


def finish():
    return {"action": "finish", "reason": "ready", "final_answer": "proposed answer"}


class ScriptedClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def complete(self, model, messages, timeout):
        self.requests.append({"model": model.model, "messages": json.loads(json.dumps(messages))})
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return Reply(value if isinstance(value, str) else json.dumps(value),
                     {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, "stop")


class FakeTests:
    def __init__(self):
        self.calls = []

    def run(self, code, tests, timeout):
        self.calls.append((code, tests))
        return {"status": "completed", "result": {"passed": 0, "total": 1,
                "failures": [{"error": "AssertionError"}]}}


class WorkflowTests(unittest.TestCase):
    def run_case(self, responses, *, tests=None, limits=None):
        client = ScriptedClient(responses)
        coordinator = Model("http://localhost:8000/v1", "coordinator")
        models = {r: Model("http://localhost:8001/v1", "worker-" + r) for r in DEFAULT_ROLES}
        engine = CoordinatorWorkflow(client, coordinator, FixedModelSelector(models),
                                     DEFAULT_ROLES, limits, tests)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        out = Path(folder.name) / "run"
        result = engine.run(Task("test", "Implement f()", ("assert f() == 1",)), out)
        events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
        return result, client, events, out

    def test_coordinator_revises_plan_and_reuses_role(self):
        tool = FakeTests()
        result, client, events, out = self.run_case([
            dispatch("coder", plan=["implement"]), worker(code="def f(): return 0", run_tests=True),
            dispatch("reviewer"), worker("Visible assertion failed; return 1"),
            dispatch("coder", "Apply reviewer feedback", plan=["repair", "finish"]),
            worker(code="def f(): return 1"), finish(),
        ], tests=tool)
        self.assertEqual(result["status"], "coordinator_finished")
        self.assertEqual([s["role"] for s in result["history"]], ["coder", "reviewer", "coder"])
        self.assertEqual(result["plan"], ["repair", "finish"])
        self.assertEqual(result["metrics"]["llm_calls"], 7)
        self.assertEqual(result["metrics"]["token_usage"]["total_tokens"], 105)
        self.assertEqual((out / "final.py").read_text(), "def f(): return 1")
        # The coordinator actually receives failed visible feedback on its next call.
        coordinator_after = json.loads(client.requests[2]["messages"][1]["content"])
        self.assertEqual(coordinator_after["ledger"]["history"][0]["visible_feedback"]["result"]["passed"], 0)
        # The second coder receives earlier review and original code, not an isolated task.
        second_coder = json.loads(client.requests[5]["messages"][1]["content"])
        self.assertIn("Visible assertion failed", second_coder["ledger"]["history"][1]["message"])
        transitions = [e["payload"] for e in events if e["event"] == "worker_transition"]
        self.assertEqual(len(transitions[0]["state_after"]["history"]), 1)
        self.assertEqual(len(transitions[2]["state_before"]["history"]), 2)
        self.assertIsNone(result["metrics"]["final_correctness"])

    def test_unknown_role_is_repaired_before_worker_executes(self):
        result, client, _, _ = self.run_case([
            dispatch("unknown"), dispatch("analyst"), worker(), finish()
        ])
        self.assertEqual(result["status"], "coordinator_finished")
        self.assertEqual([r["model"] for r in client.requests],
                         ["coordinator", "coordinator", "worker-analyst", "coordinator"])

    def test_bad_protocol_cannot_loop_indefinitely(self):
        result, client, _, _ = self.run_case(["bad", "also bad"])
        self.assertEqual(result["status"], "error")
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(result["history"], [])

    def test_call_limit_preserves_latest_code(self):
        result, client, _, out = self.run_case([
            dispatch("coder"), worker(code="def f(): return 1")
        ], limits=Limits(max_llm_calls=2))
        self.assertEqual(result["status"], "limit_reached")
        self.assertEqual(result["stop_reason"], "max_llm_calls")
        self.assertTrue((out / "final.py").exists())
        self.assertEqual(len(client.requests), 2)

    def test_worker_limit_still_allows_final_coordinator_decision(self):
        result, _, _, _ = self.run_case([
            dispatch("analyst"), worker(), finish()
        ], limits=Limits(max_worker_steps=1))
        self.assertEqual(result["status"], "coordinator_finished")

    def test_worker_limit_blocks_additional_dispatch(self):
        result, client, _, _ = self.run_case([
            dispatch("analyst"), worker(), dispatch("coder")
        ], limits=Limits(max_worker_steps=1))
        self.assertEqual(result["status"], "limit_reached")
        self.assertEqual(result["metrics"]["worker_steps"], 1)
        self.assertEqual(len(client.requests), 3)

    def test_no_test_execution_is_not_a_pass(self):
        result, _, _, _ = self.run_case([
            dispatch("coder"), worker(code="def f(): return 1", run_tests=True), finish()
        ])
        feedback = result["history"][0]["visible_feedback"]
        self.assertEqual(feedback["status"], "disabled")
        self.assertIsNone(feedback["passed"])
        self.assertIsNone(result["metrics"]["final_correctness"])

    def test_failed_http_call_is_logged_and_run_is_saved(self):
        result, _, events, out = self.run_case([ConnectionError("private server details")])
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["metrics"]["llm_calls"], 1)
        self.assertTrue(any(e["event"] == "llm_call_failed" for e in events))
        self.assertNotIn("private server details", (out / "ledger.json").read_text())

    def test_hidden_answer_cannot_enter_task_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "task.json"
            path.write_text(json.dumps({"task_id": "a", "problem": "p", "answer": "secret"}))
            with self.assertRaisesRegex(ValueError, "hidden"):
                Task.load(path)

    def test_json_fence_handling(self):
        self.assertEqual(json_object('```json\n{"a": 1}\n```'), {"a": 1})
        with self.assertRaises(ProtocolError):
            json_object("``````")

    def test_docker_missing_never_executes_on_host(self):
        with patch("workflow.subprocess.Popen", side_effect=FileNotFoundError):
            result = DockerTests().run("raise RuntimeError('should not run')", ("assert 1",), 1)
        self.assertEqual(result["status"], "unavailable")

    def test_existing_run_is_not_overwritten(self):
        result, _, _, out = self.run_case([dispatch("analyst"), worker(), finish()])
        engine = CoordinatorWorkflow(ScriptedClient([]), Model("http://localhost/v1", "c"),
            FixedModelSelector({"analyst": Model("http://localhost/v1", "w")}),
            {"analyst": "analyze"})
        with self.assertRaisesRegex(ValueError, "existing"):
            engine.run(Task("again", "p"), out)

    def test_http_client_and_workflow_contract(self):
        replies = iter([dispatch("analyst"), worker("Approach is ready"), finish()])
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                content = json.dumps({"choices": [{"message": {"content": json.dumps(next(replies))},
                                                  "finish_reason": "stop"}],
                                      "usage": {"total_tokens": 7}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            def log_message(self, *_args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            model = Model(f"http://127.0.0.1:{server.server_port}/v1", "contract-model")
            engine = CoordinatorWorkflow(HTTPChatClient(), model,
                FixedModelSelector({"analyst": model}), {"analyst": "Analyze requirements"})
            with tempfile.TemporaryDirectory() as folder:
                result = engine.run(Task("http-test", "Analyze this problem"), Path(folder) / "run")
            self.assertEqual(result["status"], "coordinator_finished")
            self.assertEqual(result["metrics"]["token_usage"]["total_tokens"], 21)
            self.assertEqual(len(requests), 3)
            self.assertTrue(all(path == "/v1/chat/completions" for path, _ in requests))
            self.assertEqual(requests[0][1]["max_tokens"], 2048)
            final_input = json.loads(requests[2][1]["messages"][1]["content"])
            self.assertEqual(final_input["ledger"]["history"][0]["message"], "Approach is ready")
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()


if __name__ == "__main__":
    unittest.main()
