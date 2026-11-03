import tempfile
import unittest
from pathlib import Path

from code_checks import validate_code
from workflow import CoordinatorWorkflow, FixedModelSelector, Model, Task
from test_workflow import ScriptedClient, dispatch, worker, finish


class PublicTool:
    def __init__(self):
        self.calls = []
    def run(self, code, tests, timeout):
        self.calls.append(code)
        return {'status': 'completed', 'result': {'passed': len(tests), 'total': len(tests), 'failures': []}}


class CodeQualityTests(unittest.TestCase):
    def test_candidate_readiness_requires_role_specific_support(self):
        from dataclasses import replace
        from progress import Candidate, ProgressPredictor, ProgressScorer, coding_observation
        class Encoder:
            def encode(self, texts):
                return [[1., 0.] for _ in texts]
        encoder = Encoder()
        predictor = ProgressPredictor(encoder, min_samples=4, update_every=1,
                                      min_candidate_samples=2, n_estimators=4)
        ledger = {'task': {'problem': 'Implement f', 'visible_tests': []}, 'history': []}
        score = ProgressScorer(encoder).score(coding_observation(ledger))
        for _ in range(4):
            predictor.observe(ledger, score, Candidate('coder', 'small', 'implement'), score, exploration=True)
        self.assertTrue(predictor.predict(ledger, score, Candidate('coder', 'small', 'implement'))['ready'])
        self.assertFalse(predictor.predict(ledger, score, Candidate('coder', 'strong', 'implement'))['ready'])
        for _ in range(2):
            predictor.observe(ledger, score, Candidate('coder', 'strong', 'implement'), replace(score, value=.7), exploration=True)
        self.assertTrue(predictor.predict(ledger, score, Candidate('coder', 'strong', 'implement'))['ready'])
        self.assertFalse(predictor.predict(ledger, score, Candidate('reviewer', 'strong', 'review'))['ready'])

    def test_placeholders_syntax_and_wrong_interface(self):
        problem = 'def target(x):\n    """Implement target."""\n'
        for source in (None, '', 'null', 'None', '...', 'def target(x): pass', 'def target(', 'def wrong(x): return x'):
            with self.subTest(source=source):
                self.assertEqual(validate_code(source, problem)['status'], 'invalid')
        self.assertEqual(validate_code('def target(x): return x', problem)['status'], 'valid')

    def test_invalid_proposal_does_not_destroy_previous_code(self):
        model = Model('http://localhost/v1', 'm')
        responses = [dispatch('coder'), worker(code='def f(): return 1'),
                     dispatch('coder', 'repair'), worker(code='null'), finish()]
        engine = CoordinatorWorkflow(ScriptedClient(responses), model,
            FixedModelSelector({'coder': model}), {'coder': 'code'})
        with tempfile.TemporaryDirectory() as folder:
            ledger = engine.run(Task('t', 'Implement f'), Path(folder)/'run')
        self.assertEqual(ledger['current_code'], 'def f(): return 1')
        self.assertEqual(ledger['history'][1]['rejected_code'], 'null')
        self.assertEqual(ledger['history'][1]['visible_feedback']['status'], 'invalid_code')

    def test_rejected_proposal_does_not_invalidate_accepted_code_test_provenance(self):
        from workflow import prompt_ledger
        ledger = {'current_code': 'def f(): return 1', 'history': [
            {'code': 'def f(): return 1', 'visible_feedback': {'status': 'completed',
                'result': {'passed': 1, 'total': 1, 'failures': []}}},
            {'code': None, 'rejected_code': 'null', 'visible_feedback': {'status': 'invalid_code'}}]}
        self.assertEqual(prompt_ledger(ledger)['latest_visible_feedback']['status'], 'completed')

    def test_changed_code_runs_public_checks_even_without_worker_request(self):
        model = Model('http://localhost/v1', 'm')
        tool = PublicTool()
        engine = CoordinatorWorkflow(ScriptedClient([dispatch('coder'), worker(code='def f(): return 1'), finish()]),
            model, FixedModelSelector({'coder': model}), {'coder': 'code'}, tests=tool)
        with tempfile.TemporaryDirectory() as folder:
            ledger = engine.run(Task('t', 'Implement f', ('assert f() == 1',)), Path(folder)/'run')
        self.assertEqual(len(tool.calls), 1)
        self.assertEqual(ledger['history'][0]['test_trigger'], 'automatic_code_change')

    def test_finish_without_code_is_repaired_by_dispatch(self):
        model = Model('http://localhost/v1', 'm')
        client = ScriptedClient([dispatch('coder'), worker(code='null'), finish(),
            dispatch('coder', 'write full code'), worker(code='def f(): return 1'), finish()])
        engine = CoordinatorWorkflow(client, model, FixedModelSelector({'coder': model}), {'coder': 'code'})
        with tempfile.TemporaryDirectory() as folder:
            ledger = engine.run(Task('t', 'Implement f'), Path(folder)/'run')
        self.assertEqual(ledger['status'], 'coordinator_finished')
        self.assertEqual(ledger['current_code'], 'def f(): return 1')
        coder_prompt = client.requests[1]['messages'][0]['content']
        self.assertIn('omit copied problem docstrings', coder_prompt)
        self.assertNotIn('"code":null', coder_prompt)

    def test_rejected_code_is_not_executed(self):
        model = Model('http://localhost/v1', 'm')
        tool = PublicTool()
        engine = CoordinatorWorkflow(ScriptedClient([dispatch('coder'), worker(code='def f(', run_tests=True), finish()]),
            model, FixedModelSelector({'coder': model}), {'coder': 'code'}, tests=tool)
        with tempfile.TemporaryDirectory() as folder:
            ledger = engine.run(Task('t', 'Implement f', ('assert f() == 1',)), Path(folder)/'run')
        self.assertEqual(tool.calls, [])
        self.assertIsNone(ledger['current_code'])


if __name__ == '__main__':
    unittest.main()
