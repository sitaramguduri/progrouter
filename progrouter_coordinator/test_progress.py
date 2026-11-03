import dataclasses
import importlib.util
import unittest

from progress import (Candidate, Observation, ProgressPredictor, ProgressScorer,
                      coding_observation)


class TestEncoder:
    def encode(self, sentences):
        return [[1., float('repair' in s), float('strong' in s)] for s in sentences]


def ledger():
    return {"task": {"problem": "Implement f", "visible_tests": ["assert f() == 1"]},
            "history": [], "plan": [], "current_code": None}


class ScorerTests(unittest.TestCase):
    def setUp(self):
        self.scorer = ProgressScorer(TestEncoder())

    def test_regimes_and_regression(self):
        a = Observation('initial', {'a': False, 'b': False})
        b = Observation('repair', {'a': True, 'b': False})
        c = Observation('done', {'a': True, 'b': True})
        self.assertEqual(self.scorer.score(a).regime, 'recoverable')
        s = self.scorer.score(b, a)
        self.assertEqual(s.regime, 'partial_success')
        self.assertGreater(s.trend, 0)
        self.assertGreater(s.quality, 0)
        self.assertLess(self.scorer.score(a, b).quality, 0)
        self.assertEqual(self.scorer.score(c, b).value, 1)
        self.assertEqual(self.scorer.score(dataclasses.replace(c, valid=False)).value, 0)
        self.assertNotEqual(self.scorer.score(Observation('done')).regime, 'complete')

    def test_trend_window_and_no_novelty_reward_without_evidence(self):
        a, b = Observation('initial'), Observation('repair')
        self.assertEqual(self.scorer.score(b, a).quality, 0)
        self.assertAlmostEqual(self.scorer.score(b, a, [1., .3, .6]).trend, .3)

    def test_changed_code_invalidates_old_passes(self):
        state = ledger()
        state['history'] = [{'code': 'def f(): return 1', 'visible_feedback': {'status': 'completed',
            'result': {'passed': 1, 'total': 1, 'failures': []}}}]
        self.assertEqual(self.scorer.score(coding_observation(state)).regime, 'complete')
        state['history'].append({'code': 'def f(): return 2', 'visible_feedback': None})
        self.assertFalse(coding_observation(state).requirements['0'])
        state['status'] = 'coordinator_finished'
        self.assertNotEqual(self.scorer.score(coding_observation(state)).regime, 'complete')

    @unittest.skipUnless(importlib.util.find_spec('sklearn'), 'requires scikit-learn')
    def test_predictor_training_and_candidate_conditioning(self):
        model = ProgressPredictor(TestEncoder(), min_samples=12, update_every=12, n_estimators=8)
        state = ledger()
        score = self.scorer.score(coding_observation(state))
        weak, strong = Candidate('coder', 'weak', 'implement'), Candidate('coder', 'strong', 'repair')
        self.assertFalse(model.predict(state, score, weak)['ready'])
        model.observe(state, score, weak, score, exploration=False)
        self.assertEqual(len(model.samples), 0)
        for i in range(24):
            candidate = strong if i % 2 else weak
            after = dataclasses.replace(score, value=score.value + (.7 if i % 2 else -.1))
            model.observe(state, score, candidate, after, exploration=True)
        self.assertGreater(model.predict(state, score, strong)['gain'], model.predict(state, score, weak)['gain'])
        self.assertAlmostEqual(model.samples[1][2], .7)
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'predictor.joblib'
            model.save(path)
            restored = ProgressPredictor.load(path, TestEncoder())
            self.assertEqual(model.predict(state, score, strong), restored.predict(state, score, strong))
            self.assertEqual(restored.seen, model.seen)
        self.assertTrue(model.predict(state, score, Candidate('new', 'unseen', 'x'))['ready'])

    def test_workflow_records_realized_gain(self):
        import tempfile
        from pathlib import Path
        from workflow import CoordinatorWorkflow, FixedModelSelector, Model, Task
        from test_workflow import ScriptedClient, FakeTests, dispatch, worker, finish
        m = Model('http://localhost/v1', 'test')
        engine = CoordinatorWorkflow(ScriptedClient([dispatch('coder'), worker(code='def f(): return 0', run_tests=True), finish()]),
            m, FixedModelSelector({'coder': m}), {'coder': 'code'}, tests=FakeTests(), progress_scorer=self.scorer)
        with tempfile.TemporaryDirectory() as folder:
            result = engine.run(Task('t', 'Implement f', ('assert f() == 1',)), Path(folder)/'run')
        self.assertEqual(result['status'], 'coordinator_finished')
        self.assertIn('progress_gain', result['history'][0])
        self.assertNotEqual(result['progress']['regime'], 'complete')


if __name__ == '__main__':
    unittest.main()
