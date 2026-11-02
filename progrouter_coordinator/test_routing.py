import dataclasses
import math
from pathlib import Path
import tempfile
import unittest

from progress import ProgressScorer
from routing import ModelOption, ProgressRouter, RoutingConfig
from workflow import CoordinatorWorkflow, LimitReached, Model, Task, Limits
from test_progress import TestEncoder, ledger
from test_workflow import ScriptedClient, dispatch, worker, finish


class Predictor:
    def __init__(self, gains=None):
        self.gains = gains or {'small': .1, 'strong': .8}
        self.requests, self.observations = [], []
        self.ready = True

    def predict(self, state, score, candidate):
        self.requests.append(candidate)
        return {'ready': self.ready, 'gain': self.gains[candidate.model] if self.ready else None}

    def observe(self, *args, **kwargs):
        self.observations.append((args, kwargs))
        return False


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.scorer = ProgressScorer(TestEncoder())
        self.score = self.scorer.score(__import__('progress').coding_observation(ledger()))
        self.small = ModelOption(Model('http://localhost/v1', 'small'), 1, 1)
        self.strong = ModelOption(Model('http://localhost/v1', 'strong'), 5, 2)

    def router(self, **overrides):
        cfg = dict(energy_budget=20, time_budget=20, average_energy=10, tradeoff=100, epsilon=0)
        cfg.update(overrides)
        return ProgressRouter({'coder': [self.small, self.strong]}, Predictor(), RoutingConfig(**cfg))

    def select(self, router, score=None):
        return router.select('coder', ledger(), subtask='repair', score=score or self.score)

    def test_exact_equation_and_all_candidates_receive_dispatch(self):
        router = self.router(queue=2)
        selected = self.select(router)
        self.assertEqual(selected.model, 'strong')
        self.assertEqual([c.model for c in router.predictor.requests], ['small', 'strong'])
        self.assertTrue(all(c.subtask == 'repair' for c in router.predictor.requests))
        row = router.last_decision['candidates'][1]
        expected = 100*(1-self.score.value)*.8 - 2*(5-10) - math.exp(2/20)*2 - math.exp(5/20)*5
        self.assertAlmostEqual(row['score'], expected)
        self.assertFalse(router.last_decision['exploration'])

    def test_progress_gap_queue_and_budget_pressure_change_choice(self):
        router = self.router()
        self.assertEqual(self.select(router).model, 'strong')
        self.assertEqual(self.select(router, dataclasses.replace(self.score, value=1)).model, 'small')
        self.assertEqual(self.select(self.router(queue=20)).model, 'small')
        pressure = self.router(tradeoff=11)
        self.assertEqual(self.select(pressure).model, 'strong')
        pressure.energy_used = 14
        pressure.time_used = 15
        self.assertEqual(self.select(pressure).model, 'small')

    def test_feasibility_exhaustion_and_retry_charging(self):
        timed = self.router(time_budget=1)
        self.assertEqual(self.select(timed).model, 'small')
        router = self.router(energy_budget=2, time_budget=2)
        self.assertEqual(self.select(router).model, 'small')
        self.assertFalse(router.last_decision['candidates'][1]['feasible'])
        router.charge_attempt(self.small.model)
        router.charge_attempt(self.small.model)
        with self.assertRaises(LimitReached):
            router.charge_attempt(self.small.model)
        with self.assertRaises(LimitReached):
            self.select(router)
        router.finish_task()
        router.begin_task()
        self.assertEqual(router.energy_used, 0)

    def test_cold_start_and_seeded_exploration(self):
        a, b = self.router(epsilon=1), self.router(epsilon=1)
        a.predictor.ready = b.predictor.ready = False
        draws = [self.select(a).model for _ in range(30)]
        self.assertEqual(draws, [self.select(b).model for _ in range(30)])
        self.assertEqual(set(draws), {'small', 'strong'})
        self.assertTrue(a.last_decision['exploration'])
        self.assertEqual(a.last_decision['reason'], 'cold_start')
        a.predictor.ready = True
        self.select(a)
        self.assertEqual(a.last_decision['reason'], 'epsilon')

    def test_queue_updates_once_and_carries_between_tasks(self):
        router = self.router(average_energy=2)
        self.select(router)
        router.charge_attempt(self.strong.model)
        update = router.finish_task()
        self.assertEqual(update['queue_after'], 3)
        self.assertIsNone(router.finish_task())
        router.begin_task()
        self.select(router)
        self.assertEqual(router.last_decision['queue'], 3)
        router.finish_task()  # zero worker cost still constitutes a served task
        self.assertEqual(router.queue, 1)
        router.begin_task()
        router.finish_task()
        self.assertEqual(router.queue, 0)
        self.assertEqual(router.completed_tasks, 3)

    def test_epsilon_decay_only_after_exploration_samples(self):
        router = self.router(epsilon=1, epsilon_decay=.5, epsilon_min=.2)
        from progress import Candidate
        candidate = Candidate('coder', 'small', 'repair')
        router.observe_transition(ledger(), self.score, candidate, self.score, exploration=False)
        self.assertEqual(router.epsilon, 1)
        for expected in (.5, .25, .2, .2):
            router.observe_transition(ledger(), self.score, candidate, self.score, exploration=True)
            self.assertEqual(router.epsilon, expected)
        self.assertEqual(router.collected_samples, 4)

    def test_checkpoint_preserves_queue_rng_predictor_and_decay(self):
        router = self.router(epsilon=1, epsilon_decay=.5, average_energy=0)
        from progress import Candidate
        self.select(router)
        router.charge_attempt(router.active_option.model)
        router.observe_transition(ledger(), self.score, Candidate('coder', 'small', 'repair'), self.score, exploration=True)
        router.finish_task()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'router.joblib'
            router.save(path)
            restored = self.router(epsilon=1, epsilon_decay=.5, average_energy=0)
            restored.restore(path)
            self.assertEqual(restored.queue, router.queue)
            self.assertEqual(restored.epsilon, router.epsilon)
            self.assertEqual(restored.completed_tasks, 1)
            self.assertEqual(len(restored.predictor.observations), 1)
            router.begin_task()
            restored.begin_task()
            self.assertEqual([self.select(router).model for _ in range(20)], [self.select(restored).model for _ in range(20)])
            with self.assertRaises(ValueError):
                self.router(tradeoff=999).restore(path)

    def test_invalid_configuration(self):
        for settings in ({'energy_budget':0}, {'epsilon':2}, {'queue':float('nan')}, {'epsilon_decay':0}, {'epsilon_min':.5}):
            with self.assertRaises(ValueError):
                self.router(**settings)
        with self.assertRaises(ValueError):
            ModelOption(self.small.model, -1, 1)

    def run_engine(self, router, responses):
        engine = CoordinatorWorkflow(ScriptedClient(responses), self.small.model, router,
            {'coder': 'implement'}, Limits(max_worker_steps=4), progress_scorer=self.scorer)
        with tempfile.TemporaryDirectory() as folder:
            result = engine.run(Task('t', 'Implement f'), Path(folder)/'run')
            import json
            events = [json.loads(line) for line in (Path(folder)/'run/events.jsonl').read_text().splitlines()]
        return result, events

    def test_workflow_selection_logging_and_exploration_training(self):
        router = self.router(epsilon=1)
        result, events = self.run_engine(router, [dispatch('coder', 'repair'), worker(code='def f(): return 1'), finish()])
        self.assertEqual(result['status'], 'coordinator_finished')
        self.assertEqual(len(router.predictor.observations), 1)
        args, kwargs = router.predictor.observations[0]
        self.assertTrue(kwargs['exploration'])
        self.assertEqual(args[2].subtask, 'repair')
        self.assertEqual(args[0]['history'], [])
        selected = result['history'][0]['model']
        self.assertEqual(result['metrics']['routing']['estimated_worker_energy'], 1 if selected == 'small' else 5)
        self.assertTrue(any(e['event'] == 'routing_decision' for e in events))

    def test_retry_budget_stops_before_extra_inference(self):
        router = self.router(energy_budget=1)
        result, events = self.run_engine(router, [dispatch('coder'), 'bad JSON'])
        self.assertEqual(result['status'], 'limit_reached')
        self.assertEqual(result['metrics']['llm_calls'], 2)
        self.assertEqual(router.energy_used, 1)

    @unittest.skipUnless(__import__('importlib.util', fromlist=['find_spec']).find_spec('sklearn'), 'requires sklearn')
    def test_cli_routing_configuration_and_checkpoint(self):
        import json
        from unittest.mock import patch
        import run
        replies = ScriptedClient([dispatch('coder'), worker(code='def f(): return 1'), finish()])
        cfg = {'coordinator': {'base_url': 'http://localhost/v1', 'model': 'coordinator'},
               'roles': {'coder': 'implement'},
               'routing': {'energy_budget': 20, 'time_budget': 20, 'average_energy': 0},
               'candidates': {'coder': [{'model': dataclasses.asdict(self.small.model), 'energy': 1, 'seconds': 1}]}}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'config.json').write_text(json.dumps(cfg))
            (root/'task.json').write_text(json.dumps({'task_id': 't', 'problem': 'Implement f'}))
            for i in range(2):
                argv = ['run.py', '--task', str(root/'task.json'), '--config', str(root/'config.json'), '--output', str(root/str(i))]
                if i:
                    argv += ['--router-checkpoint', str(root/'0/router.joblib')]
                replies = ScriptedClient([dispatch('coder'), worker(code='def f(): return 1'), finish()])
                with patch('sys.argv', argv), patch('progress.SentenceEncoder', return_value=TestEncoder()), patch('run.HTTPChatClient', return_value=replies), patch('builtins.print'):
                    self.assertEqual(run.main(), 0)
                self.assertTrue((root/str(i)/'predictor.joblib').is_file())
                self.assertTrue((root/str(i)/'router.joblib').is_file())
                import joblib
                state = joblib.load(root/str(i)/'router.joblib')
                self.assertEqual(state['queue'], i+1)
                self.assertEqual(state['completed_tasks'], i+1)
                self.assertEqual(state['collected_samples'], i+1)
                from progress import ProgressPredictor
                predictor = ProgressPredictor.load(root/str(i)/'predictor.joblib', TestEncoder())
                self.assertEqual(predictor.seen, i+1)

    @unittest.skipUnless(__import__('importlib.util', fromlist=['find_spec']).find_spec('sklearn'), 'requires sklearn')
    def test_real_predictor_refits_online_across_tasks(self):
        from progress import ProgressPredictor
        predictor = ProgressPredictor(TestEncoder(), min_samples=4, update_every=1, n_estimators=4)
        router = ProgressRouter({'coder': [self.small, self.strong]}, predictor,
            RoutingConfig(20, 20, 0, epsilon=1, epsilon_decay=1))
        for i in range(5):
            result, events = self.run_engine(router, [dispatch('coder'), worker(code='def f(): return 1'), finish()])
            self.assertEqual(result['status'], 'coordinator_finished')
            self.assertEqual(predictor.seen, i+1)
            refit = [e for e in events if e['event'] == 'predictor_observation'][0]['payload']['refitted']
            self.assertEqual(refit, i >= 3)
        self.assertIsNotNone(predictor.meta)
        self.assertEqual(router.completed_tasks, 5)
        self.assertGreater(router.queue, 0)

    def test_transport_failure_still_charges(self):
        router = self.router(energy_budget=1, average_energy=0)
        result, events = self.run_engine(router, [dispatch('coder'), ConnectionError('failed')])
        self.assertEqual(result['status'], 'error')
        self.assertEqual(router.energy_used, 1)
        self.assertEqual(router.queue, 1)
        self.assertEqual(router.completed_tasks, 1)
        self.assertEqual(len([e for e in events if e['event'] == 'virtual_queue_update']), 1)


if __name__ == '__main__':
    unittest.main()
