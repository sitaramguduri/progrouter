import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks import adapt
from configure_routing import build
from progress import coding_observation, ProgressScorer
from public_checks import prompt_checks
from routing import ModelOption
from token_budget import check_context
from workflow import Model, LimitReached


class Encoder:
    def encode(self, texts):
        return [[1., 0.] for text in texts]


class IntegrationTests(unittest.TestCase):
    def test_schema_transport_constrains_worker_field_types(self):
        import io
        from workflow import HTTPChatClient
        from output_schemas import schema
        worker_schema = schema('worker:reviewer', {'coder': 'code'})
        response = {'choices': [{'message': {'content': '{"message":"ok","code":null,"run_tests":false}'},
                                  'finish_reason': 'stop'}]}
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as opened:
            HTTPChatClient().complete(Model('http://localhost/v1', 'm', response_schema=worker_schema), [], 1)
        body = json.loads(opened.call_args.args[0].data)
        self.assertEqual(body['response_format']['type'], 'json_schema')
        self.assertEqual(body['response_format']['json_schema']['schema']['properties']['code']['type'], ['string', 'null'])

    def test_progress_requirement_universe_is_stable(self):
        state = {'task': {'problem': 'Implement f', 'visible_tests': ['assert f() == 1']}, 'history': []}
        before = coding_observation(state)
        state['current_code'] = 'def f(): return 1'
        after = coding_observation(state)
        self.assertEqual(set(before.requirements), set(after.requirements))
        self.assertFalse(after.requirements['0'])

    def test_candidate_is_preserved_before_long_state_text(self):
        from progress import ProgressPredictor, Candidate
        class RecordingEncoder(Encoder):
            def encode(self, texts):
                self.last = texts[0]
                return super().encode(texts)
        encoder = RecordingEncoder()
        model = ProgressPredictor(encoder)
        state = {'task': {'problem': 'long '*2000, 'visible_tests': []}, 'history': []}
        model._features(state, ProgressScorer(encoder).score(coding_observation(state)), Candidate('coder', 'strong', 'implement'))
        self.assertTrue(encoder.last.startswith('Candidate model: strong'))

    def test_checkpoint_rejects_an_incompatible_encoder(self):
        from progress import ProgressPredictor
        import tempfile
        class AlternateEncoder(Encoder):
            identity = {'model_name': 'different'}
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder)/'predictor.joblib'
            ProgressPredictor(Encoder()).save(checkpoint)
            with self.assertRaises(ValueError):
                ProgressPredictor.load(checkpoint, AlternateEncoder())

    def test_exploration_worker_failure_produces_a_training_target(self):
        import tempfile
        from workflow import CoordinatorWorkflow, Task, Limits
        from progress import ProgressPredictor
        from routing import ProgressRouter, RoutingConfig
        from test_workflow import ScriptedClient, dispatch
        predictor = ProgressPredictor(Encoder(), min_samples=4)
        option = ModelOption(Model('http://localhost/v1', 'm'), 1, 1)
        router = ProgressRouter({'coder': [option]}, predictor, RoutingConfig(10, 10, 0))
        engine = CoordinatorWorkflow(ScriptedClient([dispatch('coder'), 'bad', 'bad']),
            option.model, router, {'coder': 'code'}, Limits(), progress_scorer=ProgressScorer(Encoder()))
        with tempfile.TemporaryDirectory() as folder:
            result = engine.run(Task('t', 'Implement f'), Path(folder)/'run')
        self.assertEqual(result['status'], 'error')
        self.assertEqual(predictor.seen, 1)
        self.assertLess(predictor.samples[0][2], 0)

    def test_public_examples_are_preserved_without_private_answers(self):
        prompt = 'def f(x):\n    """Double x.\n    >>> f(2)\n    4\n    """\n'
        row = {'task_id': 'HumanEval/0', 'prompt': prompt, 'canonical_solution': 'SECRET',
               'entry_point': 'f', 'base_input': [[3]], 'plus_input': [[99]]}
        public, private = adapt(row, 'humaneval_plus')
        self.assertEqual(len(public['visible_tests']), 1)
        self.assertNotIn('SECRET', json.dumps(public))
        self.assertNotIn('99', json.dumps(public))
        # Authored fixtures only; production candidate execution stays in Docker.
        import contextlib
        import io
        namespace = {'f': lambda x: x*2}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(public['visible_tests'][0], namespace)
            with self.assertRaises(AssertionError):
                exec(public['visible_tests'][0], {'f': lambda x: 0})
        self.assertEqual(prompt_checks('not python'), [])

    def test_progress_without_public_examples_never_claims_completion(self):
        state = {'task': {'problem': 'Implement f', 'visible_tests': []}, 'history': []}
        scorer = ProgressScorer(Encoder())
        before = scorer.score(coding_observation(state))
        state['current_code'] = 'def f(): return 1'
        after = scorer.score(coding_observation(state))
        self.assertGreater(after.value, before.value)
        self.assertNotEqual(after.regime, 'complete')
        state['current_code'] = 'def broken('
        self.assertLess(scorer.score(coding_observation(state)).value, after.value)

    def test_context_reserves_output_and_rejects_without_truncation(self):
        model = Model('http://localhost/v1', 'm', max_tokens=1024,
                      context_tokens=4096, tokenizer_name='local')
        with patch('token_budget.input_tokens', return_value=3000):
            self.assertEqual(check_context(model, []), 3000)
        with patch('token_budget.input_tokens', return_value=3010):
            with self.assertRaises(LimitReached):
                check_context(model, [])

    def test_cost_profile_load_and_context_dependence(self):
        profile = dict(load_joules=10, prefill_joules_per_token=1, decode_joules_per_token=2,
            load_seconds=1, prefill_seconds_per_token=.1, decode_seconds_per_token=.2,
            coordinator_model='coord', role='coder', role_description='Code')
        option = ModelOption(Model('http://localhost/v1', 'worker', max_tokens=10), 1, 1, profile)
        with patch('token_budget.input_tokens', return_value=5):
            cost = option.estimate({}, 'implement')
        self.assertEqual(cost.energy, 35)
        warm = copy.deepcopy(profile)
        warm['coordinator_model'] = 'worker'
        with patch('token_budget.input_tokens', return_value=5):
            self.assertEqual(ModelOption(option.model, 1, 1, warm).estimate({}, 'implement').energy, 25)

    def test_routing_builder_requires_real_costs(self):
        root = Path(__file__).parent
        base = json.loads((root/'examples/ollama_config.json').read_text())
        calibration = json.loads((root/'examples/calibration.template.json').read_text())
        with self.assertRaises((TypeError, ValueError)):
            build(base, calibration)
        calibration['routing'].update(energy_budget=1000, time_budget=300, average_energy=500)
        calibration['energy_measurement']['idle_watts'] = [10]
        for value in calibration['profiles'].values():
            value.update(energy=10, seconds=1)
        configured = build(base, calibration)
        self.assertEqual(len(configured['candidates']['coder']), 4)
        self.assertEqual(configured['coordinator']['context_tokens'], 4096)
        from create_baselines import variants
        baselines = variants(configured)
        self.assertEqual(baselines['random']['routing']['epsilon'], 1)
        self.assertEqual(baselines['fixed-small']['workers']['coder']['model'], 'progrouter-coder:0.5b')
        self.assertEqual(baselines['fixed-7b']['task_budget']['energy_joules'], 1000)

    def test_seeded_preparation_is_reproducible_and_preserves_subset_coverage(self):
        import tempfile
        from benchmarks import prepare
        row = {'prompt': 'def f():\n    """Return one."""\n',
               'test': 'def check(candidate): assert candidate() == 1', 'entry_point': 'f'}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'source.json'
            source.write_text(json.dumps([dict(row, task_id=str(i)) for i in range(10)]))
            a = prepare(source, 'humaneval', root/'a', root/'a-private.json', seed=3, limit=5)
            b = prepare(source, 'humaneval', root/'b', root/'b-private.json', seed=3, limit=5)
            self.assertEqual(a['tasks'], b['tasks'])
            self.assertEqual(len(a['tasks']), 5)


if __name__ == '__main__':
    unittest.main()
