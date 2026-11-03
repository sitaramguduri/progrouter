import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from output_schemas import schema
from prompt_budget import prepare_messages
from workflow import CoordinatorWorkflow, FixedModelSelector, Limits, Model, Task, LimitReached, prompt_ledger
from test_workflow import ScriptedClient, dispatch, worker, finish


class RecoveryTests(unittest.TestCase):
    def test_tokenizer_requests_ids_instead_of_counting_mapping_keys(self):
        from token_budget import input_tokens
        model = Model('http://localhost/v1', 'm', tokenizer_name='test')
        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                if kwargs.get('return_dict') is False:
                    return list(range(137))
                return {'input_ids': list(range(137)), 'attention_mask': [1]*137}
        with patch('token_budget.tokenizer', return_value=FakeTokenizer()):
            self.assertEqual(input_tokens(model, []), 137)

    def test_coder_schema_requires_full_code(self):
        self.assertEqual(schema('worker:coder', {'coder': 'code'})['properties']['code']['type'], 'string')
        self.assertEqual(schema('worker:coder', {'coder': 'code'})['properties']['code']['minLength'], 1)

    def engine(self, responses, limits=None):
        model = Model('http://localhost/v1', 'm')
        return CoordinatorWorkflow(ScriptedClient(responses), model,
            FixedModelSelector({'coder': model}), {'coder': 'code'}, limits)

    def test_action_branches_require_their_fields(self):
        branches = {s['properties']['action']['const']: s for s in schema('coordinator', {'coder': 'code'})['oneOf']}
        self.assertIn('final_answer', branches['finish']['required'])
        self.assertIn('role', branches['dispatch']['required'])
        self.assertIn('subtask', branches['dispatch']['required'])
        self.assertEqual(branches['dispatch']['properties']['role']['enum'], ['coder'])
        self.assertIn('blocker', branches['abort']['required'])
        terminal = schema('coordinator', {'coder': 'code'}, ['finish', 'abort'])
        self.assertEqual([s['properties']['action']['const'] for s in terminal['oneOf']], ['finish', 'abort'])

    def test_repetition_is_not_a_protocol_error(self):
        responses = [dispatch('coder'), worker('analysis'), dispatch('coder'), worker('more analysis', code='def f(): return 1'), finish()]
        with tempfile.TemporaryDirectory() as folder:
            result = self.engine(responses).run(Task('t', 'Implement f'), Path(folder)/'run')
        self.assertEqual(result['status'], 'coordinator_finished')
        self.assertEqual(len(result['history']), 2)

    def test_stagnation_is_bounded_and_not_an_error_or_forced_success(self):
        responses = []
        for _ in range(3):
            responses += [dispatch('coder'), worker('no code')]
        responses += [dispatch('coder')]
        with tempfile.TemporaryDirectory() as folder:
            result = self.engine(responses, Limits(max_stagnant_steps=1)).run(Task('t', 'Implement f'), Path(folder)/'run')
        self.assertEqual(result['status'], 'stalled')
        self.assertIsNone(result['metrics']['final_correctness'])
        self.assertEqual(len(result['history']), 3)

    def test_code_updates_allow_repeated_repairs(self):
        responses = []
        for i in range(4):
            responses += [dispatch('coder', 'repair'), worker(code=f'def f(): return {i}')]
        responses += [finish()]
        with tempfile.TemporaryDirectory() as folder:
            result = self.engine(responses, Limits(max_stagnant_steps=1)).run(Task('t', 'Implement f'), Path(folder)/'run')
        self.assertEqual(result['status'], 'coordinator_finished')
        self.assertTrue(all(s['material_change'] for s in result['history']))

    def test_token_compaction_preserves_code_task_and_feedback(self):
        model = Model('http://localhost/v1', 'm', max_tokens=100, context_tokens=1000, tokenizer_name='test')
        ledger = {'task': {'problem': 'Implement f'}, 'current_code': 'def f(): return 1',
            'history': [{'message': 'analysis '*100, 'visible_feedback': None} for _ in range(10)]}
        ledger['history'][-1]['visible_feedback'] = {'status': 'completed',
            'output_tail': 'Expected: 4; Got: 0', 'result': {'passed': 0, 'total': 1, 'failures': [{'index': 0}]}}
        messages = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': json.dumps({'ledger': ledger})}]
        with patch('token_budget.input_tokens', side_effect=lambda m, x: sum(len(s['content']) for s in x)//4):
            prepared, metadata = prepare_messages(model, messages, 100)
        compact = json.loads(prepared[1]['content'])['ledger']
        self.assertGreater(metadata['history_entries_omitted'], 0)
        self.assertEqual(compact['task'], ledger['task'])
        self.assertEqual(compact['current_code'], ledger['current_code'])
        self.assertEqual(compact['latest_visible_feedback']['result']['passed'], 0)
        self.assertIn('Expected: 4; Got: 0', compact['latest_visible_feedback']['output_tail'])
        self.assertEqual(len(ledger['history']), 10)

    def test_token_limit_takes_precedence_over_character_heuristic(self):
        model = Model('http://localhost/v1', 'm', max_tokens=100, context_tokens=1000, tokenizer_name='test')
        messages = [{'role': 'user', 'content': 'a'*2000}]
        with patch('token_budget.input_tokens', return_value=500):
            self.assertEqual(prepare_messages(model, messages, 100)[0], messages)

    def test_oversized_mandatory_input_is_not_silently_truncated(self):
        model = Model('http://localhost/v1', 'm', max_tokens=100, context_tokens=500, tokenizer_name='test')
        messages = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': json.dumps({'ledger': {
            'task': {'problem': 'x'*5000}, 'current_code': 'full code', 'history': []}})}]
        with patch('token_budget.input_tokens', return_value=2000):
            with self.assertRaisesRegex(LimitReached, 'cannot fit'):
                prepare_messages(model, messages)

    def test_old_passes_are_not_advertised_for_new_code(self):
        ledger = {'current_code': 'new', 'history': [
            {'code': 'old', 'visible_feedback': {'status': 'completed', 'result': {'passed': 1}}},
            {'code': 'new', 'visible_feedback': None}]}
        self.assertEqual(prompt_ledger(ledger)['latest_visible_feedback']['status'], 'not_run_for_current_code')


if __name__ == '__main__':
    unittest.main()
