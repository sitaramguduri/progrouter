"""Evaluator unit tests use only authored fixtures and mocked external judges."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from grading import grade


class GradingTests(unittest.TestCase):
    def test_python_heldout_success_and_failure_with_authored_fixtures(self):
        data = {'kind':'python', 'code':'def f(): return 1', 'tests':'assert f() == 1'}
        self.assertTrue(grade(data)['passed'])
        data['tests'] = 'assert f() == 2'
        self.assertFalse(grade(data)['passed'])
        data['code'] = 'invalid syntax ???'
        self.assertFalse(grade(data)['passed'])

    def test_math_reference_and_prediction_order(self):
        seen = []
        def verify(gold, answer):
            seen.append((gold, answer))
            return gold == ['$2$'] and answer == ['\\boxed{2}']
        module = SimpleNamespace(parse=lambda text: [text], verify=verify)
        with patch.dict('sys.modules', math_verify=module):
            result = grade({'kind':'math', 'answer':'2', 'prediction':'\\boxed{2}'})
        self.assertTrue(result['passed'])
        self.assertEqual(seen[0], (['$2$'], ['\\boxed{2}']))

    def test_asqa_unknown_citations_are_in_denominator(self):
        class Judge:
            model = SimpleNamespace(config=SimpleNamespace(id2label={0:'contradiction',1:'entailment'}))
            def __call__(self, pair, **kwargs):
                return {'label':'entailment'}
        with patch.dict('sys.modules', transformers=SimpleNamespace(pipeline=lambda *args, **kwargs: Judge())):
            result = grade({'kind':'asqa', 'nli_model':'/models/nli', 'entailment_label':'entailment',
                'documents':[{'id':1,'text':'A source'}], 'prediction':'A fact [1] [999].\nAn uncited fact.'})
        self.assertEqual(result['citation_precision'], .5)
        self.assertEqual(result['uncited_claims'], 1)
        self.assertIsNone(result['passed'])

    def test_evalplus_uses_official_oracle_after_input_deserialization(self):
        record = {'task_id':'Mbpp/124','base_input':[[1]],'plus_input':[[2]],'canonical_solution':'reference'}
        seen = []
        def check(dataset, cid, problem, code, expected, **kwargs):
            seen.append((dataset, problem['base_input'], code))
            return {'base':('pass',[]),'plus':('fail',[])}
        modules = {
            'evalplus.evaluate': SimpleNamespace(get_groundtruth=lambda problems, key, tasks: {'Mbpp/124':{}}, check_correctness=check),
            'evalplus.eval._special_oracle': SimpleNamespace(MBPP_OUTPUT_NOT_NONE_TASKS=[]),
            'evalplus.data.mbpp': SimpleNamespace(mbpp_deserialize_inputs=lambda tid, inputs: [[complex(1, 2)]])}
        with patch.dict('sys.modules', modules):
            result = grade({'kind':'evalplus','dataset':'mbpp','record':record,'code':'candidate'})
        self.assertFalse(result['passed'])
        self.assertEqual(seen[0], ('mbpp',[[complex(1,2)]],'candidate'))


if __name__ == '__main__':
    unittest.main()
