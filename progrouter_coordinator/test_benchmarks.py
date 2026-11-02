import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks import adapt, digest, evaluate_run, load_manifest, prepare, run_stream, summarize
from sandbox import DockerSandbox


HUMAN = {'task_id': 'HumanEval/0', 'prompt': 'def f():\n    """Return an integer."""\n',
         'test': 'def check(candidate):\n    assert candidate() == 1234567',
         'entry_point': 'f', 'canonical_solution': '    return 1234567'}


class FakeSandbox:
    image = 'test@sha256:fixture'
    def __init__(self, result=None, status='completed'):
        self.result = result or {'status': 'evaluated', 'passed': False, 'metric': 'heldout_tests'}
        self.status = status
        self.files = None
    def run(self, files):
        self.files = files
        return {'status': self.status, 'stdout': '__PROGROUTER_FINAL_RESULT__='+json.dumps(self.result)}


class BenchmarkTests(unittest.TestCase):
    def test_humaneval_reference_never_enters_public_task(self):
        public, private = adapt(HUMAN, 'humaneval')
        self.assertNotIn('1234567', json.dumps(public))
        self.assertIn('1234567', private['evaluator']['tests'])
        self.assertEqual(private['task_hash'], digest(public))
        self.assertEqual(set(public), {'task_id', 'problem', 'visible_tests'})

    def test_mbpp_visible_split_keeps_independent_tests(self):
        public, private = adapt({'task_id': 11, 'text': 'Implement f()', 'code': 'SECRET',
            'test_list': ['assert f(1)==1', 'assert f(2)==2'], 'test_setup_code': 'import math',
            'challenge_test_list': ['assert f(3)==3']}, 'mbpp', visible_count=1)
        self.assertNotIn('SECRET', json.dumps(public))
        self.assertEqual(len(public['visible_tests']), 1)
        self.assertNotIn('f(1)', private['evaluator']['tests'])
        self.assertIn('f(2)', private['evaluator']['tests'])
        self.assertIn('f(3)', private['evaluator']['tests'])
        with self.assertRaises(ValueError):
            adapt({'text': 'x', 'test_list': ['assert 1']}, 'mbpp', visible_count=1)

    def test_plus_and_math_schema(self):
        row = dict(HUMAN, base_input=[[]], plus_input=[[]], atol=0)
        public, private = adapt(row, 'humaneval_plus')
        self.assertNotIn('canonical_solution', public)
        self.assertEqual(private['evaluator']['kind'], 'evalplus')
        self.assertIn('plus_input', private['evaluator']['record'])
        public, private = adapt({'unique_id': 'test/algebra/1.json', 'problem': 'Find x', 'answer': 'SECRET', 'solution': 'PRIVATE'}, 'math500')
        self.assertNotIn('SECRET', json.dumps(public))
        self.assertNotIn('PRIVATE', json.dumps(public))
        self.assertEqual(private['evaluator']['answer'], 'SECRET')

    def test_asqa_only_question_and_independent_corpus(self):
        row = {'task_id': 'q', 'ambiguous_question': 'Where?', 'qa_pairs': [{'short_answers':['SECRET']}], 'annotations': [{'long_answer':'PRIVATE'}]}
        docs = [{'id':1, 'title':'Source', 'text':'Public retrieved evidence'}]
        public, private = adapt(row, 'asqa', documents=docs)
        self.assertNotIn('SECRET', json.dumps(public))
        self.assertNotIn('PRIVATE', json.dumps(public))
        self.assertIn('Public retrieved evidence', public['problem'])
        with self.assertRaises(ValueError):
            adapt(row, 'asqa')

    def test_preparation_layout_tamper_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'dataset.jsonl'
            source.write_text(json.dumps(HUMAN)+'\n')
            public, private = root/'public', root/'private.json'
            prepare(source, 'humaneval', public, private)
            manifest = load_manifest(public)
            self.assertEqual(len(manifest['tasks']), 1)
            self.assertEqual(private.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(ValueError):
                prepare(source, 'humaneval', public, private)
            with self.assertRaises(ValueError):
                prepare(source, 'humaneval', root/'public2', root/'public2/private.json')
            (public/'00000.json').write_text('{}')
            with self.assertRaises((ValueError, KeyError)):
                load_manifest(public)

    def evaluate(self, sandbox):
        public, reference = adapt(HUMAN, 'humaneval')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ledger = {'task': public, 'status': 'coordinator_finished', 'current_code': 'def f(): return 0',
                      'final_answer': 'done', 'metrics': {'energy_joules': None, 'final_correctness':None}}
            (root/'ledger.json').write_text(json.dumps(ledger))
            (root/'final.py').write_text('def f(): return 1234567')
            result = evaluate_run(root, reference, sandbox)
            bad = dict(reference, task_hash='wrong')
            with self.assertRaises(ValueError):
                evaluate_run(root, bad, sandbox)
        return result

    def test_final_grading_is_independent_and_uses_frozen_ledger_code(self):
        sandbox = FakeSandbox()
        result = self.evaluate(sandbox)
        self.assertFalse(result['final_correctness'])
        data = json.loads(sandbox.files['input.json'])
        self.assertEqual(data['code'], 'def f(): return 0')
        self.assertIn('1234567', data['tests'])
        self.assertIsNone(result['workflow_metrics']['final_correctness'])

    def test_unavailable_and_malformed_evaluator_do_not_pass(self):
        for sandbox in (FakeSandbox(status='unavailable'), FakeSandbox({'status':'evaluated','passed':1})):
            result = self.evaluate(sandbox)
            self.assertIsNone(result['final_correctness'])
            self.assertIsNone(summarize([result])['pass_rate_all_tasks'])

    def test_stream_runner_never_reads_private_records(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'source.json').write_text(json.dumps([HUMAN, dict(HUMAN, task_id='HumanEval/1')]))
            prepare(root/'source.json', 'humaneval', root/'public', root/'private.json')
            (root/'config.json').write_text('{}')
            calls = []
            def invoke(args, check):
                calls.append(args)
                output = Path(args[args.index('--output')+1])
                output.mkdir()
                (output/'ledger.json').write_text('{}')
                return type('Result', (), {'returncode': 0})()
            with patch('benchmarks.subprocess.run', side_effect=invoke):
                result = run_stream(root/'public', root/'config.json', root/'runs')
            self.assertEqual(len(result), 2)
            self.assertTrue(all('private.json' not in ' '.join(call) for call in calls))

    def test_evaluation_cli_writes_separate_report(self):
        import benchmarks
        public, reference = adapt(HUMAN, 'humaneval')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'runs/00000').mkdir(parents=True)
            ledger = {'task':public,'status':'coordinator_finished','current_code':'def f(): return 0',
                      'final_answer':'done','metrics':{'energy_joules':None,'final_correctness':None}}
            frozen = json.dumps(ledger)
            (root/'runs/00000/ledger.json').write_text(frozen)
            (root/'runs/stream.json').write_text(json.dumps({'tasks':[{'task_id':public['task_id'],'directory':'00000'}]}))
            (root/'private.json').write_text(json.dumps({'dataset_sha256':'fixture','records':[reference]}))
            argv = ['benchmarks.py','evaluate','--private',str(root/'private.json'),'--runs',str(root/'runs'),'--output',str(root/'report.json')]
            with patch('sys.argv',argv), patch('benchmarks.DockerSandbox',return_value=FakeSandbox()), patch('builtins.print'):
                self.assertEqual(benchmarks.main(),0)
            report = json.loads((root/'report.json').read_text())
            self.assertEqual(report['summary']['pass_rate_all_tasks'],0)
            self.assertEqual((root/'runs/00000/ledger.json').read_text(),frozen)

    def test_sandbox_missing_docker_never_falls_back_to_host(self):
        with patch('sandbox.subprocess.Popen', side_effect=FileNotFoundError), patch('sandbox.subprocess.run'):
            result = DockerSandbox().run({'grade.py': 'raise RuntimeError("must not execute")'})
        self.assertEqual(result['status'], 'unavailable')

    def test_sandbox_contract_and_limits(self):
        commands = []
        class Process:
            returncode = 0
            def poll(self): return 0
        def spawn(cmd, stdout, stderr):
            commands.append(cmd)
            stdout.write(b'__PROGROUTER_FINAL_RESULT__={"passed": false}\n')
            return Process()
        with patch('sandbox.subprocess.Popen', side_effect=spawn), patch('sandbox.subprocess.run'):
            result = DockerSandbox().run({'grade.py':'pass'})
        self.assertEqual(result['status'], 'completed')
        cmd = commands[0]
        for flag in ('--network', '--read-only', '--pids-limit', '--memory', '--user', '--pull=never'):
            self.assertIn(flag, cmd)
        with self.assertRaises(ValueError):
            DockerSandbox().run({'../bad.py':'pass'})


if __name__ == '__main__':
    unittest.main()
