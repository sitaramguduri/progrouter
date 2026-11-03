"""Offline benchmark preparation, stream execution and independent final grading.

Dataset exports must be provisioned locally. Private records are never read by run.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import re
from pathlib import Path
import subprocess
import sys
import random
from collections import Counter

from workflow import Task
from sandbox import DockerSandbox


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def records_from(path, benchmark, split):
    import gzip
    text = gzip.open(path, 'rt', encoding='utf-8').read() if path.suffix == '.gz' else path.read_text()
    if path.name.endswith(('.jsonl', '.jsonl.gz')):
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        records = json.loads(text)
    if benchmark == 'asqa' and isinstance(records, dict):
        if split not in records:
            raise ValueError('ASQA export must contain the selected split')
        records = [dict(record, task_id=str(identifier)) for identifier, record in records[split].items()]
    if not isinstance(records, list):
        raise ValueError('Provide a JSON list, JSONL records, or the original ASQA split dictionary')
    return records


def adapt(record, benchmark, index=0, visible_count=0, documents=None):
    if visible_count < 0:
        raise ValueError('visible_count must be nonnegative')
    task_id = str(record.get('task_id', record.get('unique_id', f'{benchmark}/{index}')))
    tests = []
    if benchmark in {'humaneval', 'humaneval_plus', 'mbpp_plus'}:
        problem = record['prompt']+'\nReturn the complete Python implementation, including its function signature.'
        from public_checks import prompt_checks
        tests = prompt_checks(record['prompt'])
        if benchmark == 'humaneval':
            entry = record['entry_point']
            if not entry.isidentifier():
                raise ValueError('Invalid entry_point')
            evaluator = {'kind': 'python', 'tests': record['test']+'\ncheck('+entry+')\n'}
        else:
            if not record.get('base_input') or not record.get('plus_input'):
                raise ValueError('EvalPlus records require nonempty base_input and plus_input')
            evaluator = {'kind': 'evalplus', 'dataset': 'mbpp' if benchmark == 'mbpp_plus' else 'humaneval', 'record': record}
        if visible_count:
            raise ValueError('HumanEval/EvalPlus adapters expose prompt examples only, not held-out inputs')
    elif benchmark == 'mbpp':
        problem = record.get('text', record.get('prompt'))
        if not isinstance(problem, str):
            raise ValueError('MBPP requires text or prompt')
        raw_tests = record['test_list']
        if visible_count >= len(raw_tests):
            raise ValueError('Keep at least one MBPP test held out')
        setup = record.get('test_setup_code', '')+'\n'+'\n'.join(record.get('test_imports', []))+'\n'
        tests = [setup+test for test in raw_tests[:visible_count]]
        evaluator = {'kind': 'python', 'tests': setup+'\n'.join(raw_tests[visible_count:]+record.get('challenge_test_list', []))}
    elif benchmark == 'math500':
        problem = record['problem']+'\nGive your final answer in \\boxed{...} in the final response.'
        evaluator = {'kind': 'math', 'answer': record['answer']}
    elif benchmark == 'asqa':
        if not documents:
            raise ValueError('ASQA requires a separately supplied retrieval corpus per task')
        ids = set()
        for doc in documents:
            identifier = str(doc['id'])
            if not identifier.isdigit() or identifier in ids or not isinstance(doc['text'], str):
                raise ValueError('Documents require unique numeric ids and string text')
            ids.add(identifier)
        problem = record['ambiguous_question']+'\nAnswer using the sources below. Use one claim per line and cite source IDs as [1].\n'
        problem += '\n'.join('['+str(doc['id'])+'] '+doc.get('title', '')+'\n'+doc['text'] for doc in documents)
        evaluator = {'kind': 'asqa', 'documents': documents}
    else:
        raise ValueError('Unsupported benchmark')
    task = Task(task_id, problem, tuple(tests))
    public = asdict(task)
    public['visible_tests'] = list(public['visible_tests'])
    private = {'task_id': task_id, 'task_hash': digest(public), 'benchmark': benchmark, 'evaluator': evaluator}
    return public, private


def prepare(source, benchmark, public_dir, private_path, *, split='dev', visible_count=0, documents_path=None,
            seed=None, limit=None):
    rows = records_from(source, benchmark, split)
    if seed is not None:
        random.Random(seed).shuffle(rows)
    if limit is not None:
        if limit <= 0:
            raise ValueError('limit must be positive')
        rows = rows[:limit]
    if not rows:
        raise ValueError('Empty benchmark')
    documents = json.loads(documents_path.read_text()) if documents_path else {}
    prepared, ids = [], set()
    for i, record in enumerate(rows):
        identifier = str(record.get('task_id', record.get('unique_id', f'{benchmark}/{i}')))
        public, private = adapt(record, benchmark, i, visible_count, documents.get(identifier))
        if public['task_id'] in ids:
            raise ValueError('Duplicate task_id')
        ids.add(public['task_id'])
        prepared.append((public, private))
    public_dir = public_dir.resolve()
    private_path = private_path.resolve()
    if private_path.is_relative_to(public_dir):
        raise ValueError('Private references must live outside the public task directory')
    if private_path.exists() or (public_dir.exists() and any(public_dir.iterdir())):
        raise ValueError('Use fresh public and private output paths')
    public_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'version': 1, 'benchmark': benchmark, 'seed': seed, 'limit': limit,
                'public_check_protocol': 'prompt-doctests-v2',
                'dataset_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'tasks': []}
    private_records = []
    for i, (public, private) in enumerate(prepared):
        filename = f'{i:05d}.json'
        (public_dir/filename).write_text(json.dumps(public, indent=2)+'\n')
        manifest['tasks'].append({'task_id': public['task_id'], 'file': filename, 'hash': digest(public)})
        private_records.append(private)
    (public_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    private_path.parent.mkdir(parents=True, exist_ok=True)
    with private_path.open('x', encoding='utf-8') as file:
        private_path.chmod(0o600)
        json.dump({'version': 1, 'dataset_sha256': manifest['dataset_sha256'], 'records': private_records}, file)
    return manifest


def load_manifest(public_dir):
    manifest = json.loads((public_dir/'manifest.json').read_text())
    ids = set()
    for item in manifest['tasks']:
        if not re.fullmatch(r'\d{5}\.json', item['file']) or item['task_id'] in ids:
            raise ValueError('Invalid manifest')
        public = json.loads((public_dir/item['file']).read_text())
        if digest(public) != item['hash'] or public['task_id'] != item['task_id']:
            raise ValueError('Public task differs from prepared manifest')
        Task.load(public_dir/item['file'])
        ids.add(item['task_id'])
    return manifest


def run_stream(public_dir, config, output_dir, checkpoint=None, enable_tests=False):
    manifest = load_manifest(public_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Use a fresh stream output directory')
    output_dir.mkdir(parents=True, exist_ok=True)
    config_data = json.loads(config.read_text())
    routed = config_data.get('routing') is not None
    protocol = config_data.get('evaluation_protocol', {})
    stabilized = False
    results = []
    for i, item in enumerate(manifest['tasks']):
        run_dir = output_dir/f'{i:05d}'
        args = [sys.executable, str(Path(__file__).with_name('run.py')), '--task', str((public_dir/item['file']).resolve()),
                '--config', str(config.resolve()), '--output', str(run_dir.resolve())]
        if checkpoint:
            args.extend(['--router-checkpoint', str(checkpoint.resolve())])
        if enable_tests:
            args.append('--enable-tests')
        process = subprocess.run(args, check=False)
        if not (run_dir/'ledger.json').exists():
            raise RuntimeError(f'Run did not produce a ledger: task {item["task_id"]}')
        if routed and not (run_dir/'router.joblib').exists():
            raise RuntimeError('Routing checkpoint missing; stop stream to preserve queue/learning continuity')
        checkpoint = run_dir/'router.joblib' if (run_dir/'router.joblib').exists() else None
        phase = 'steady_state' if stabilized else 'warm_up'
        ledger = json.loads((run_dir/'ledger.json').read_text())
        metrics = ledger.get('metrics', {}).get('routing', {})
        if protocol and metrics.get('predictor_ready') and (
                metrics.get('buffer_samples', 0) >= protocol['min_samples'] and
                metrics.get('epsilon', 1) <= protocol['max_epsilon']):
            stabilized = True
        results.append({'task_id': item['task_id'], 'directory': f'{i:05d}', 'exit_code': process.returncode,
                        'phase': phase if protocol else 'unspecified',
                        'stabilized_after_task': stabilized})
        (output_dir/'stream.json').write_text(json.dumps({'tasks': results}, indent=2)+'\n')
    return results


def evaluate_run(run_dir, reference, sandbox, *, nli_model=None, entailment_label='entailment'):
    ledger = json.loads((run_dir/'ledger.json').read_text())
    public = ledger['task']
    if ledger.get('status') == 'running':
        raise ValueError('Cannot evaluate a running workflow')
    if public['task_id'] != reference['task_id'] or digest(public) != reference['task_hash']:
        raise ValueError('Evaluation reference does not match this task')
    data = dict(reference['evaluator'])
    metric = data['kind']
    if metric in {'python', 'evalplus'}:
        code = ledger.get('current_code')
        if code is None:
            result = {'status': 'evaluated', 'passed': False, 'metric': metric, 'reason': 'no_final_code'}
        else:
            data['code'] = code  # Never execute a mutable/unrelated final.py artifact.
            result = None
    else:
        data['prediction'] = ledger.get('final_answer') or ''
        result = None
        if metric == 'asqa':
            if not nli_model:
                raise ValueError('ASQA evaluation requires an NLI model path inside the evaluator image')
            data.update(nli_model=nli_model, entailment_label=entailment_label)
    if result is None:
        outcome = sandbox.run({'grade.py': Path(__file__).with_name('grading.py').read_text(), 'input.json': json.dumps(data)})
        if outcome['status'] != 'completed':
            result = {'status': outcome['status'], 'passed': None, 'metric': metric}
        else:
            lines = [line for line in outcome['stdout'].splitlines() if line.startswith('__PROGROUTER_FINAL_RESULT__=')]
            try:
                result = json.loads(lines[-1].split('=', 1)[1]) if len(lines) == 1 else {}
            except ValueError:
                result = {}
            valid = isinstance(result, dict) and result.get('status') in {'evaluated', 'error'}
            if valid:
                valid = type(result.get('passed')) in (bool, type(None))
                if result['status'] == 'error':
                    valid = valid and result.get('passed') is None
                elif metric != 'asqa':
                    valid = valid and type(result.get('passed')) is bool
                else:
                    precision = result.get('citation_precision')
                    valid = valid and result.get('passed') is None and isinstance(precision, (int, float)) and math.isfinite(precision) and 0 <= precision <= 1
            if not valid:
                result = {'status': 'error', 'passed': None, 'metric': metric, 'reason': 'invalid_evaluator_result'}
    return {'task_id': reference['task_id'], 'benchmark': reference['benchmark'],
            'model_calls': dict(Counter(step['model'] for step in ledger.get('history', []))),
            'task_hash': reference['task_hash'], 'reference_hash': digest(reference), 'artifact_hash': digest({'code': ledger.get('current_code'), 'answer': ledger.get('final_answer')}),
            'evaluation': result, 'final_correctness': result.get('passed') if result.get('status') == 'evaluated' else None, 'workflow_status': ledger['status'], 'workflow_metrics': ledger['metrics'],
            'evaluator_image': sandbox.image}


def summarize(reports):
    distribution = Counter()
    for report in reports:
        distribution.update(report.get('model_calls', {}))
    evaluated = [r['evaluation'] for r in reports if r['evaluation']['status'] == 'evaluated']
    boolean = [r for r in evaluated if type(r.get('passed')) is bool]
    citations = [r for r in evaluated if 'citation_precision' in r]
    energies = [r['workflow_metrics'].get('energy_joules') for r in reports]
    return {'tasks': len(reports), 'worker_model_distribution': dict(distribution),
            'evaluated': len(evaluated), 'unavailable_or_error': len(reports)-len(evaluated),
            'pass_rate_evaluated': sum(r['passed'] for r in boolean)/len(boolean) if boolean else None,
            'pass_rate_all_tasks': sum(r['passed'] for r in boolean)/len(reports) if boolean and len(boolean) == len(reports) else None,
            'citation_precision_macro': sum(r['citation_precision'] for r in citations)/len(citations) if citations else None,
            'measured_energy_tasks': sum(e is not None for e in energies),
            'mean_measured_energy_joules': sum(energies)/len(energies) if energies and all(e is not None for e in energies) else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--dataset', type=Path, required=True)
    prep.add_argument('--benchmark', choices=['humaneval', 'humaneval_plus', 'mbpp', 'mbpp_plus', 'math500', 'asqa'], required=True)
    prep.add_argument('--public', type=Path, required=True)
    prep.add_argument('--private', type=Path, required=True)
    prep.add_argument('--split', default='dev')
    prep.add_argument('--visible-count', type=int, default=0)
    prep.add_argument('--documents', type=Path)
    prep.add_argument('--seed', type=int, help='Shuffle once before selecting tasks')
    prep.add_argument('--limit', type=int, help='Optional smoke-test subset size')
    run = sub.add_parser('run')
    run.add_argument('--public', type=Path, required=True)
    run.add_argument('--config', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--router-checkpoint', type=Path)
    run.add_argument('--enable-tests', action='store_true')
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('--private', type=Path, required=True)
    evaluate.add_argument('--runs', type=Path, required=True, help='Stream directory containing stream.json')
    evaluate.add_argument('--output', type=Path, required=True)
    evaluate.add_argument('--image', default='python:3.12-slim')
    evaluate.add_argument('--timeout', type=float, default=120)
    evaluate.add_argument('--nli-model')
    evaluate.add_argument('--entailment-label', default='entailment')
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            result = prepare(args.dataset, args.benchmark, args.public, args.private,
                split=args.split, visible_count=args.visible_count, documents_path=args.documents,
                seed=args.seed, limit=args.limit)
            print(json.dumps({'tasks': len(result['tasks']), 'dataset_sha256': result['dataset_sha256']}))
        elif args.command == 'run':
            run_stream(args.public, args.config, args.output, args.router_checkpoint, args.enable_tests)
        else:
            private_file = json.loads(args.private.read_text())
            references = private_file['records']
            private = {r['task_id']: r for r in references}
            if len(private) != len(references):
                raise ValueError('Duplicate private reference ids')
            runs = json.loads((args.runs/'stream.json').read_text())['tasks']
            if len({r['task_id'] for r in runs}) != len(runs):
                raise ValueError('Duplicate stream task ids')
            if set(private) != {r['task_id'] for r in runs}:
                raise ValueError('Evaluation requires complete, matching task coverage')
            reports = []
            for item in runs:
                if not re.fullmatch(r'\d{5}', item['directory']):
                    raise ValueError('Invalid run directory')
                reports.append(evaluate_run(args.runs/item['directory'], private[item['task_id']],
                    DockerSandbox(args.image, args.timeout), nli_model=args.nli_model, entailment_label=args.entailment_label))
            with args.output.open('x', encoding='utf-8') as file:
                json.dump({'dataset_sha256': private_file['dataset_sha256'], 'summary': summarize(reports), 'reports': reports}, file, indent=2)
            print(json.dumps(summarize(reports), indent=2))
        return 0
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
