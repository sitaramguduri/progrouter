"""Trusted evaluator entry point, copied into an isolated container with data.

No inference workflow imports this file or its held-out input. Candidate code and
reference programs execute only inside the evaluator container.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import re


def grade(data):
    kind = data['kind']
    if kind == 'python':
        namespace = {'__name__': '__candidate__'}
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                exec(compile(data['code'], 'candidate.py', 'exec'), namespace)
                exec(compile(data['tests'], 'heldout.py', 'exec'), namespace)
            return {'status': 'evaluated', 'passed': True, 'metric': 'heldout_tests'}
        except BaseException:
            return {'status': 'evaluated', 'passed': False, 'metric': 'heldout_tests'}
    if kind == 'evalplus':
        # Use EvalPlus's special oracles, tolerances and time-limit machinery.
        from evalplus.evaluate import get_groundtruth, check_correctness
        from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
        import hashlib
        from evalplus.data.mbpp import mbpp_deserialize_inputs
        problem = data['record']
        reference_hash = hashlib.sha256(json.dumps(problem, sort_keys=True).encode()).hexdigest()
        if data['dataset'] == 'mbpp':
            for key in ('base_input', 'plus_input'):
                problem[key] = mbpp_deserialize_inputs(problem['task_id'], problem[key])
        expected = get_groundtruth({problem['task_id']: problem}, reference_hash,
            MBPP_OUTPUT_NOT_NONE_TASKS if data['dataset'] == 'mbpp' else [])
        result = check_correctness(data['dataset'], 0, problem, data['code'],
                                  expected[problem['task_id']], fast_check=True)
        base, plus = result['base'][0], result['plus'][0]
        return {'status': 'evaluated', 'passed': base == 'pass' and plus == 'pass',
                'metric': 'evalplus_base_and_plus', 'base_status': base, 'plus_status': plus}
    if kind == 'math':
        from math_verify import parse, verify
        reference = parse('$'+data['answer']+'$')
        if not reference:
            return {'status': 'error', 'passed': None, 'metric': 'math_verify', 'reason': 'unparseable_reference'}
        predicted = parse(data['prediction'])
        return {'status': 'evaluated', 'passed': bool(predicted and verify(reference, predicted)), 'metric': 'math_verify'}
    if kind == 'asqa':
        from transformers import pipeline
        entail = pipeline('text-classification', model=data['nli_model'], tokenizer=data['nli_model'],
                          model_kwargs={'local_files_only': True}, trust_remote_code=False, device=-1)
        labels = {str(v).lower() for v in entail.model.config.id2label.values()}
        if data['entailment_label'].lower() not in labels:
            raise ValueError('Configure an entailment label from the NLI model id2label mapping')
        documents = {str(doc['id']): doc['text'] for doc in data['documents']}
        cited = supported = uncited = claims = 0
        for sentence in re.split(r'(?<=[.!?])\s+|\n+', data['prediction']):
            ids = re.findall(r'\[(\d+)\]', sentence)
            claim = re.sub(r'\[\d+\]', '', sentence).strip()
            if not claim:
                continue
            claims += 1
            if not ids:
                uncited += 1
            for identifier in ids:
                cited += 1
                if identifier not in documents:
                    continue
                judgment = entail({'text': documents[identifier], 'text_pair': claim}, truncation=True)
                supported += judgment['label'].lower() == data['entailment_label'].lower()
        return {'status': 'evaluated', 'passed': None, 'metric': 'nli_citation_precision',
                'citation_precision': supported/cited if cited else 0., 'citations': cited,
                'supported_citations': supported, 'claims': claims, 'uncited_claims': uncited,
                'judge_model': data['nli_model'], 'entailment_label': data['entailment_label']}
    raise ValueError('Unsupported evaluator kind')


if __name__ == '__main__':
    data = json.loads(Path('/work/input.json').read_text())
    try:
        result = grade(data)
    except Exception as error:
        result = {'status': 'error', 'passed': None, 'reason': type(error).__name__}
    print('__PROGROUTER_FINAL_RESULT__='+json.dumps(result))
