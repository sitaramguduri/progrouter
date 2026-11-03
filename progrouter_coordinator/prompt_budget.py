"""Explicit, auditable prompt compaction; preserves task, current code and latest checks."""
import copy
import json


def prepare_messages(model, messages, max_chars=120000):
    from workflow import LimitReached
    from token_budget import input_tokens
    result = copy.deepcopy(messages)

    def fits():
        if model.context_tokens is None and sum(len(m['content']) for m in result) > max_chars:
            return False
        return (model.context_tokens is None or
                input_tokens(model, result)+model.max_tokens+64 <= model.context_tokens)

    if fits():
        return result, None
    try:
        payload = json.loads(result[1]['content'])
        ledger = payload['ledger']
    except (IndexError, ValueError, KeyError, TypeError):
        raise LimitReached('prompt_budget: oversized non-ledger input') from None
    history = ledger.get('history', [])
    original_count = len(history)
    latest = ledger.get('latest_visible_feedback')
    if latest is None:
        latest = next((s['visible_feedback'] for s in reversed(history) if s.get('visible_feedback')), None)
    if latest:
        ledger['latest_visible_feedback'] = copy.deepcopy(latest)
        tail = ledger['latest_visible_feedback'].get('output_tail')
        if isinstance(tail, str) and len(tail) > 1200:
            ledger['latest_visible_feedback']['output_tail'] = tail[:1200]+' [shortened; full output in events]'
    metadata = {'history_entries_omitted': 0, 'fields_shortened': [],
                'preserved': ['task', 'current_code', 'latest_visible_feedback']}
    ledger['prompt_compaction'] = metadata
    # Completed-subtask strings duplicate history; keep the full list in artifacts.
    if 'completed_subtasks' in payload:
        payload['completed_subtask_count'] = len(payload.pop('completed_subtasks'))
        metadata['fields_shortened'].append('completed_subtasks_duplicate')
    for entry in history:
        feedback = entry.get('visible_feedback')
        if isinstance(feedback, dict):
            if feedback.pop('output_tail', None) is not None:
                metadata['fields_shortened'].append('historical_output_tail_duplicate')
    if latest and isinstance(latest.get('output_tail'), str) and len(latest['output_tail']) > 1200:
        metadata['fields_shortened'].append('latest_output_tail')

    def render():
        result[1]['content'] = json.dumps(payload, ensure_ascii=False)

    render()
    while not fits() and len(history) > 1:
        history.pop(0)
        metadata['history_entries_omitted'] = original_count-len(history)
        render()
    # Preserve the latest attempt and authoritative test results; shorten prose only.
    if not fits():
        for key in ('state_summary',):
            if isinstance(ledger.get(key), str) and len(ledger[key]) > 300:
                ledger[key] = ledger[key][:300]+' [shortened; full text in events]'
                metadata['fields_shortened'].append(key)
        for entry in history:
            if len(entry.get('message', '')) > 500:
                entry['message'] = entry['message'][:500]+' [shortened; full text in events]'
                metadata['fields_shortened'].append('latest_worker_message')
        if len(ledger.get('plan', [])) > 3:
            ledger['plan'] = ledger['plan'][-3:]
            metadata['fields_shortened'].append('plan')
        render()
    if not fits():
        raise LimitReached('prompt_budget: task/current_code/latest feedback cannot fit with output reserve; '
                           'increase context or reduce task size, not worker-step count')
    return result, metadata
