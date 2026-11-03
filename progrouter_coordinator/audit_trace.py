"""Independently recompute routing equations and observed gains from saved traces."""
import argparse
import json
import math
import random
from pathlib import Path


def close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f'{label}: actual={actual}, expected={expected}')


def audit(directory):
    counts = {'routing_decisions': 0, 'gains': 0, 'queue_updates': 0}
    events = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    for event in events:
        data = event['payload']
        if event['event'] == 'routing_decision':
            cfg = data['config']
            if 'collected_samples' in data:
                close(data['epsilon'], max(cfg['epsilon_min'], cfg['epsilon']*cfg['epsilon_decay']**data['collected_samples']), 'epsilon')
            eligible = []
            for row in data['candidates']:
                e, t = row['estimated_energy'], row['estimated_seconds']
                feasible = data['energy_used']+e <= cfg['energy_budget'] and data['seconds_used']+t <= cfg['time_budget']
                if feasible != row['feasible']:
                    raise ValueError('Incorrect feasibility')
                if feasible:
                    eligible.append(row)
                if row['score'] is not None:
                    # Infer no state from hidden references: use the logged progress term
                    # with its logged prediction and reconstruct g below on transitions.
                    expected = (row['progress_term'] - data['queue']*(e-cfg['average_energy'])
                        - math.exp((data['seconds_used']+t)/cfg['time_budget'])*t
                        - math.exp((data['energy_used']+e)/cfg['energy_budget'])*e)
                    close(row['score'], expected, 'Eq12')
            if data['selected'] is not None:
                selected = next(r for r in eligible if r['model'] == data['selected'])
                if not data['exploration']:
                    close(selected['score'], max(r['score'] for r in eligible), 'argmax')
            counts['routing_decisions'] += 1
        elif event['event'] == 'worker_transition' and 'progress_gain' in data['entry']:
            before, after = data['state_before']['progress']['value'], data['state_after']['progress']['value']
            from progress import ProgressScorer, coding_observation
            for state in (data['state_before'], data['state_after']):
                score = state['progress']
                weight = ProgressScorer(None).weights[score['regime']]
                expected = max(0., min(1., weight.base + weight.completion*score['completion']
                    + weight.trend*score['trend'] + weight.quality*score['quality']))
                close(score['value'], expected, 'Eq2 (default scorer weights)')
                observation = coding_observation(state)
                completion = sum(observation.requirements.values())/len(observation.requirements) if observation.requirements else 0.
                close(score['completion'], completion, 'observable_completion')
            close(data['entry']['progress_gain'], after-before, 'realized_gain')
            routing = data['entry'].get('routing')
            if routing:
                for row in routing['candidates']:
                    if row['score'] is not None:
                        close(row['progress_term'], routing['config']['tradeoff']*(1-before)*row['prediction']['gain'], 'Eq9')
            counts['gains'] += 1
        elif event['event'] == 'virtual_queue_update' and data is not None:
            if data['queue_updated']:
                close(data['queue_after'], max(0, data['queue_before']+data['task_cost']-data['average_energy']), 'Eq7')
            else:
                close(data['queue_after'], data['queue_before'], 'failed_measurement_queue')
            counts['queue_updates'] += 1
    return counts


def audit_stream(directories, checkpoint=None):
    results = {str(d): audit(d) for d in directories}
    rng = None
    observed_targets = []
    decisions = 0
    for directory in directories:
        for line in (directory/'events.jsonl').read_text().splitlines():
            event = json.loads(line)
            data = event['payload']
            if event['event'] == 'routing_decision' and data['selected'] is not None:
                if rng is None:
                    rng = random.Random(data['config']['seed'])
                eligible = [r for r in data['candidates'] if r['feasible']]
                cold = any(not r['prediction']['ready'] for r in eligible)
                explore = cold or rng.random() < data['epsilon']
                if explore != data['exploration']:
                    raise ValueError('Exploration flag disagrees with seeded epsilon procedure')
                selected = rng.choice(eligible) if explore else max(eligible, key=lambda r: r['score'])
                if selected['model'] != data['selected']:
                    raise ValueError('Selection disagrees with seeded exploration/argmax')
                decisions += 1
            elif event['event'] == 'worker_transition':
                entry = data['entry']
                if entry.get('routing', {}).get('exploration'):
                    observed_targets.append(entry['progress_gain'])
    result = {'traces': results, 'seeded_selections_checked': decisions}
    if checkpoint is not None:
        # Explicit flag: joblib files must be trusted, since deserialization can run code.
        import joblib
        state = joblib.load(checkpoint)
        training = state['predictor']
        targets = [s[2] for s in training['samples']]
        if len(targets) > len(observed_targets):
            raise ValueError('Supply the complete fresh stream for checkpoint auditing')
        expected = observed_targets[-len(targets):] if targets else []
        if targets != expected or training['seen'] != len(observed_targets):
            raise ValueError('Training targets/count disagree with executed exploration transitions')
        ledger = json.loads((directories[-1]/'ledger.json').read_text())
        close(state['queue'], ledger['metrics']['routing']['queue'], 'checkpoint_queue')
        if rng is not None and state['random_state'] != rng.getstate():
            raise ValueError('Checkpoint RNG differs from replayed routing RNG')
        result['checkpoint_targets_checked'] = len(targets)
        result['predictor_fitted'] = training['meta'] is not None
        result['checkpoint_rng_verified'] = True
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--trusted-checkpoint', type=Path,
                        help='Optional trusted router.joblib; directories must contain the complete fresh stream')
    args = parser.parse_args()
    print(json.dumps(audit_stream(args.directories, args.trusted_checkpoint), indent=2))
