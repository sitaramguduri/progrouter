"""Report all-task, warm-up and common steady-state metrics from existing artifacts."""
import argparse
from collections import Counter
import json
from pathlib import Path
from benchmarks import summarize


def summarize_phase(reports):
    result = summarize(reports)
    elapsed = [r['workflow_metrics'].get('elapsed_seconds') for r in reports]
    result['mean_seconds'] = sum(elapsed)/len(elapsed) if elapsed and all(x is not None for x in elapsed) else None
    result['workflow_statuses'] = dict(Counter(r['workflow_status'] for r in reports))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-stream', type=Path, required=True,
                        help='ProgRouter stream.json supplies common phase membership')
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tasks = json.loads(args.reference_stream.read_text())['tasks']
    phases = {r['task_id']: r['phase'] for r in tasks}
    output = {}
    for path in args.reports:
        data = json.loads(path.read_text())
        reports = data['reports']
        if {r['task_id'] for r in reports} != set(phases):
            raise ValueError('Reports must cover the same tasks as the reference stream')
        output[str(path)] = {'all': summarize_phase(reports)}
        for phase in ('warm_up', 'steady_state'):
            output[str(path)][phase] = summarize_phase([r for r in reports if phases[r['task_id']] == phase])
    with args.output.open('x') as file:
        json.dump(output, file, indent=2)


if __name__ == '__main__':
    main()
