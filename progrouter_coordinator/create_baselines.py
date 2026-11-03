"""Write fixed-small, fixed-7B and uniform-random configs; no runs are started."""
import argparse
import copy
import json
from pathlib import Path


def variants(config):
    random_config = copy.deepcopy(config)
    random_config['routing'].update(epsilon=1, epsilon_min=1, epsilon_decay=1)
    random_config.pop('evaluation_protocol', None)
    result = {'random': random_config}
    for label, name in (('fixed-small', 'progrouter-coder:0.5b'), ('fixed-7b', 'progrouter-coder:7b')):
        fixed = copy.deepcopy(config)
        fixed['workers'] = {role: next(o['model'] for o in options if o['model']['model'] == name)
                            for role, options in fixed.pop('candidates').items()}
        routing = fixed.pop('routing')
        fixed.pop('evaluation_protocol', None)
        if routing['cost_basis'] != 'measured_task':
            raise ValueError('Use measured_task for cost-comparable baselines')
        fixed['task_budget'] = {'energy_joules': routing['energy_budget'], 'seconds': routing['time_budget']}
        result[label] = fixed
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    configs = variants(json.loads(args.config.read_text()))
    args.output.mkdir(parents=True, exist_ok=False)
    for name, config in configs.items():
        (args.output/(name+'.json')).write_text(json.dumps(config, indent=2)+'\n')
