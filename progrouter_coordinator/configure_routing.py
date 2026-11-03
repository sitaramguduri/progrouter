"""Generate the four-model routing configuration from operator-calibrated costs.

No inference, downloads, or system changes. Refuses placeholder/null costs.
"""
import argparse
import json
import math
from pathlib import Path


def build(base, calibration):
    from routing import RoutingConfig
    from workflow import Model
    routing = calibration['routing']
    RoutingConfig(**routing)
    pools = {}
    options = []
    for size in ('0.5b', '1.5b', '3b', '7b'):
        name = 'progrouter-coder:'+size
        profile = calibration['profiles'][name]
        for key in ('energy', 'seconds'):
            value = profile[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name}: supply measured positive {key}')
        model = dict(base['coordinator'], model=name)
        Model(**model)
        options.append({'model': model, 'energy': profile['energy'], 'seconds': profile['seconds']})
    from workflow import DEFAULT_ROLES
    import copy
    for role in base['workers']:
        pools[role] = copy.deepcopy(options)
        for option in pools[role]:
            size = option['model']['model'].split(':')[1].replace('b', 'B')
            option['model'].update(context_tokens=4096,
                tokenizer_name=f'Qwen/Qwen2.5-Coder-{size}-Instruct')
            coefficients = calibration['profiles'][option['model']['model']].get('coefficients')
            if coefficients is not None:
                option['profile'] = dict(coefficients, role=role,
                    role_description=base.get('roles', DEFAULT_ROLES)[role],
                    coordinator_model=base['coordinator']['model'])
    result = dict(base)
    result.pop('workers')
    result.update(routing=routing, candidates=pools)
    if 'evaluation_protocol' in calibration:
        protocol = calibration['evaluation_protocol']
        if (not isinstance(protocol['min_samples'], int) or protocol['min_samples'] < 1 or
                not 0 <= protocol['max_epsilon'] <= 1):
            raise ValueError('Invalid stabilization thresholds')
        result['evaluation_protocol'] = protocol
    result['coordinator'] = dict(base['coordinator'], context_tokens=4096,
        tokenizer_name='Qwen/Qwen2.5-Coder-7B-Instruct')
    if routing.get('cost_basis') == 'measured_task':
        measurement = calibration['energy_measurement']
        if not measurement['devices'] or len(measurement['devices']) != len(measurement['idle_watts']):
            raise ValueError('Supply GPU devices and measured idle baselines')
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
               for v in measurement['idle_watts']):
            raise ValueError('Idle baselines must be measured finite nonnegative watts')
        result['energy_measurement'] = measurement
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('examples/ollama_config.json'))
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build(json.loads(args.base.read_text()), json.loads(args.calibration.read_text()))
        with args.output.open('x') as file:
            json.dump(result, file, indent=2)
            file.write('\n')
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.exit(2, str(error)+'\n')


if __name__ == '__main__':
    main()
