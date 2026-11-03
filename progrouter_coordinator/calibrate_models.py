"""Explicit GPU calibration workload. Loads/unloads models and generates text.

Run only when ready: python calibrate_models.py --output calibration.json
Uses local Ollama and NVML; no generated code is executed. Requires exclusive GPU
use for meaningful attribution. Output budgets are deliberately null for review.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def post(base, route, data):
    request = urllib.request.Request(base+route, data=json.dumps(data).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:11434')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--idle-seconds', type=float, default=5)
    parser.add_argument('--settle-seconds', type=float, default=10,
                        help='Allow clocks/power to settle after unloading before idle sampling')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    if args.idle_seconds < 1:
        parser.error('Measure idle for at least one second')
    if args.settle_seconds < 0:
        parser.error('settle-seconds must be nonnegative')
    from energy import NVMLPower, EnergyMonitor
    import numpy as np
    from scipy.optimize import nnls
    source = NVMLPower([args.device], [0])
    profiles, samples = {}, []
    try:
        # Unload all currently resident models before idle calibration.
        with urllib.request.urlopen(args.base_url+'/api/ps', timeout=10) as response:
            resident = json.load(response)['models']
        for item in resident:
            post(args.base_url, '/api/generate', {'model': item['name'], 'keep_alive': 0})
        time.sleep(args.settle_seconds)
        readings = []
        deadline = time.monotonic()+args.idle_seconds
        while time.monotonic() < deadline:
            readings.append(source.read_watts()[0])
            time.sleep(.1)
        idle = float(np.mean(readings))
        source.idle_watts = [idle]
        for size in ('0.5b', '1.5b', '3b', '7b'):
            name = 'progrouter-coder:'+size
            rows = []
            # Vary input and output lengths; collect both cold and warm calls.
            for repeats, output in ((1, 64), (32, 64), (1, 256), (64, 256)):
                post(args.base_url, '/api/generate', {'model': name, 'keep_alive': 0})
                for cold in (True, False):
                    monitor = EnergyMonitor(source)
                    monitor.start()
                    started = time.monotonic()
                    try:
                        result = post(args.base_url, '/api/chat', {
                            'model': name, 'stream': False, 'keep_alive': '5m',
                            'messages': [{'role': 'user', 'content':
                                f'Sample {len(samples)}. Explain a Python algorithm with examples. '+('Consider edge cases. '*repeats)}],
                            'options': {'num_ctx': 4096, 'num_predict': output, 'temperature': 0}})
                    finally:
                        measured = monitor.stop()
                    if measured['energy_joules'] is None:
                        raise RuntimeError('Energy measurement failed')
                    row = {'model': name, 'cold': cold, 'input_tokens': result['prompt_eval_count'],
                           'output_tokens': result['eval_count'], 'seconds': time.monotonic()-started,
                           'server_load_seconds': result.get('load_duration', 0)/1e9,
                           'server_prefill_seconds': result.get('prompt_eval_duration', 0)/1e9,
                           'energy': measured['energy_joules']}
                    rows.append(row)
                    samples.append(row)
                    print(json.dumps(row), flush=True)
            design = np.asarray([[int(r['cold']), r['input_tokens'], r['output_tokens']] for r in rows], dtype=float)
            energy = nnls(design, np.asarray([r['energy'] for r in rows]))[0]
            latency = nnls(design, np.asarray([r['seconds'] for r in rows]))[0]
            coefficients = dict(zip(('load_joules', 'prefill_joules_per_token', 'decode_joules_per_token',
                                     'load_seconds', 'prefill_seconds_per_token', 'decode_seconds_per_token'),
                                    map(float, list(energy)+list(latency))))
            profiles[name] = {'energy': max(r['energy'] for r in rows),
                              'seconds': max(r['seconds'] for r in rows), 'coefficients': coefficients,
                              'fit_diagnostics': {
                                  'energy_mae': float(np.mean(np.abs(design@energy-np.asarray([r['energy'] for r in rows])))),
                                  'latency_mae': float(np.mean(np.abs(design@latency-np.asarray([r['seconds'] for r in rows])))),
                                  'sample_count': len(rows), 'validation': 'in-sample only'}}
        result = {'profiles': profiles, 'samples': samples,
            'energy_measurement': {'devices': [args.device], 'idle_watts': [idle], 'interval': .1},
            'routing': {'energy_budget': None, 'time_budget': None, 'average_energy': None,
                        'tradeoff': 1000, 'epsilon': .1, 'epsilon_decay': .99,
                        'epsilon_min': .01, 'seed': 0, 'cost_basis': 'measured_task'}}
        with args.output.open('x') as file:
            json.dump(result, file, indent=2)
        print('Saved calibration. Review fit and set budgets before generating routing config.')
    finally:
        source.close()


if __name__ == '__main__':
    main()
