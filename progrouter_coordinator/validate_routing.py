"""Real local algorithm smoke test, NOT HumanEval accuracy evaluation.

Explicit invocation runs Ollama and NVML, with no generated-code execution.
Uses permissive validation-only budgets, a four-sample training threshold, and
epsilon zero after cold start to exercise both exploration and exploitation.
"""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-tasks', type=int, default=6)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new validation output directory')
    if args.max_tasks < 1:
        parser.error('max-tasks must be positive')
    from configure_routing import build
    from energy import EnergyMonitor, NVMLPower, TaskBudget
    from progress import ProgressPredictor, ProgressScorer, SentenceEncoder
    from routing import ModelOption, ProgressRouter, RoutingConfig
    from workflow import CoordinatorWorkflow, HTTPChatClient, Limits, Model, Task, DEFAULT_ROLES
    calibration = json.loads(args.calibration.read_text())
    calibration['routing'].update(energy_budget=100000, time_budget=120,
        average_energy=500, tradeoff=10000, epsilon=0, epsilon_min=0)
    config = build(json.loads(Path(__file__).with_name('examples').joinpath('ollama_config.json').read_text()), calibration)
    config['limits'].update(max_worker_steps=4, max_llm_calls=14, max_seconds=120)
    config['validation_only'] = True
    args.output.mkdir(parents=True)
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    encoder = SentenceEncoder()
    predictor = ProgressPredictor(encoder, min_samples=4, update_every=1, n_estimators=16)
    router = ProgressRouter({r: [ModelOption(Model(**o['model']), o['energy'], o['seconds'], o.get('profile'))
        for o in pool] for r, pool in config['candidates'].items()}, predictor, RoutingConfig(**config['routing']))
    router.measurement_signature = config['energy_measurement']
    source = NVMLPower(config['energy_measurement']['devices'], config['energy_measurement']['idle_watts'])
    records = []
    decisions = []
    try:
        for i in range(args.max_tasks):
            run_dir = args.output/f'{i:05d}'
            task = Task(f'algorithm-smoke/{i}', f'Implement value_{i}() in Python returning the integer {i}. '
                'Execution tools are disabled and cannot be enabled by workers. '
                'Produce full code, perform at most one review, then finish with unverified correctness.')
            engine = CoordinatorWorkflow(HTTPChatClient(), Model(**config['coordinator']), router,
                DEFAULT_ROLES, Limits(**config['limits']), progress_scorer=ProgressScorer(encoder),
                energy_monitor=EnergyMonitor(source), task_budget=TaskBudget(100000, 120))
            ledger = engine.run(task, run_dir)
            record = {'task_id': task.task_id, 'directory': f'{i:05d}', 'status': ledger['status'],
                      'stop_reason': ledger.get('stop_reason'), 'metrics': ledger['metrics']}
            records.append(record)
            task_decisions = [e['routing'] for e in ledger['history'] if e.get('routing')]
            decisions.extend(task_decisions)
            print(json.dumps(record), flush=True)
            if predictor.meta is not None and any(d['reason'] == 'argmax' for d in task_decisions):
                break
    finally:
        source.close()
        checkpoint_roundtrip = False
        if records:
            restored = ProgressRouter(router.pools, ProgressPredictor(encoder), router.config)
            restored.measurement_signature = config['energy_measurement']
            restored.restore(run_dir/'router.joblib')
            assert restored.queue == router.queue
            assert restored.collected_samples == router.collected_samples
            assert restored.completed_tasks == router.completed_tasks
            assert restored.random.getstate() == router.random.getstate()
            assert [s[2] for s in restored.predictor.samples] == [s[2] for s in predictor.samples]
            checkpoint_roundtrip = True
        result = {'validation_only': True, 'generated_code_executed': False,
                  'checkpoint_roundtrip': checkpoint_roundtrip,
                  'hidden_correctness_validated': False, 'tasks': records,
                  'predictor_ready': predictor.meta is not None, 'collected_samples': predictor.seen,
                  'exploration_steps': sum(d['exploration'] for d in decisions),
                  'exploitation_steps': sum(not d['exploration'] for d in decisions),
                  'selected_models': dict(Counter(d['selected'] for d in decisions)),
                  'queue': router.queue}
        (args.output/'validation_summary.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
