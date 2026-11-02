import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from energy import EnergyMonitor, NVMLPower, TaskBudget
from workflow import CoordinatorWorkflow, FixedModelSelector, Model, Task
from test_workflow import ScriptedClient, dispatch, worker, finish


class Source:
    devices = ['fake-gpu']
    idle_watts = [10.]
    def __init__(self):
        self.watts = 30.
    def read_watts(self):
        return [self.watts]


class Meter:
    def __init__(self):
        self.energy = 0.
        self.fail = False
    def start(self, path):
        pass
    def snapshot(self):
        return {'status': 'error' if self.fail else 'measured', 'energy_joules': None if self.fail else self.energy}
    def stop(self):
        return self.snapshot()


class EnergyTests(unittest.TestCase):
    def test_trapezoid_and_idle_subtraction(self):
        source = Source()
        now = [0.]
        monitor = EnergyMonitor(source, interval=1000, clock=lambda: now[0])
        with tempfile.TemporaryDirectory() as folder:
            monitor.start(Path(folder)/'energy.jsonl')
            now[0], source.watts = 2., 50.
            result = monitor.snapshot()
            self.assertAlmostEqual(result['gross_joules'], 80.)
            self.assertAlmostEqual(result['energy_joules'], 60.)
            now[0], source.watts = 3., 0.
            result = monitor.stop()
            self.assertAlmostEqual(result['energy_joules'], 80.)
            self.assertEqual(len((Path(folder)/'energy.jsonl').read_text().splitlines()), 3)

    def test_measurement_error_is_unknown_not_zero(self):
        source = Source()
        monitor = EnergyMonitor(source, interval=1000)
        monitor.start()
        source.watts = float('nan')
        result = monitor.stop()
        self.assertEqual(result['status'], 'error')
        self.assertIsNone(result['energy_joules'])

    def test_initial_measurement_failure_is_stoppable(self):
        source = Source()
        source.watts = -1
        monitor = EnergyMonitor(source)
        with self.assertRaises(RuntimeError):
            monitor.start()
        self.assertIsNone(monitor.stop()['energy_joules'])

    def test_nvml_milliwatts_to_watts(self):
        from types import SimpleNamespace
        nvml = SimpleNamespace(nvmlInit=lambda: None, nvmlShutdown=lambda: None,
            nvmlDeviceGetHandleByIndex=lambda i: i, nvmlDeviceGetUUID=lambda i: 'gpu-'+str(i),
            nvmlDeviceGetPowerUsage=lambda h: 150000)
        with patch.dict('sys.modules', pynvml=nvml):
            source = NVMLPower([0], [50])
            self.assertEqual(source.read_watts(), [150.])
            source.close()

    def test_budgets_require_measurement(self):
        m = Model('http://localhost/v1', 'small')
        with self.assertRaises(ValueError):
            CoordinatorWorkflow(ScriptedClient([]), m, FixedModelSelector({'coder': m}), {'coder':'code'}, task_budget=TaskBudget(1))
        for value in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                TaskBudget(value)

    def engine_case(self, *, energy_budget=100, fail=False):
        from routing import ProgressRouter, RoutingConfig, ModelOption
        from progress import ProgressScorer
        from test_progress import TestEncoder
        from test_routing import Predictor
        meter = Meter()
        m = Model('http://localhost/v1', 'small')
        router = ProgressRouter({'coder': [ModelOption(m, 1, 1)]}, Predictor(),
            RoutingConfig(energy_budget, 100, 0, queue=10, cost_basis='measured_task'))
        class Client(ScriptedClient):
            def complete(self, *args):
                meter.energy += 1
                if fail and self.requests:
                    meter.fail = True
                return super().complete(*args)
        client = Client([dispatch('coder'), worker(code='def f(): return 1'), finish()])
        engine = CoordinatorWorkflow(client, m, router, {'coder': 'code'}, progress_scorer=ProgressScorer(TestEncoder()),
            energy_monitor=meter, task_budget=TaskBudget(energy_budget, 100))
        with tempfile.TemporaryDirectory() as folder:
            result = engine.run(Task('t', 'Implement f'), Path(folder)/'run')
            saved = json.loads((Path(folder)/'run/ledger.json').read_text())
        return result, router, client

    def test_measured_whole_task_updates_queue_and_metrics(self):
        result, router, client = self.engine_case()
        self.assertEqual(result['status'], 'coordinator_finished')
        self.assertEqual(result['metrics']['energy_joules'], 3)
        self.assertEqual(router.queue, 13)
        self.assertEqual(result['metrics']['routing']['estimated_worker_energy'], 1)
        self.assertEqual(result['metrics']['routing']['accounted_energy'], 3)

    def test_energy_limit_stops_further_calls_and_preserves_overshoot(self):
        result, router, client = self.engine_case(energy_budget=1.5)
        # Coordinator costs one joule, leaving insufficient headroom for worker.
        self.assertEqual(result['status'], 'limit_reached')
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(result['metrics']['energy_joules'], 1)
        self.assertEqual(router.queue, 11)

    def test_inflight_call_can_overshoot_and_is_reported(self):
        meter = Meter()
        m = Model('http://localhost/v1', 'small')
        class Client(ScriptedClient):
            def complete(self, *args):
                meter.energy += 2
                return super().complete(*args)
        engine = CoordinatorWorkflow(Client([dispatch('coder'), worker(code='def f(): return 1')]),
            m, FixedModelSelector({'coder':m}), {'coder':'code'}, energy_monitor=meter,
            task_budget=TaskBudget(3))
        with tempfile.TemporaryDirectory() as folder:
            result = engine.run(Task('t','Implement f'), Path(folder)/'run')
        self.assertEqual(result['status'], 'limit_reached')
        self.assertEqual(result['stop_reason'], 'task_energy_joules')
        self.assertEqual(result['current_code'], 'def f(): return 1')
        self.assertEqual(len(result['history']), 1)
        self.assertEqual(result['metrics']['energy_joules'], 4)
        self.assertTrue(result['metrics']['task_budget_exceeded']['energy'])

    def test_measurement_failure_freezes_queue(self):
        result, router, client = self.engine_case(fail=True)
        self.assertEqual(result['status'], 'error')
        self.assertIsNone(result['metrics']['energy_joules'])
        self.assertEqual(router.queue, 10)
        self.assertEqual(len(client.requests), 2)


if __name__ == '__main__':
    unittest.main()
