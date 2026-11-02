"""Sample local NVML board power and integrate gross/idle-adjusted joules.

Monitoring is observational: remote inference and overlapping GPU workloads cannot
be attributed to this run. Select dedicated inference GPUs, including coordinator.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
import threading
import time


@dataclass(frozen=True)
class TaskBudget:
    energy_joules: float | None = None
    seconds: float | None = None

    def __post_init__(self):
        for value in (self.energy_joules, self.seconds):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError('Task budgets must be finite and positive')


class NVMLPower:
    def __init__(self, devices, idle_watts):
        if not devices or len(set(devices)) != len(devices) or len(devices) != len(idle_watts):
            raise ValueError('Supply unique NVML device indices and one idle baseline per device')
        if any(not math.isfinite(x) or x < 0 for x in idle_watts):
            raise ValueError('Idle baselines must be finite nonnegative watts')
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        try:
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in devices]
            self.devices = [str(pynvml.nvmlDeviceGetUUID(h)) for h in self.handles]
            self.idle_watts = list(idle_watts)
            self.read_watts()  # Fail before inference if measurement is unavailable.
        except BaseException:
            pynvml.nvmlShutdown()
            raise

    def read_watts(self):
        return [self.nvml.nvmlDeviceGetPowerUsage(h)/1000. for h in self.handles]

    def close(self):
        self.nvml.nvmlShutdown()


class EnergyMonitor:
    def __init__(self, source, interval=.1, clock=time.monotonic):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError('Sampling interval must be positive')
        self.source, self.interval, self.clock = source, interval, clock
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.started = False
        self.error = None
        self.gross = self.incremental = 0.0
        self.previous = None
        self.samples = 0
        self.log = None

    def _sample(self):
        with self.lock:
            if self.error:
                return
            try:
                watts = list(self.source.read_watts())
                now = self.clock()
                if len(watts) != len(self.source.idle_watts) or not watts or any(not math.isfinite(p) or p < 0 for p in watts):
                    raise ValueError('Invalid power observation')
                gross = sum(watts)
                active = sum(max(0., p-idle) for p, idle in zip(watts, self.source.idle_watts))
                if self.previous is not None:
                    then, old_gross, old_active = self.previous
                    dt = now-then
                    if dt < 0:
                        raise ValueError('Nonmonotonic sample time')
                    self.gross += dt*(old_gross+gross)/2
                    self.incremental += dt*(old_active+active)/2
                else:
                    self.start_time = now
                self.previous = (now, gross, active)
                self.samples += 1
                if self.log:
                    self.log.write(json.dumps({'seconds': now-self.start_time, 'watts': watts,
                        'gross_joules': self.gross, 'incremental_joules': self.incremental})+'\n')
                    self.log.flush()
            except Exception as exc:
                self.error = type(exc).__name__

    def start(self, path=None):
        if self.started:
            raise ValueError('Use a fresh energy monitor per task')
        if path is not None:
            self.log = open(path, 'x', encoding='utf-8')
        self.started = True
        self._sample()
        if self.error:
            if self.log:
                self.log.close()
            self.started = False
            raise RuntimeError('Energy measurement unavailable: '+self.error)
        def sample_loop():
            while not self.stop_event.wait(self.interval):
                self._sample()
        self.thread = threading.Thread(target=sample_loop, daemon=True)
        self.thread.start()

    def snapshot(self):
        if not self.started:
            return {'status': 'not_started', 'energy_joules': None}
        self._sample()
        with self.lock:
            return {'status': 'error' if self.error else 'measured',
                'energy_joules': None if self.error else self.incremental,
                'gross_joules': None if self.error else self.gross,
                'partial_incremental_joules': self.incremental,
                'samples': self.samples, 'error': self.error,
                'devices': self.source.devices, 'idle_watts': self.source.idle_watts,
                'interval_seconds': self.interval}

    def stop(self):
        if not self.started:
            return {'status': 'not_started', 'energy_joules': None}
        self.stop_event.set()
        if getattr(self, 'thread', None) is not None and self.thread.is_alive():
            self.thread.join()
        result = self.snapshot()
        if self.log:
            self.log.close()
            self.log = None
        return result


def main():
    """Read-only power sampling smoke test; does not run any workload."""
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--devices', type=int, nargs='+', required=True)
    parser.add_argument('--idle-watts', type=float, nargs='+', required=True)
    parser.add_argument('--seconds', type=float, default=2.)
    parser.add_argument('--interval', type=float, default=.1)
    parser.add_argument('--output', type=Path, required=True, help='New power-sample JSONL file')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds must be positive')
    source = NVMLPower(args.devices, args.idle_watts)
    monitor = EnergyMonitor(source, args.interval)
    try:
        monitor.start(args.output)
        try:
            threading.Event().wait(args.seconds)
        finally:
            result = monitor.stop()
        print(json.dumps(result, indent=2))
    finally:
        source.close()


if __name__ == '__main__':
    main()
