"""Candidate comparison and Eqs. (9)–(12), using explicit per-call estimates.

Profiles are supplied/calibrated by the operator, not measured energy. Budgets
cover worker LLM attempts, including protocol retries; coordinator/tools excluded.
"""
from dataclasses import dataclass, asdict, replace
import math
import random

from progress import Candidate
from workflow import LimitReached, Model


@dataclass(frozen=True)
class ModelOption:
    model: Model
    energy: float
    seconds: float
    profile: dict | None = None

    def __post_init__(self):
        if any(not math.isfinite(x) or x < 0 for x in (self.energy, self.seconds)):
            raise ValueError('Candidate costs must be finite and nonnegative')
        if self.profile is not None:
            for key in ('load_joules', 'prefill_joules_per_token', 'decode_joules_per_token',
                        'load_seconds', 'prefill_seconds_per_token', 'decode_seconds_per_token'):
                value = self.profile[key]
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError('Invalid calibrated cost coefficient: '+key)

    def estimate(self, ledger, subtask, *, messages=None, cold=None):
        if self.profile is None:
            return self
        from workflow import WORKER_PROMPT, prompt_ledger
        from token_budget import input_tokens
        import json
        role = self.profile['role']
        payload = {'role': role, 'role_description': self.profile['role_description'],
                   'subtask': subtask, 'ledger': prompt_ledger(ledger)}
        if messages is None:
            messages = [{'role': 'system', 'content': WORKER_PROMPT},
                        {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
        count = input_tokens(self.model, messages)
        # One resident model: each worker follows the configured coordinator.
        if cold is None:
            cold = self.model.model != self.profile['coordinator_model']
        p = self.profile
        energy = cold*p['load_joules'] + count*p['prefill_joules_per_token'] + self.model.max_tokens*p['decode_joules_per_token']
        seconds = cold*p['load_seconds'] + count*p['prefill_seconds_per_token'] + self.model.max_tokens*p['decode_seconds_per_token']
        return replace(self, energy=max(self.energy, energy), seconds=max(self.seconds, seconds))


@dataclass(frozen=True)
class RoutingConfig:
    energy_budget: float
    time_budget: float
    average_energy: float
    tradeoff: float = 100.0
    queue: float = 0.0
    epsilon: float = 0.1
    seed: int = 0
    epsilon_decay: float = 0.99
    epsilon_min: float = 0.0
    cost_basis: str = "estimated_worker"

    def __post_init__(self):
        if any(not math.isfinite(x) or x < 0 for x in (
                self.energy_budget, self.time_budget, self.average_energy,
                self.tradeoff, self.queue, self.epsilon, self.epsilon_decay, self.epsilon_min)):
            raise ValueError('Routing settings must be finite and nonnegative')
        if self.energy_budget <= 0 or self.time_budget <= 0 or self.epsilon > 1:
            raise ValueError('Budgets must be positive; epsilon must be in [0,1]')
        if self.cost_basis not in {"estimated_worker", "measured_task"}:
            raise ValueError("cost_basis must be estimated_worker or measured_task")
        if not 0 < self.epsilon_decay <= 1 or not 0 <= self.epsilon_min <= self.epsilon:
            raise ValueError('Require decay in (0,1] and epsilon_min in [0,epsilon]')


class ProgressRouter:
    def __init__(self, pools: dict[str, list[ModelOption]], predictor, config: RoutingConfig):
        if not pools or any(not options for options in pools.values()):
            raise ValueError('Each role needs a nonempty candidate pool')
        for options in pools.values():
            if len({o.model.model for o in options}) != len(options):
                raise ValueError('Candidate model names must be unique within each role')
        self.pools, self.predictor, self.config = pools, predictor, config
        # Compatibility with the workflow role-set validation.
        self.workers = {role: options[0].model for role, options in pools.items()}
        self.random = random.Random(config.seed)
        self.queue = config.queue
        self.collected_samples = 0
        self.completed_tasks = 0
        self.task_finalized = True
        self.begin_task()

    @property
    def epsilon(self):
        return max(self.config.epsilon_min,
                   self.config.epsilon * self.config.epsilon_decay ** self.collected_samples)

    def begin_task(self):
        if not self.task_finalized and getattr(self, "energy_used", 0) > 0:
            raise ValueError('Finalize the previous task before beginning another')
        self.task_finalized = False
        self.energy_used = self.time_used = 0.0
        self.estimated_energy = self.estimated_seconds = 0.0
        self.measurement_valid = self.config.cost_basis != "measured_task"
        self.last_decision = None
        self.active_option = None

    def _feasible(self, option):
        return (self.energy_used + option.energy <= self.config.energy_budget
                and self.time_used + option.seconds <= self.config.time_budget)

    def select(self, role, ledger, *, subtask, score):
        if self.task_finalized:
            raise ValueError('Call begin_task before dispatch')
        if score is None:
            raise ValueError('Progress routing requires a progress scorer')
        cfg = self.config
        rows = []
        evaluated_options = []
        for option in self.pools[role]:
            option = option.estimate(ledger, subtask)
            evaluated_options.append(option)
            prediction = self.predictor.predict(ledger, score, Candidate(role, option.model.model, subtask))
            feasible = self._feasible(option)
            row = {'model': option.model.model, 'estimated_energy': option.energy,
                   'estimated_seconds': option.seconds, 'feasible': feasible,
                   'prediction': prediction, 'score': None}
            if prediction['ready']:
                gain = prediction['gain']
                if not isinstance(gain, (int, float)) or not math.isfinite(gain) or not -1 <= gain <= 1:
                    raise ValueError('Predicted gain must be finite and in [-1,1]')
                # Ratios are at most one for feasible models; avoid exp overflow
                # for infeasible candidates, whose score is never used.
                if feasible:
                    progress = cfg.tradeoff * (1 - score.value) * gain
                    queue = self.queue * (option.energy - cfg.average_energy)
                    time = math.exp((self.time_used + option.seconds) / cfg.time_budget) * option.seconds
                    energy = math.exp((self.energy_used + option.energy) / cfg.energy_budget) * option.energy
                    row.update(progress_term=progress, queue_penalty=queue,
                               time_penalty=time, energy_penalty=energy,
                               score=progress - queue - time - energy)
            rows.append(row)
        eligible = [i for i, row in enumerate(rows) if row['feasible']]
        self.last_decision = {'role': role, 'subtask': subtask, 'candidates': rows,
                              'energy_used': self.energy_used, 'seconds_used': self.time_used,
                              'config': asdict(cfg), 'queue': self.queue, 'epsilon': self.epsilon,
                              'collected_samples': self.collected_samples,
                              'selected': None, 'exploration': False}
        if not eligible:
            raise LimitReached('routing_budget: no feasible candidate')
        cold = any(not rows[i]['prediction']['ready'] for i in eligible)
        exploration = cold or self.random.random() < self.epsilon
        selected = self.random.choice(eligible) if exploration else max(eligible, key=lambda i: rows[i]['score'])
        self.active_option = evaluated_options[selected]
        self.active_attempts = 0
        self.last_decision.update(selected=self.active_option.model.model, exploration=exploration,
                                  reason='cold_start' if cold else ('epsilon' if exploration else 'argmax'))
        return self.active_option.model

    def charge_attempt(self, model, *, messages=None):
        """Reserve calibrated estimates before every worker attempt, even failures."""
        if self.task_finalized:
            raise ValueError('Cannot charge a finalized task')
        option = self.active_option
        if option is None or option.model != model:
            raise ValueError('Worker model does not match selected candidate')
        if option.profile is not None and messages is not None:
            base = next(o for o in self.pools[self.last_decision['role']] if o.model == model)
            cold = self.active_attempts == 0 and model.model != base.profile['coordinator_model']
            option = base.estimate({}, '', messages=messages, cold=cold)
        if not self._feasible(option):
            raise LimitReached('routing_budget: worker attempt exceeds remaining budget')
        self.estimated_energy += option.energy
        self.estimated_seconds += option.seconds
        self.energy_used += option.energy
        self.time_used += option.seconds
        self.active_attempts += 1

    def sync_measurement(self, energy, seconds):
        if self.config.cost_basis == 'measured_task':
            self.measurement_valid = energy is not None
            if energy is not None:
                self.energy_used, self.time_used = energy, seconds

    def observe_transition(self, ledger, before, candidate, after, *, exploration):
        fitted = self.predictor.observe(ledger, before, candidate, after, exploration=exploration)
        if exploration:
            self.collected_samples += 1
        return fitted

    def finish_task(self):
        """Eq. (7), once per terminal task, including aborts and failed attempts."""
        if self.task_finalized:
            return None
        previous = self.queue
        if self.measurement_valid:
            self.queue = max(0.0, previous + self.energy_used - self.config.average_energy)
        self.completed_tasks += 1
        self.task_finalized = True
        return {'queue_before': previous, 'queue_after': self.queue,
                'task_cost': self.energy_used if self.measurement_valid else None,
                'cost_basis': self.config.cost_basis, 'queue_updated': self.measurement_valid,
                'average_energy': self.config.average_energy,
                'completed_tasks': self.completed_tasks}

    def _signature(self):
        return {'measurement': getattr(self, 'measurement_signature', None),
                'features': getattr(self.predictor, 'feature_signature', None), 'config': asdict(self.config),
                'pools': {role: [asdict(option) for option in options]
                          for role, options in self.pools.items()}}

    def save(self, path):
        """Atomic terminal-task checkpoint. Load only trusted joblib files."""
        if not self.task_finalized:
            raise ValueError('Router checkpoints require a finalized task')
        import joblib
        from pathlib import Path
        import tempfile
        path = Path(path)
        predictor_state = dict(self.predictor.__dict__)
        predictor_state.pop('encoder', None)
        state = {'version': 1, 'signature': self._signature(),
                 'queue': self.queue, 'completed_tasks': self.completed_tasks,
                 'collected_samples': self.collected_samples,
                 'random_state': self.random.getstate(), 'predictor': predictor_state}
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.', delete=False) as file:
            temp = Path(file.name)
        try:
            joblib.dump(state, temp)
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)

    def restore(self, path):
        """Restore the whole stream; matching configuration/encoder are required.

        joblib can execute code: never load an untrusted checkpoint.
        """
        import joblib
        if self.energy_used or self.completed_tasks:
            raise ValueError('Restore only into a fresh router')
        state = joblib.load(path)
        signature = state.get('signature', {})
        signature.setdefault('measurement', None)
        signature.get('config', {}).setdefault('cost_basis', 'estimated_worker')
        if state.get('version') != 1 or signature != self._signature():
            raise ValueError('Router checkpoint configuration or candidate pools differ')
        if not math.isfinite(state['queue']) or state['queue'] < 0:
            raise ValueError('Invalid checkpoint queue')
        self.predictor.__dict__.update(state['predictor'])
        self.queue, self.completed_tasks = state['queue'], state['completed_tasks']
        self.collected_samples = state['collected_samples']
        self.random.setstate(state['random_state'])
        self.task_finalized = True

    def metrics(self):
        return {'estimated_worker_energy': self.estimated_energy,
                'estimated_worker_seconds': self.estimated_seconds,
                'accounted_energy': self.energy_used if self.measurement_valid else None,
                'cost_basis': self.config.cost_basis,
                'energy_budget': self.config.energy_budget, 'time_budget': self.config.time_budget,
                'queue': self.queue, 'epsilon': self.epsilon,
                'predictor_ready': getattr(self.predictor, 'meta', None) is not None,
                'buffer_samples': len(getattr(self.predictor, 'samples', [])),
                'collected_samples': self.collected_samples, 'completed_tasks': self.completed_tasks}
