"""CLI for the coordinator-workflow milestone, not the full ProgRouter router."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from workflow import (CoordinatorWorkflow, DEFAULT_ROLES, DockerTests, FixedModelSelector,
                      HTTPChatClient, Limits, Model, Reply, Task)


class ScriptedDemo:
    """Offline plumbing demonstration. No model inference or test execution."""
    def __init__(self):
        self.responses = iter([
            {"action": "dispatch", "role": "analyst", "subtask": "Clarify edge cases",
             "plan": ["Analyze requirements", "Implement", "Review"], "reason": "Need an approach"},
            {"message": "Handle zero and positive n; reject negative inputs.", "code": None, "run_tests": False},
            {"action": "dispatch", "role": "coder", "subtask": "Implement the proposed algorithm",
             "reason": "Approach is available"},
            {"message": "Initial implementation", "code": "def triangular(n):\n    return n * (n + 1) / 2\n", "run_tests": False},
            {"action": "dispatch", "role": "reviewer", "subtask": "Review boundary behavior and return type",
             "reason": "Check initial code before finishing"},
            {"message": "Use integer division; explicitly reject negative n.", "code": None, "run_tests": False},
            {"action": "dispatch", "role": "coder", "subtask": "Repair the issues identified by the reviewer",
             "plan": ["Requirements analyzed", "Repair integer return and negative inputs", "Finish"],
             "reason": "Revision is needed"},
            {"message": "Revised implementation; not independently evaluated.",
             "code": "def triangular(n):\n    if n < 0:\n        raise ValueError('n must be nonnegative')\n    return n * (n + 1) // 2\n", "run_tests": False},
            {"action": "finish", "reason": "The requested revision is available",
             "final_answer": "See current_code for the proposed solution."},
        ])

    def complete(self, model, messages, timeout):
        return Reply(json.dumps(next(self.responses)), {}, "stop")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, default=Path("examples/task.json"))
    parser.add_argument("--config", type=Path, default=Path("examples/config.json"))
    parser.add_argument("--output", type=Path, required=True, help="New, empty run directory")
    parser.add_argument("--demo", action="store_true", help="Explicit scripted offline plumbing demo")
    parser.add_argument("--enable-tests", action="store_true", help="Execute visible tests in Docker only")
    parser.add_argument("--enable-progress", action="store_true", help="Score visible progress using locally cached MiniLM weights")
    parser.add_argument("--predictor-checkpoint", type=Path, help="Trusted predictor checkpoint to resume; uses the same MiniLM encoder")
    parser.add_argument("--router-checkpoint", type=Path, help="Trusted complete routing stream checkpoint")
    args = parser.parse_args()
    if args.router_checkpoint and args.predictor_checkpoint:
        parser.error("Choose either a complete router checkpoint or predictor-only initialization")
    if args.demo and args.enable_tests:
        parser.error("--demo does not execute generated code; omit --enable-tests")
    try:
        task = Task.load(args.task)
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        coordinator = Model(**cfg["coordinator"])
        roles = cfg.get("roles", DEFAULT_ROLES)
        routing_cfg = cfg.get("routing")
        if (args.predictor_checkpoint or args.router_checkpoint) and routing_cfg is None:
            parser.error("Checkpoints require routing configuration")
        if args.demo and routing_cfg is not None:
            parser.error("Scripted demo cannot train routing models; use the fixed configuration")
        if args.demo:
            if task.task_id != "triangular-demo":
                parser.error("The scripted demo is only for examples/task.json (triangular-demo)")
            if not {"analyst", "coder", "reviewer"}.issubset(roles):
                parser.error("Scripted demo requires analyst, coder, reviewer roles")
        tool = DockerTests(**cfg.get("visible_test_tool", {})) if args.enable_tests else None
        scorer = predictor = None
        if args.enable_progress or routing_cfg is not None:
            from progress import ProgressScorer, SentenceEncoder
            encoder = SentenceEncoder()
            scorer = ProgressScorer(encoder)
        if routing_cfg is not None:
            from progress import ProgressPredictor
            from routing import ModelOption, ProgressRouter, RoutingConfig
            predictor = (ProgressPredictor.load(args.predictor_checkpoint, encoder)
                         if args.predictor_checkpoint else ProgressPredictor(encoder, **cfg.get("predictor", {})))
            pools = {role: [ModelOption(Model(**option["model"]), option["energy"], option["seconds"], option.get("profile"))
                            for option in options] for role, options in cfg["candidates"].items()}
            selector = ProgressRouter(pools, predictor, RoutingConfig(**routing_cfg))
            if selector.config.cost_basis == "measured_task":
                selector.measurement_signature = cfg.get("energy_measurement")
            if args.router_checkpoint:
                selector.restore(args.router_checkpoint)
        else:
            workers = {role: Model(**value) for role, value in cfg["workers"].items()}
            selector = FixedModelSelector(workers)
        from energy import TaskBudget, EnergyMonitor, NVMLPower
        energy_cfg = cfg.get('energy_measurement')
        if args.demo and energy_cfg:
            parser.error('Scripted demo does not support energy measurement')
        budgets = cfg.get('task_budget', {})
        if routing_cfg and routing_cfg.get('cost_basis') == 'measured_task' and not budgets:
            budgets = {'energy_joules': routing_cfg['energy_budget'], 'seconds': routing_cfg['time_budget']}
        task_budget = TaskBudget(**budgets)
        source = NVMLPower(energy_cfg['devices'], energy_cfg['idle_watts']) if energy_cfg else None
        try:
            monitor = EnergyMonitor(source, interval=energy_cfg.get('interval', .1)) if source else None
            workflow = CoordinatorWorkflow(ScriptedDemo() if args.demo else HTTPChatClient(),
                coordinator, selector, roles, Limits(**cfg.get("limits", {})), tool,
                progress_scorer=scorer, progress_predictor=predictor,
                energy_monitor=monitor, task_budget=task_budget)
            result = workflow.run(task, args.output)
        finally:
            if source is not None:
                source.close()
        if predictor is not None:
            predictor.save(args.output / "predictor.joblib")
        print(json.dumps({"mode": "scripted_demo" if args.demo else "real_http_inference",
                          "status": result["status"], "metrics": result["metrics"],
                          "output": str(args.output)}, indent=2))
        return 0 if result["status"] == "coordinator_finished" else 1
    except (ValueError, KeyError, OSError, ImportError, TypeError) as exc:
        print(f"Configuration/input error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
