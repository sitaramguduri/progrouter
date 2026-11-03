"""Sections 3.2–3.3 reconstruction. Defaults are choices, not paper parameters.

No reference answers are consumed. A domain adapter supplies observable milestones;
'complete' means all visible milestones, never independently verified correctness.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol
import math
import ast


class Encoder(Protocol):
    def encode(self, sentences: list[str]): ...


class SentenceEncoder:
    """Lazy optional MiniLM backend. Loading may download model weights."""
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", *, local_files_only=True, device="cpu"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, local_files_only=local_files_only, device=device)

    @property
    def identity(self):
        return {'model_name': self.model_name, 'max_seq_length': self.model.max_seq_length,
                'normalize_embeddings': True}

    def encode(self, sentences):
        return self.model.encode(sentences, normalize_embeddings=True).tolist()


@dataclass(frozen=True)
class Observation:
    summary: str
    requirements: dict[str, bool] = field(default_factory=dict)
    valid: bool = True
    failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class RegimeWeights:
    base: float
    completion: float
    trend: float
    quality: float


@dataclass(frozen=True)
class Score:
    regime: str
    value: float
    completion: float
    trend: float
    quality: float
    semantic_change: float
    structural_change: float


class ProgressScorer:
    def __init__(self, encoder: Encoder, weights=None, trend_window=3):
        self.encoder = encoder
        self.weights = weights or {
            "invalid": RegimeWeights(0, 0, 0, 0),
            "recoverable": RegimeWeights(.1, .3, .05, .05),
            "partial_success": RegimeWeights(.4, .4, .05, .05),
            "complete": RegimeWeights(1, 0, 0, 0),
        }
        if trend_window < 1 or set(self.weights) != {"invalid", "recoverable", "partial_success", "complete"}:
            raise ValueError("Supply four regimes and a positive trend_window")
        for weight in self.weights.values():
            if any(not math.isfinite(v) or v < 0 for v in asdict(weight).values()):
                raise ValueError("Scoring weights must be finite and nonnegative")
        self.trend_window = trend_window

    def score(self, current: Observation, previous: Observation | None = None,
              recent_completion_deltas=()) -> Score:
        completion = sum(current.requirements.values()) / len(current.requirements) if current.requirements else 0.0
        if not current.valid:
            regime = "invalid"
        elif current.requirements and completion == 1 and not current.failures:
            regime = "complete"
        elif completion > 0:
            regime = "partial_success"
        else:
            regime = "recoverable"
        semantic = structural = quality = 0.0
        deltas = list(recent_completion_deltas)
        if previous is not None:
            old = sum(previous.requirements.values()) / len(previous.requirements) if previous.requirements else 0.0
            deltas.append(completion - old)
            vectors = self.encoder.encode([previous.summary, current.summary])
            a, b = vectors
            norm = math.sqrt(sum(float(x)**2 for x in a) * sum(float(x)**2 for x in b))
            similarity = sum(float(x)*float(y) for x, y in zip(a, b)) / norm if norm else 1.0
            semantic = (1 - max(-1.0, min(1.0, similarity))) / 2
            keys = set(previous.requirements) | set(current.requirements)
            structural = sum(int(current.requirements.get(k, False)) - int(previous.requirements.get(k, False)) for k in keys)
            structural += len(set(previous.failures) - set(current.failures)) - len(set(current.failures) - set(previous.failures))
            structural /= max(1, len(keys) + len(set(previous.failures) | set(current.failures)))
            # Semantic novelty is useful only alongside observable improvement.
            quality = structural * (.5 + .5 * semantic)
        trend = sum(deltas[-self.trend_window:]) / min(len(deltas), self.trend_window) if deltas else 0.0
        w = self.weights[regime]
        value = max(0.0, min(1.0, w.base + w.completion*completion + w.trend*trend + w.quality*quality))
        return Score(regime, value, completion, trend, quality, semantic, structural)


def coding_observation(ledger: dict) -> Observation:
    """Visible assertions are requirements; analysis-only steps cannot invent passes.

    Test observations are invalidated whenever code changes. Coordinator finish is
    intentionally not used as a completion signal.
    """
    tests = ledger["task"].get("visible_tests", ())
    requirements = {str(i): False for i in range(len(tests))}
    failures = ()
    code = None
    for entry in ledger.get("history", []):
        if entry.get("code") is not None and entry["code"] != code:
            requirements = dict.fromkeys(requirements, False)
            failures = ()
            code = entry["code"]
        feedback = entry.get("visible_feedback") or {}
        result = feedback.get("result")
        if (feedback.get("status") == "completed" and isinstance(result, dict)
                and result.get("total") == len(tests)
                and isinstance(result.get("failures"), list)) :
            errors = result.get("failures", [])
            failed = {f.get("index") for f in errors}
            requirements = {str(i): None not in failed and i not in failed for i in range(len(tests))}
            failures = tuple(str(f.get("error", "test failure")) for f in errors)
        elif feedback.get("status") in {"error", "timeout", "output_limit"}:
            failures = (feedback["status"],)
    history = ledger.get("history", [])
    worker_failed = bool(history and history[-1].get('worker_error'))
    if worker_failed:
        failures = tuple(failures) + ('worker_execution_failed',)
    if tests:
        # Keep a fixed requirement universe before and after code creation.
        # Some legacy ledgers have code only in history.
        source = ledger.get('current_code') or code or ''
        requirements['code_present'] = bool(source)
        try:
            parsed = ast.parse(source)
            requirements['syntax_valid'] = bool(source)
            requirements['function_defined'] = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                                                    for n in parsed.body)
        except SyntaxError:
            requirements['syntax_valid'] = False
            requirements['function_defined'] = False
            failures = tuple(failures) + ('syntax_error',)
    if not tests:
        # Dense, observable milestones for tasks with no executable public examples.
        # The final false requirement prevents syntax alone implying completion.
        source = ledger.get("current_code") or ""
        syntax = False
        interface = False
        if source:
            try:
                parsed = ast.parse(source)
                syntax = True
                interface = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                                for n in parsed.body)
            except SyntaxError:
                failures = tuple(failures) + ("syntax_error",)
        requirements = {"code_present": bool(source), "syntax_valid": syntax,
                        "function_defined": interface, "public_behavior_verified": False}
    summary = str(ledger.get("state_summary") or ledger["task"]["problem"]) + "\nPlan: " + str(ledger.get("plan", []))
    summary += "\nLatest worker: " + (str(history[-1].get("message", "")) if history else "none")
    summary += "\nCode: " + str(ledger.get("current_code") or "none")
    valid = 'syntax_error' not in failures and not (worker_failed and not ledger.get('current_code'))
    return Observation(summary, requirements, valid, failures)


@dataclass(frozen=True)
class Candidate:
    role: str
    model: str
    subtask: str


class ProgressPredictor:
    """Cold-start dual random forests and a tree meta-learner (Eqs. 3–6).

    Only explicitly exploration-labelled samples are collected. Periodic batch
    refits approximate online learning. This module does not implement routing.
    """
    def __init__(self, encoder: Encoder, *, min_samples=12, update_every=4,
                 max_samples=2000, random_state=0, n_estimators=64):
        if min_samples < 4 or update_every < 1 or max_samples < min_samples:
            raise ValueError("Invalid training buffer settings")
        self.encoder = encoder
        self.min_samples, self.update_every, self.max_samples = min_samples, update_every, max_samples
        self.random_state, self.n_estimators = random_state, n_estimators
        self.samples = []
        self.seen = 0
        self.structured = self.semantic = self.meta = None

    @property
    def feature_signature(self):
        return {'version': 2, 'encoder': getattr(self.encoder, 'identity',
            type(self.encoder).__module__+'.'+type(self.encoder).__qualname__)}

    def _features(self, ledger, score, candidate):
        history = ledger.get("history", [])
        features = {"progress": score.value, "completion": score.completion,
                    "trend": score.trend, "quality": score.quality,
                    "steps": len(history), "role=" + candidate.role: 1,
                    "model=" + candidate.model: 1, "regime=" + score.regime: 1}
        for step in history:
            key = "calls=" + str(step.get("model", "unknown"))
            features[key] = features.get(key, 0) + 1
        # MiniLM truncates long inputs. Preserve the routing decision at the front,
        # rather than appending it after potentially thousands of code tokens.
        text = "Candidate model: " + candidate.model + "\nRole: " + candidate.role
        text += "\nSubtask: " + candidate.subtask[:200]
        text += f"\nProgress: {score.regime}, g={score.value:.3f}, completion={score.completion:.3f}"
        latest = str(history[-1].get('message', '')) if history else 'initial'
        text += "\nLatest state: " + str(ledger.get('state_summary') or latest)[:600]
        text += "\nCurrent code: " + str(ledger.get('current_code') or 'none')[:600]
        text += "\nTask: " + ledger['task']['problem'][:400]
        embedding = list(map(float, self.encoder.encode([text])[0]))
        if not embedding or not all(math.isfinite(x) for x in embedding):
            raise ValueError("Encoder must return finite nonempty embeddings")
        return features, embedding

    def predict(self, ledger, score, candidate):
        if self.meta is None:
            return {"ready": False, "structured": None, "semantic": None, "gain": None}
        features, embedding = self._features(ledger, score, candidate)
        a = float(self.structured.predict(self.vectorizer.transform([features]))[0])
        b = float(self.semantic.predict([embedding])[0])
        gain = float(self.meta.predict([[a, b]])[0])
        return {"ready": True, "structured": a, "semantic": b, "gain": max(-1., min(1., gain))}

    def observe(self, ledger, before: Score, candidate: Candidate, after: Score, *, exploration: bool):
        if not exploration:
            return False
        features, embedding = self._features(ledger, before, candidate)
        self.samples.append((features, embedding, after.value - before.value))
        self.samples = self.samples[-self.max_samples:]
        self.seen += 1
        if len(self.samples) >= self.min_samples and (self.meta is None or self.seen % self.update_every == 0):
            self.fit()
            return True
        return False

    def save(self, path):
        """Persist buffer, encoder configuration and trained models (trusted pickle)."""
        import joblib
        # The encoder can own a GPU model; caller supplies it again when loading.
        state = dict(self.__dict__)
        state.pop("encoder")
        joblib.dump({"version": 2, "features": self.feature_signature, "state": state}, path)

    @classmethod
    def load(cls, path, encoder):
        """Load only files you trust: joblib/pickle can execute arbitrary code."""
        import joblib
        data = joblib.load(path)
        if data["version"] != 2:
            raise ValueError("Unsupported predictor checkpoint version")
        predictor = cls(encoder)
        if data.get('features') != predictor.feature_signature:
            raise ValueError('Predictor encoder/feature configuration differs')
        predictor.__dict__.update(data["state"])
        predictor.encoder = encoder
        return predictor

    def fit(self):
        if len(self.samples) < self.min_samples:
            raise ValueError("Not enough exploration samples")
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.model_selection import KFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.tree import DecisionTreeRegressor
        records, embeddings, targets = zip(*self.samples)
        targets = np.asarray(targets)
        def forest():
            return RandomForestRegressor(n_estimators=self.n_estimators, max_depth=8,
                                         random_state=self.random_state, n_jobs=1)
        folds = KFold(n_splits=min(5, len(targets)), shuffle=True, random_state=self.random_state)
        # Fit vectorizers within folds, including categorical vocabulary.
        structured = make_pipeline(DictVectorizer(sparse=False), forest())
        a = cross_val_predict(structured, list(records), targets, cv=folds)
        b = cross_val_predict(forest(), np.asarray(embeddings), targets, cv=folds)
        meta = DecisionTreeRegressor(max_depth=3, min_samples_leaf=2, random_state=self.random_state)
        meta.fit(np.column_stack([a, b]), targets)
        vectorizer = DictVectorizer(sparse=False)
        x = vectorizer.fit_transform(records)
        structured_final, semantic_final = forest(), forest()
        structured_final.fit(x, targets)
        semantic_final.fit(embeddings, targets)
        self.vectorizer, self.structured, self.semantic, self.meta = vectorizer, structured_final, semantic_final, meta
