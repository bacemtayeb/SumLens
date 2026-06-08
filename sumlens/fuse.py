"""Signal fusion, calibration, and labelling.

Fusion combines the three sentence-level signals into one grounding probability.
If a trained model is present at `model_path` it is used; otherwise we fall back to
an identity fusion (mean of grounding-oriented signals) so the pipeline still runs
without weights (CI, fresh checkout). Same idea for Platt calibration.

Convention: class 1 = **grounded**. The fusion model is trained with `grounded`
labels (1 if the summary sentence is grounded, 0 if hallucinated), so its
`predict_proba[:, 1]` is the grounding score that `label` thresholds. Feature order
is `FEATURE_ORDER`; missing signals are imputed to neutral 0.5.

Scikit-learn is imported lazily so the pipeline does not depend on it unless a
trained model is actually loaded or fitted.
"""

import pickle
from pathlib import Path
from typing import Any, Literal

from sumlens.types import AnalysisConfig, SignalScores

_NEUTRAL = 0.5
FEATURE_ORDER = ("classifier", "nli", "attribution")


def fuse(signals: dict[str, SignalScores], model_path: Path) -> dict[str, float]:
    if not model_path.exists():
        return {sentence_id: _grounding(score) for sentence_id, score in signals.items()}
    model = _load(model_path)
    ids = list(signals)
    features = [_feature_vector(signals[i]) for i in ids]
    grounded_proba = model.predict_proba(features)[:, 1]
    return {i: float(p) for i, p in zip(ids, grounded_proba, strict=True)}


def calibrate(scores: dict[str, float], platt_path: Path) -> dict[str, float]:
    if not platt_path.exists():
        return dict(scores)
    platt = _load(platt_path)
    ids = list(scores)
    calibrated = platt.predict_proba([[scores[i]] for i in ids])[:, 1]
    return {i: float(c) for i, c in zip(ids, calibrated, strict=True)}


def label(score: float, cfg: AnalysisConfig) -> Literal["grounded", "weak", "hallucinated"]:
    if score < cfg.tau_hallucinated:
        return "hallucinated"
    if score >= cfg.tau_grounded:
        return "grounded"
    return "weak"


def fit_fusion(features: list[list[float]], grounded: list[int]) -> Any:
    """Fit the fusion LogisticRegression. `grounded` = 1 if grounded, 0 if hallucinated."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000)
    model.fit(features, grounded)
    return model


def fit_platt(scores: list[float], grounded: list[int]) -> Any:
    """Fit a 1-D Platt calibrator mapping a fused score to a calibrated grounding prob."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000)
    model.fit([[s] for s in scores], grounded)
    return model


def _feature_vector(signals: SignalScores) -> list[float]:
    """Signals in FEATURE_ORDER; missing values imputed to neutral 0.5."""
    values = signals.model_dump()
    return [_NEUTRAL if values[name] is None else float(values[name]) for name in FEATURE_ORDER]


def _grounding(signals: SignalScores) -> float:
    """Identity fusion: mean of available grounding-oriented signals; 0.5 if none."""
    contributions = []
    if signals.classifier is not None:
        contributions.append(1.0 - signals.classifier)
    if signals.nli is not None:
        contributions.append(signals.nli)
    if signals.attribution is not None:
        contributions.append(signals.attribution)
    if not contributions:
        return _NEUTRAL
    return sum(contributions) / len(contributions)


def _load(path: Path) -> Any:
    with path.open("rb") as fh:
        return pickle.load(fh)
