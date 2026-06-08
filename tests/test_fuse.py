"""Fusion tests — identity fallback, trained model path, calibration, labelling."""

import pickle
from pathlib import Path

import pytest

from sumlens.fuse import _feature_vector, calibrate, fit_fusion, fit_platt, fuse, label
from sumlens.types import AnalysisConfig, SignalScores

_UNUSED = Path("does-not-exist.pkl")


def test_fuse_all_signals() -> None:
    signals = {"sum-0000": SignalScores(classifier=0.2, nli=0.6, attribution=0.7)}
    # grounding: (1 - 0.2) + 0.6 + 0.7 = 2.1 / 3 = 0.7
    assert fuse(signals, _UNUSED)["sum-0000"] == pytest.approx(0.7)


def test_fuse_partial_signals() -> None:
    signals = {"sum-0000": SignalScores(classifier=0.1, nli=None, attribution=0.5)}
    # (1 - 0.1) + 0.5 = 1.4 / 2 = 0.7
    assert fuse(signals, _UNUSED)["sum-0000"] == pytest.approx(0.7)


def test_fuse_no_signals_is_neutral() -> None:
    signals = {"sum-0000": SignalScores(classifier=None, nli=None, attribution=None)}
    assert fuse(signals, _UNUSED)["sum-0000"] == pytest.approx(0.5)


def test_calibrate_is_passthrough() -> None:
    scores = {"sum-0000": 0.42, "sum-0001": 0.91}
    assert calibrate(scores, _UNUSED) == scores


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, "hallucinated"),
        (0.29, "hallucinated"),
        (0.30, "weak"),  # boundary: not < tau_hallucinated
        (0.50, "weak"),
        (0.69, "weak"),
        (0.70, "grounded"),  # boundary: >= tau_grounded
        (1.0, "grounded"),
    ],
)
def test_label_thresholds(score: float, expected: str) -> None:
    assert label(score, AnalysisConfig()) == expected


def test_feature_vector_imputes_missing() -> None:
    assert _feature_vector(SignalScores(classifier=0.2, nli=None, attribution=0.7)) == [
        0.2,
        0.5,
        0.7,
    ]


# Separable toy data: grounded (1) = low classifier, high nli/attribution; flipped for 0.
_X = [[0.1, 0.9, 0.9], [0.05, 0.95, 0.85], [0.9, 0.1, 0.1], [0.95, 0.2, 0.05]] * 5
_Y = [1, 1, 0, 0] * 5


def test_fuse_uses_trained_model(tmp_path: Path) -> None:
    model = fit_fusion(_X, _Y)
    path = tmp_path / "fusion.pkl"
    with path.open("wb") as fh:
        pickle.dump(model, fh)

    signals = {
        "g": SignalScores(classifier=0.1, nli=0.9, attribution=0.9),
        "h": SignalScores(classifier=0.9, nli=0.1, attribution=0.1),
    }
    out = fuse(signals, path)
    assert out["g"] > 0.5 > out["h"]  # grounded scores higher than hallucinated


def test_calibrate_uses_trained_platt(tmp_path: Path) -> None:
    platt = fit_platt([0.1, 0.2, 0.8, 0.9] * 5, [0, 0, 1, 1] * 5)
    path = tmp_path / "platt.pkl"
    with path.open("wb") as fh:
        pickle.dump(platt, fh)

    out = calibrate({"a": 0.85, "b": 0.15}, path)
    assert 0.0 <= out["a"] <= 1.0
    assert out["a"] > out["b"]
