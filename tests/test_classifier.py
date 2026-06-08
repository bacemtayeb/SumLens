"""Classifier (signal A) tests — LettuceDetect mocked at the `_get_detector` boundary."""

import pytest

from sumlens.signals import classifier as classifier_mod
from sumlens.signals.classifier import _aggregate, classify
from sumlens.types import AnalysisConfig, Document, Sentence, Summary

# LettuceDetect "tokens" output: {token, pred, prob} — no char offsets.
_GROUNDED_TOKENS: list[dict[str, object]] = [
    {"token": "a", "pred": 0, "prob": 0.05},
    {"token": "b", "pred": 0, "prob": 0.02},
]
_HALLUCINATED_TOKENS: list[dict[str, object]] = [
    {"token": "a", "pred": 1, "prob": 0.91},
    {"token": "b", "pred": 1, "prob": 0.84},
    {"token": "c", "pred": 0, "prob": 0.10},
]
# "spans" output: {start, end, confidence, text} — the char offsets.
_HALLUCINATED_SPANS: list[dict[str, object]] = [
    {"start": 0, "end": 4, "confidence": 0.9, "text": "Inve"},
    {"start": 5, "end": 9, "confidence": 0.8, "text": "nted"},
]


class _FakeDetector:
    def predict(
        self, *, context: list[str], question: str, answer: str, output_format: str
    ) -> list[dict[str, object]]:
        grounded = answer == "Grounded claim here."
        if output_format == "spans":
            return [] if grounded else _HALLUCINATED_SPANS
        return _GROUNDED_TOKENS if grounded else _HALLUCINATED_TOKENS


def _summary() -> Summary:
    return Summary(
        id="doc-1-summary",
        document_id="doc-1",
        text="Grounded claim here. Invented figure cited.",
        sentences=[
            Sentence(id="sum-0000", text="Grounded claim here.", char_start=0, char_end=20),
            Sentence(id="sum-0001", text="Invented figure cited.", char_start=21, char_end=43),
        ],
        model_name="facebook/bart-large-cnn",
    )


def _document() -> Document:
    return Document(id="doc-1", raw_text="Some source text.", sentences=[], source="text")


def test_classify(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(classifier_mod, "_get_detector", lambda model_path: _FakeDetector())

    result = classify(_document(), _summary(), AnalysisConfig())

    assert set(result) == {"sum-0000", "sum-0001"}

    grounded_score, grounded_spans = result["sum-0000"]
    assert grounded_score == pytest.approx((0.05 + 0.02) / 2)
    assert grounded_spans == []

    halluc_score, halluc_spans = result["sum-0001"]
    assert halluc_score == pytest.approx((0.91 + 0.84 + 0.10) / 3)
    assert halluc_spans == [(0, 4), (5, 9)]


def test_classify_empty_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(classifier_mod, "_get_detector", lambda model_path: _FakeDetector())
    summary = Summary(
        id="s", document_id="doc-1", text="", sentences=[], model_name="m"
    )
    assert classify(_document(), summary, AnalysisConfig()) == {}


def test_aggregate_uses_top_k() -> None:
    tokens = [{"prob": p} for p in (0.9, 0.8, 0.7, 0.6, 0.1)]
    assert _aggregate(tokens) == pytest.approx((0.9 + 0.8 + 0.7) / 3)


def test_aggregate_empty() -> None:
    assert _aggregate([]) == 0.0
