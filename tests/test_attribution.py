"""Attribution (signal C) tests — Inseq mocked at the `_source_token_attributions` boundary."""

import pytest

from sumlens.signals import attribution as attribution_mod
from sumlens.signals.attribution import _aggregate_to_source_sentences, attribute
from sumlens.types import AnalysisConfig, Document, Sentence, Summary

# raw_text = "Alpha beta. Gamma delta. Epsilon zeta."
#             0          12           25
_SOURCE = [
    Sentence(id="src-0000", text="Alpha beta.", char_start=0, char_end=11),
    Sentence(id="src-0001", text="Gamma delta.", char_start=12, char_end=24),
    Sentence(id="src-0002", text="Epsilon zeta.", char_start=25, char_end=38),
]

# (char_start, char_end, score) source-token records per target sentence text.
# Negative score on src-0001 exercises the abs-mass aggregation.
_ATTRS: dict[str, list[tuple[int, int, float]]] = {
    "Heavily grounded.": [(0, 5, 0.2), (12, 17, -0.6), (25, 32, 0.2)],
    "No support.": [],
}


def _document() -> Document:
    return Document(
        id="doc-1",
        raw_text="Alpha beta. Gamma delta. Epsilon zeta.",
        sentences=_SOURCE,
        source="text",
    )


def _summary() -> Summary:
    return Summary(
        id="doc-1-summary",
        document_id="doc-1",
        text="Heavily grounded. No support.",
        sentences=[
            Sentence(id="sum-0000", text="Heavily grounded.", char_start=0, char_end=17),
            Sentence(id="sum-0001", text="No support.", char_start=18, char_end=29),
        ],
        model_name="facebook/bart-large-cnn",
    )


def test_attribute(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        attribution_mod,
        "_source_token_attributions",
        lambda source_text, target_text, cfg: _ATTRS[target_text],
    )

    result = attribute(_document(), _summary(), AnalysisConfig())

    peak, top_ids = result["sum-0000"]
    # abs masses 0.2 / 0.6 / 0.2 -> normalised 0.2 / 0.6 / 0.2; peak = 0.6
    assert peak == pytest.approx(0.6)
    assert top_ids == ["src-0001", "src-0000", "src-0002"]

    assert result["sum-0001"] == (0.0, [])


def test_aggregate_normalises_to_one() -> None:
    masses = _aggregate_to_source_sentences([(0, 5, 0.2), (12, 17, -0.6), (25, 32, 0.2)], _SOURCE)
    assert sum(masses.values()) == pytest.approx(1.0)
    assert masses == {
        "src-0000": pytest.approx(0.2),
        "src-0001": pytest.approx(0.6),
        "src-0002": pytest.approx(0.2),
    }


def test_aggregate_no_tokens() -> None:
    assert _aggregate_to_source_sentences([], _SOURCE) == {}
