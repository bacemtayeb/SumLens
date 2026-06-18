"""Support attribution (signal C) tests — NLI mocked at the `_get_nli` boundary."""

import pytest

from sumlens.signals import support as support_mod
from sumlens.signals.support import support_attribution
from sumlens.types import AnalysisConfig, Document, Sentence, Summary

# Entailment lookup: (premise source sentence, hypothesis summary sentence) -> prob.
_TABLE = {
    ("Src A.", "Claim one."): 0.9,
    ("Src B.", "Claim one."): 0.2,
    ("Src C.", "Claim one."): 0.1,
}


class _FakeNLI:
    def __call__(
        self, pairs: list[dict[str, str]], top_k: object = None, batch_size: object = None
    ) -> list[list[dict[str, object]]]:
        return [
            [
                {"label": "entailment", "score": _TABLE[(p["text"], p["text_pair"])]},
                {"label": "contradiction", "score": 0.0},
            ]
            for p in pairs
        ]


def _document() -> Document:
    return Document(
        id="doc-1",
        raw_text="Src A. Src B. Src C.",
        sentences=[
            Sentence(id="src-0000", text="Src A.", char_start=0, char_end=6),
            Sentence(id="src-0001", text="Src B.", char_start=7, char_end=13),
            Sentence(id="src-0002", text="Src C.", char_start=14, char_end=20),
        ],
        source="text",
    )


def _summary() -> Summary:
    return Summary(
        id="doc-1-summary",
        document_id="doc-1",
        text="Claim one.",
        sentences=[Sentence(id="sum-0000", text="Claim one.", char_start=0, char_end=10)],
        model_name="m",
    )


def test_support_concentration_and_loo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(support_mod, "_get_nli", lambda model_name: _FakeNLI())

    result = support_attribution(_document(), _summary(), AnalysisConfig())

    conc, loo, support = result["sum-0000"]
    # row = [0.9, 0.2, 0.1]: top1=0.9, top2=0.2, mean=0.4
    assert conc == pytest.approx(0.9 - 0.4)  # peak minus mean
    assert loo == pytest.approx(0.9 - 0.2)  # best-supporter margin
    # only src-0000 (0.9) clears the 0.5 floor; B (0.2) and C (0.1) are filtered out
    assert len(support) == 1
    assert support[0][0] == "src-0000"
    assert support[0][1] == pytest.approx(0.9)


def test_support_below_floor_returns_no_supporters(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fabricated sentence (nothing in the source entails it) surfaces no supporters."""

    class _FlatLowNLI:
        def __call__(
            self, pairs: list[dict[str, str]], top_k: object = None, batch_size: object = None
        ) -> list[list[dict[str, object]]]:
            return [[{"label": "entailment", "score": 0.2}] for _ in pairs]

    monkeypatch.setattr(support_mod, "_get_nli", lambda model_name: _FlatLowNLI())
    conc, loo, support = support_attribution(_document(), _summary(), AnalysisConfig())["sum-0000"]
    assert support == []  # nothing clears the floor → empty heatmap


def test_support_empty_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(support_mod, "_get_nli", lambda model_name: _FakeNLI())
    empty_doc = Document(id="d", raw_text="", sentences=[], source="text")
    result = support_attribution(empty_doc, _summary(), AnalysisConfig())
    assert result == {"sum-0000": (0.0, 0.0, [])}
