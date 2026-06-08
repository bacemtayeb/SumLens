"""Summarise tests — model mocked at the `_get_summariser` boundary (no weights)."""

import pytest

from sumlens import summarise as summarise_mod
from sumlens.summarise import _length_bounds, summarise
from sumlens.types import AnalysisConfig, Document, Sentence


def _doc() -> Document:
    return Document(
        id="doc-1",
        raw_text="A long source document about a bill that passed today.",
        sentences=[
            Sentence(id="src-0000", text="A long source document.", char_start=0, char_end=23)
        ],
        source="text",
    )


def test_summarise_builds_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_pipeline(text: str, **kwargs: object) -> list[dict[str, str]]:
        return [{"summary_text": "The bill passed. It allocates funds."}]

    monkeypatch.setattr(summarise_mod, "_get_summariser", lambda model_name: fake_pipeline)

    cfg = AnalysisConfig()
    summary = summarise(_doc(), cfg)

    assert summary.document_id == "doc-1"
    assert summary.id == "doc-1-summary"
    assert summary.model_name == cfg.summariser
    assert summary.text == "The bill passed. It allocates funds."
    assert [s.text for s in summary.sentences] == ["The bill passed.", "It allocates funds."]
    assert [s.id for s in summary.sentences] == ["sum-0000", "sum-0001"]
    for sent in summary.sentences:
        assert summary.text[sent.char_start : sent.char_end] == sent.text


def test_summarise_forwards_length_and_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_pipeline(text: str, **kwargs: object) -> list[dict[str, str]]:
        captured.update(kwargs)
        captured["text"] = text
        return [{"summary_text": "Short."}]

    monkeypatch.setattr(summarise_mod, "_get_summariser", lambda model_name: fake_pipeline)

    cfg = AnalysisConfig(summary_target_words=150)
    summarise(_doc(), cfg)

    assert captured["text"] == _doc().raw_text
    assert captured["truncation"] is True
    assert captured["max_length"] == 195
    assert captured["min_length"] == 90


def test_length_bounds() -> None:
    assert _length_bounds(150) == (195, 90)
    assert _length_bounds(100) == (130, 60)
