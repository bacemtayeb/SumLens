"""App tests — display helper and `run` with the pipeline mocked (no gradio, no weights)."""

from pathlib import Path

import pytest

import app as app_mod
from app import _to_highlighted, run
from sumlens.types import (
    AnalysisConfig,
    AnalysisResult,
    Document,
    Evidence,
    Sentence,
    SentenceVerdict,
    SignalScores,
    Summary,
)


def _result() -> AnalysisResult:
    document = Document(id="doc-1", raw_text="Source.", sentences=[], source="text")
    summary = Summary(
        id="doc-1-summary",
        document_id="doc-1",
        text="Grounded one. Bad two.",
        sentences=[
            Sentence(id="sum-0000", text="Grounded one.", char_start=0, char_end=13),
            Sentence(id="sum-0001", text="Bad two.", char_start=14, char_end=22),
        ],
        model_name="m",
    )
    verdicts = [
        SentenceVerdict(
            sentence_id="sum-0000",
            fused_score=0.9,
            label="grounded",
            signals=SignalScores(classifier=0.1, nli=0.9, attribution=None),
            evidence=Evidence(
                failed_claims=[], top_source_sentence_ids=[], classifier_token_spans=[]
            ),
        ),
        SentenceVerdict(
            sentence_id="sum-0001",
            fused_score=0.1,
            label="hallucinated",
            signals=SignalScores(classifier=0.9, nli=0.2, attribution=0.3),
            evidence=Evidence(
                failed_claims=[],
                top_source_sentence_ids=["src-0000"],
                classifier_token_spans=[],
            ),
        ),
    ]
    return AnalysisResult(
        document=document,
        summary=summary,
        verdicts=verdicts,
        config=AnalysisConfig(),
        timings_ms={},
    )


def test_to_highlighted() -> None:
    assert _to_highlighted(_result()) == [
        ("Grounded one. ", "grounded"),
        ("Bad two. ", "hallucinated"),
    ]


def test_run_text_input(monkeypatch: pytest.MonkeyPatch) -> None:
    canned = _result()
    text_doc = Document(id="text", raw_text="Some pasted source text.", sentences=[], source="text")
    monkeypatch.setattr(app_mod, "load_text", lambda text: text_doc)
    monkeypatch.setattr(app_mod, "analyse", lambda document, cfg: canned)

    source, highlighted, payload = run("Some pasted source text.", None)

    assert highlighted == [("Grounded one. ", "grounded"), ("Bad two. ", "hallucinated")]
    assert payload == canned.model_dump()
    assert source == "Some pasted source text."


def test_run_prefers_pdf_when_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_pdf = tmp_path / "report.pdf"
    fake_pdf.write_bytes(b"%PDF-1.0")

    seen: dict[str, str] = {}
    pdf_doc = Document(id="doc-1", raw_text="Source.", sentences=[], source="pdf")

    def _fake_load_pdf(path: Path) -> Document:
        seen["path"] = str(path)
        return pdf_doc

    monkeypatch.setattr(app_mod, "load_pdf", _fake_load_pdf)
    monkeypatch.setattr(app_mod, "analyse", lambda document, cfg: _result())

    run("ignored text", str(fake_pdf))

    assert "report.pdf" in seen["path"]


def test_run_rejects_empty_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_mod, "analyse", lambda document, cfg: _result())
    with pytest.raises(ValueError, match="empty"):
        run("", None)


def test_run_rejects_oversized_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_mod, "analyse", lambda document, cfg: _result())
    big_text = " ".join(["word"] * 10_001)
    with pytest.raises(ValueError, match="too long"):
        run(big_text, None)


def test_run_rejects_oversized_pdf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    big_pdf = tmp_path / "big.pdf"
    big_pdf.write_bytes(b"x" * (5 * 1024 * 1024 + 1))
    monkeypatch.setattr(app_mod, "load_pdf", lambda path: _result().document)
    monkeypatch.setattr(app_mod, "analyse", lambda document, cfg: _result())
    with pytest.raises(ValueError, match="too large"):
        run("", str(big_pdf))
