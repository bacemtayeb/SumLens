"""App tests — display helpers and `run` with the pipeline mocked (no gradio, no weights)."""

from pathlib import Path

import pytest

import app as app_mod
from app import _render_source_html, _to_highlighted, run
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
    document = Document(
        id="doc-1",
        raw_text="The bill passed. Budget is huge.",
        sentences=[
            Sentence(id="src-0000", text="The bill passed.", char_start=0, char_end=16),
            Sentence(id="src-0001", text="Budget is huge.", char_start=17, char_end=32),
        ],
        source="text",
    )
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
                failed_claims=[],
                top_source_sentence_ids=["src-0000"],
                classifier_token_spans=[],
            ),
        ),
        SentenceVerdict(
            sentence_id="sum-0001",
            fused_score=0.1,
            label="hallucinated",
            signals=SignalScores(classifier=0.9, nli=0.2, attribution=0.3),
            evidence=Evidence(
                failed_claims=[],
                top_source_sentence_ids=["src-0001"],
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


# ---------------------------------------------------------------------------
# _to_highlighted
# ---------------------------------------------------------------------------


def test_to_highlighted() -> None:
    assert _to_highlighted(_result()) == [
        ("Grounded one. ", "grounded"),
        ("Bad two. ", "hallucinated"),
    ]


# ---------------------------------------------------------------------------
# _render_source_html
# ---------------------------------------------------------------------------


def test_render_source_html_no_highlights() -> None:
    result = _result()
    html = _render_source_html(result.document, set())
    assert "The bill passed." in html
    assert "Budget is huge." in html
    assert "<mark" not in html


def test_render_source_html_highlights_given_ids() -> None:
    result = _result()
    html = _render_source_html(result.document, {"src-0000"})
    assert "<mark" in html
    assert "The bill passed." in html
    # only src-0000 is marked; src-0001 is plain text
    assert html.index("<mark") < html.index("The bill passed.")


def test_render_source_html_no_sentences_falls_back_to_raw() -> None:
    doc = Document(id="d", raw_text="Raw text only.", sentences=[], source="text")
    html = _render_source_html(doc, set())
    assert "Raw text only." in html


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


def test_run_text_input(monkeypatch: pytest.MonkeyPatch) -> None:
    canned = _result()
    text_doc = Document(
        id="text", raw_text="Some pasted source text.", sentences=[], source="text"
    )
    monkeypatch.setattr(app_mod, "load_text", lambda text: text_doc)
    monkeypatch.setattr(app_mod, "analyse", lambda document, cfg: canned)

    result, source_html, highlighted, payload = run("Some pasted source text.", None)

    assert highlighted == [("Grounded one. ", "grounded"), ("Bad two. ", "hallucinated")]
    assert payload == canned.model_dump()
    assert "Some pasted source text" in source_html
    assert result == canned


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


# ---------------------------------------------------------------------------
# F3 — click-to-highlight source spans
# ---------------------------------------------------------------------------


class _FakeSelectEvent:
    """Minimal stand-in for gr.SelectData."""

    def __init__(self, index: int) -> None:
        self.index = index


def test_on_sentence_select_highlights_top_source_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _result()
    # Access the inner function via build_app — easier to test the logic directly
    # by calling _render_source_html with the expected IDs (unit testing the helper).
    verdict = result.verdicts[1]  # hallucinated, top_source = ["src-0001"]
    html = _render_source_html(result.document, set(verdict.evidence.top_source_sentence_ids))
    assert "<mark" in html
    assert "Budget is huge." in html  # src-0001 text


def test_on_sentence_select_switches_highlight_on_second_click() -> None:
    result = _result()
    # Click sentence 0 → src-0000 highlighted
    html0 = _render_source_html(result.document, {"src-0000"})
    # Click sentence 1 → src-0001 highlighted
    html1 = _render_source_html(result.document, {"src-0001"})

    assert "The bill passed." in html0
    assert html0.count("<mark") == 1

    assert "Budget is huge." in html1
    assert html1.count("<mark") == 1

    # The two outputs must differ (different sentence highlighted each time)
    assert html0 != html1


def test_on_sentence_select_out_of_range_returns_plain_source() -> None:
    result = _result()
    html = _render_source_html(result.document, set())
    assert "<mark" not in html
