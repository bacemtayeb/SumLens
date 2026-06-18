"""App tests — display helpers and `run` with the pipeline mocked (no gradio, no weights)."""

import json
from pathlib import Path

import pytest

import app as app_mod
from app import (
    _apply_tau,
    _bidirectional_md,
    _export_json,
    _export_pdf,
    _source_spans,
    _split_on_spans,
    _summary_spans,
    _to_highlighted,
    _why_panel_md,
    run,
)
from sumlens.types import (
    AnalysisConfig,
    AnalysisResult,
    Claim,
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
                source_support=[("src-0000", 0.92)],
                classifier_token_spans=[],
            ),
        ),
        SentenceVerdict(
            sentence_id="sum-0001",
            fused_score=0.1,
            label="hallucinated",
            signals=SignalScores(classifier=0.9, nli=0.2, attribution=0.3),
            evidence=Evidence(
                failed_claims=[
                    Claim(id="sum-0001-claim-1", sentence_id="sum-0001", text="Bad two.")
                ],
                top_source_sentence_ids=[],
                source_support=[],
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
# _to_highlighted / _summary_spans
# ---------------------------------------------------------------------------


def test_to_highlighted() -> None:
    assert _to_highlighted(_result()) == [
        ("Grounded one. ", "grounded"),
        ("Bad two. ", "hallucinated"),
    ]


def test_summary_spans_id_map_one_per_sentence() -> None:
    result = _result()
    labels = {v.sentence_id: v.label for v in result.verdicts}
    spans, ids = _summary_spans(result, labels)
    assert ids == ["sum-0000", "sum-0001"]
    assert spans == [("Grounded one. ", "grounded"), ("Bad two. ", "hallucinated")]


def test_summary_spans_splits_flagged_token_span() -> None:
    result = _result()
    result.verdicts[1].evidence.classifier_token_spans = [(0, 3)]  # "Bad" in "Bad two."
    labels = {v.sentence_id: v.label for v in result.verdicts}
    spans, ids = _summary_spans(result, labels)
    assert ("Bad", "flagged") in spans
    # the flagged sub-span and its remainder both map back to the same sentence
    assert ids.count("sum-0001") == 2
    assert ids.count("sum-0000") == 1


# ---------------------------------------------------------------------------
# _split_on_spans
# ---------------------------------------------------------------------------


def test_split_on_spans_no_spans_returns_whole() -> None:
    assert _split_on_spans("hello world", []) == [("hello world", False)]


def test_split_on_spans_marks_subspan() -> None:
    assert _split_on_spans("Bad two.", [(0, 3)]) == [("Bad", True), (" two.", False)]


def test_split_on_spans_merges_overlap_and_clamps() -> None:
    assert _split_on_spans("abcdef", [(1, 3), (2, 4), (10, 20)]) == [
        ("a", False),
        ("bcd", True),
        ("ef", False),
    ]


# ---------------------------------------------------------------------------
# _source_spans — graded heatmap
# ---------------------------------------------------------------------------


def test_source_spans_neutral_when_no_support() -> None:
    spans, ids = _source_spans(_result().document, {})
    assert ids == ["src-0000", "src-0001"]
    assert all(label is None for _, label in spans)


def test_source_spans_shades_by_strength() -> None:
    spans, ids = _source_spans(_result().document, {"src-0000": 0.95, "src-0001": 0.6})
    labels = {sid: label for (_, label), sid in zip(spans, ids, strict=True)}
    assert labels["src-0000"] == "strong support"  # >= 0.80
    assert labels["src-0001"] == "support"          # floor .. 0.80


def test_source_spans_falls_back_to_raw_when_no_sentences() -> None:
    doc = Document(id="d", raw_text="Raw text only.", sentences=[], source="text")
    spans, ids = _source_spans(doc, {})
    assert spans == [("Raw text only.", None)]
    assert ids == []


# ---------------------------------------------------------------------------
# _why_panel_md — explanation + contrastive evidence
# ---------------------------------------------------------------------------


def test_why_panel_grounded_shows_best_source() -> None:
    md = _why_panel_md(_result(), "sum-0000")
    assert "GROUNDED" in md
    assert "Best supporting source" in md
    assert "The bill passed." in md


def test_why_panel_hallucinated_shows_no_support_and_failed_claim() -> None:
    md = _why_panel_md(_result(), "sum-0001")
    assert "HALLUCINATED" in md
    assert "No supporting source found" in md
    assert "Bad two." in md  # the failed claim text


# ---------------------------------------------------------------------------
# _bidirectional_md — source → summary links
# ---------------------------------------------------------------------------


def test_bidirectional_lists_grounded_summary_sentences() -> None:
    md = _bidirectional_md(_result(), "src-0000")
    assert "Source sentence" in md
    assert "The bill passed." in md
    assert "Grounded one." in md  # src-0000 grounds sum-0000


def test_bidirectional_no_links() -> None:
    md = _bidirectional_md(_result(), "src-0001")  # grounds nothing
    assert "does not strongly support" in md.lower()


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

    result, payload = run("Some pasted source text.", None)

    assert result == canned
    assert payload == canned.model_dump()


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
# F4 — adjustable threshold τ (re-labels without re-running the model)
# ---------------------------------------------------------------------------


def test_apply_tau_returns_none_without_result() -> None:
    assert _apply_tau(None, 0.70, 0.30) is None


def test_apply_tau_mirrors_fuse_label_logic() -> None:
    spans = _apply_tau(_result(), 0.70, 0.30)
    assert spans is not None
    assert spans[0] == ("Grounded one. ", "grounded")
    assert spans[1] == ("Bad two. ", "hallucinated")


def test_apply_tau_relabels_without_model_rerun() -> None:
    spans = _apply_tau(_result(), 0.95, 0.30)
    assert spans is not None
    assert spans[0][1] == "weak"           # fused 0.9 now below tau_grounded 0.95
    assert spans[1][1] == "hallucinated"   # unchanged


def test_apply_tau_boundary_score_lt_tau_h_is_hallucinated() -> None:
    spans = _apply_tau(_result(), 0.70, 0.15)
    assert spans is not None
    assert spans[1][1] == "hallucinated"


def test_apply_tau_boundary_score_gte_tau_g_is_grounded() -> None:
    spans = _apply_tau(_result(), 0.85, 0.30)
    assert spans is not None
    assert spans[0][1] == "grounded"


# ---------------------------------------------------------------------------
# F5 — export JSON
# ---------------------------------------------------------------------------


def test_export_json_returns_none_without_result() -> None:
    assert _export_json(None) is None


def test_export_json_creates_valid_file() -> None:
    result = _result()
    path = _export_json(result)
    assert path is not None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert AnalysisResult.model_validate(data) == result


def test_export_json_schema_has_required_fields() -> None:
    path = _export_json(_result())
    assert path is not None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert {"document", "summary", "verdicts", "config"} <= data.keys()
    verdict = data["verdicts"][0]
    assert {"sentence_id", "fused_score", "label", "signals", "evidence"} <= verdict.keys()


# ---------------------------------------------------------------------------
# F6 — export PDF
# ---------------------------------------------------------------------------


def test_export_pdf_returns_none_without_result() -> None:
    assert _export_pdf(None) is None


def test_export_pdf_creates_pdf_file() -> None:
    path = _export_pdf(_result())
    assert path is not None
    content = Path(path).read_bytes()
    assert content.startswith(b"%PDF"), "file must be a valid PDF"


def test_export_pdf_contains_summary_text() -> None:
    import pdfplumber

    result = _result()
    path = _export_pdf(result)
    assert path is not None
    with pdfplumber.open(path) as pdf:
        text = " ".join(page.extract_text() or "" for page in pdf.pages)
    for sentence in result.summary.sentences:
        assert sentence.text[:10] in text
