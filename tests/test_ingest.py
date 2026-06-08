"""Ingestion tests — fixture string and fixture PDF.

Checks paragraph segmentation, sentence count, stable ids, and that every
sentence's char offsets reconstruct its text from `Document.raw_text`.
"""

from pathlib import Path

from fpdf import FPDF

from sumlens.ingest import load_pdf, load_text
from sumlens.types import Document


def _assert_offsets_reconstruct(doc: Document) -> None:
    for sent in doc.sentences:
        assert doc.raw_text[sent.char_start : sent.char_end] == sent.text


def test_load_text_segments_paragraphs_and_sentences() -> None:
    text = "The bill passed. It allocates funds.\n\nA second paragraph here."
    doc = load_text(text)

    assert doc.source == "text"
    assert doc.raw_text.split("\n\n") == [
        "The bill passed. It allocates funds.",
        "A second paragraph here.",
    ]
    assert [s.text for s in doc.sentences] == [
        "The bill passed.",
        "It allocates funds.",
        "A second paragraph here.",
    ]
    assert [s.id for s in doc.sentences] == ["src-0000", "src-0001", "src-0002"]
    assert doc.meta["word_count"] == 10
    _assert_offsets_reconstruct(doc)


def test_load_text_collapses_internal_whitespace() -> None:
    doc = load_text("A line\nwrapped   awkwardly.\n\n\n  Next para.  ")

    assert doc.raw_text == "A line wrapped awkwardly.\n\nNext para."
    _assert_offsets_reconstruct(doc)


def test_load_text_empty_has_no_sentences() -> None:
    doc = load_text("   \n\n   ")

    assert doc.raw_text == ""
    assert doc.sentences == []
    assert doc.meta["word_count"] == 0


def test_load_pdf(tmp_path: Path) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, text="The bill passed. It allocates funds.")
    path = tmp_path / "report.pdf"
    pdf.output(str(path))

    doc = load_pdf(path)

    assert doc.source == "pdf"
    assert doc.id == "report"
    assert doc.meta["filename"] == "report.pdf"
    assert [s.text for s in doc.sentences] == [
        "The bill passed.",
        "It allocates funds.",
    ]
    assert [s.id for s in doc.sentences] == ["src-0000", "src-0001"]
    _assert_offsets_reconstruct(doc)
