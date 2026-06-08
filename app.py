"""Gradio entry point — thin UI over `pipeline.analyse`.

All logic lives in the `sumlens` library; this module only ingests the user's
input, runs the pipeline, and shapes the result for display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sumlens.ingest import load_pdf, load_text
from sumlens.pipeline import analyse
from sumlens.types import AnalysisConfig, AnalysisResult, Document

_LABEL_COLORS: dict[str, str] = {
    "grounded": "green",
    "weak": "orange",
    "hallucinated": "red",
}
_MAX_WORDS = 10_000
_MAX_PDF_BYTES = 5 * 1024 * 1024  # 5 MB
_SOURCE_PLACEHOLDER = "<p><em>Load a document to see the source text here.</em></p>"


def _validate_text(text: str) -> str:
    text = text.strip()
    if not text:
        raise ValueError("Input is empty. Please paste some text or upload a PDF.")
    word_count = len(text.split())
    if word_count > _MAX_WORDS:
        raise ValueError(
            f"Input is too long ({word_count:,} words). Maximum is {_MAX_WORDS:,} words."
        )
    return text


def _to_highlighted(result: AnalysisResult) -> list[tuple[str, str]]:
    """Summary sentences as (text, label) spans for gr.HighlightedText colour bands."""
    labels = {v.sentence_id: v.label for v in result.verdicts}
    return [(f"{s.text} ", labels.get(s.id, "weak")) for s in result.summary.sentences]


def _render_source_html(document: Document, highlighted_ids: set[str]) -> str:
    """Source sentences as HTML; sentences in highlighted_ids get a yellow mark."""
    if not document.sentences:
        return f"<p style='line-height:1.8'>{document.raw_text}</p>"
    parts = []
    for sentence in document.sentences:
        if sentence.id in highlighted_ids:
            parts.append(
                f"<mark style='background:#fde68a;border-radius:3px;"
                f"padding:1px 3px'>{sentence.text}</mark>"
            )
        else:
            parts.append(sentence.text)
    return "<p style='line-height:1.8'>" + " ".join(parts) + "</p>"


def run(
    text: str,
    pdf_file: str | None,
) -> tuple[AnalysisResult, str, list[tuple[str, str]], dict[str, Any]]:
    if pdf_file:
        path = Path(pdf_file)
        if path.stat().st_size > _MAX_PDF_BYTES:
            raise ValueError(
                f"PDF is too large ({path.stat().st_size / 1_048_576:.1f} MB). "
                "Maximum is 5 MB."
            )
        document = load_pdf(path)
    else:
        document = load_text(_validate_text(text))

    result = analyse(document, AnalysisConfig())
    source_html = _render_source_html(document, set())
    return result, source_html, _to_highlighted(result), result.model_dump()


def build_app() -> Any:
    import gradio as gr

    with gr.Blocks(title="SumLens", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# SumLens — Summary Faithfulness Dashboard\n"
            "Paste text or upload a PDF. SumLens summarises it and flags sentences "
            "that may be hallucinated.\n\n"
            "**Green** = grounded · **Orange** = weakly grounded · **Red** = hallucinated  \n"
            "Click a summary sentence to highlight its attributed source spans."
        )

        result_state: gr.State = gr.State(value=None)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Source document")
                source_html_out = gr.HTML(value=_SOURCE_PLACEHOLDER)

            with gr.Column():
                gr.Markdown("### Summary with faithfulness highlights")
                summary_out = gr.HighlightedText(
                    label="Summary (click a sentence to highlight source spans)",
                    color_map=_LABEL_COLORS,
                    combine_adjacent=False,
                    show_legend=True,
                )

        with gr.Row():
            text_in = gr.Textbox(
                label="Paste text",
                lines=6,
                placeholder="Paste your document here…",
            )
            pdf_in = gr.File(
                label="or upload PDF (≤ 5 MB)",
                file_types=[".pdf"],
                type="filepath",
            )

        submit = gr.Button("Analyse", variant="primary")
        error_box = gr.Markdown(value="", visible=False)

        with gr.Accordion("Full result (JSON export)", open=False):
            json_out = gr.JSON(label="AnalysisResult")

        def _handle(
            text: str, pdf_file: str | None
        ) -> tuple[Any, Any, Any, Any, Any, Any]:
            try:
                result, source_html, highlighted, payload = run(text, pdf_file)
                return (
                    result,
                    source_html,
                    highlighted,
                    payload,
                    gr.update(value="", visible=False),
                    gr.update(interactive=True),
                )
            except ValueError as exc:
                return (
                    None,
                    _SOURCE_PLACEHOLDER,
                    None,
                    None,
                    gr.update(value=f"**Error:** {exc}", visible=True),
                    gr.update(interactive=True),
                )

        def _on_sentence_select(
            evt: Any, result: AnalysisResult | None
        ) -> str:
            if result is None:
                return _SOURCE_PLACEHOLDER
            idx: int = int(evt.index)
            sentences = result.summary.sentences
            if idx >= len(sentences):
                return _render_source_html(result.document, set())
            sentence_id = sentences[idx].id
            verdict = next(
                (v for v in result.verdicts if v.sentence_id == sentence_id), None
            )
            highlighted = (
                set(verdict.evidence.top_source_sentence_ids) if verdict else set()
            )
            return _render_source_html(result.document, highlighted)

        submit.click(
            fn=lambda: gr.update(interactive=False),
            inputs=[],
            outputs=[submit],
        ).then(
            fn=_handle,
            inputs=[text_in, pdf_in],
            outputs=[result_state, source_html_out, summary_out, json_out, error_box, submit],
        )

        summary_out.select(
            fn=_on_sentence_select,
            inputs=[result_state],
            outputs=[source_html_out],
        )

    return demo


if __name__ == "__main__":
    build_app().launch()
