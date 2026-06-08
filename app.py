"""Gradio entry point — thin UI over `pipeline.analyse`.

All logic lives in the `sumlens` library; this module only ingests the user's
input, runs the pipeline, and shapes the result for display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sumlens.ingest import load_pdf, load_text
from sumlens.pipeline import analyse
from sumlens.types import AnalysisConfig, AnalysisResult

_LABEL_COLORS: dict[str, str] = {
    "grounded": "green",
    "weak": "orange",
    "hallucinated": "red",
}
_MAX_WORDS = 10_000
_MAX_PDF_BYTES = 5 * 1024 * 1024  # 5 MB


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


def run(
    text: str,
    pdf_file: str | None,
) -> tuple[str, list[tuple[str, str]], dict[str, Any]]:
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
    return document.raw_text, _to_highlighted(result), result.model_dump()


def build_app() -> Any:
    import gradio as gr

    with gr.Blocks(title="SumLens", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# SumLens — Summary Faithfulness Dashboard\n"
            "Paste text or upload a PDF. SumLens summarises it and flags sentences "
            "that may be hallucinated.\n\n"
            "**Green** = grounded · **Orange** = weakly grounded · **Red** = hallucinated"
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Source document")
                text_in = gr.Textbox(
                    label="Paste text",
                    lines=14,
                    placeholder="Paste your document here…",
                )
                pdf_in = gr.File(
                    label="or upload PDF (≤ 5 MB)",
                    file_types=[".pdf"],
                    type="filepath",
                )

            with gr.Column():
                gr.Markdown("### Summary with faithfulness highlights")
                summary_out = gr.HighlightedText(
                    label="Summary",
                    color_map=_LABEL_COLORS,
                    combine_adjacent=False,
                    show_legend=True,
                )

        submit = gr.Button("Analyse", variant="primary")
        error_box = gr.Markdown(value="", visible=False)

        with gr.Accordion("Ingested source text", open=False):
            source_out = gr.Textbox(label="Processed source", lines=8, interactive=False)

        with gr.Accordion("Full result (JSON export)", open=False):
            json_out = gr.JSON(label="AnalysisResult")

        def _handle(text: str, pdf_file: str | None) -> tuple[Any, Any, Any, Any, Any]:
            try:
                source, highlighted, payload = run(text, pdf_file)
                return (
                    highlighted,
                    source,
                    payload,
                    gr.update(value="", visible=False),
                    gr.update(interactive=True),
                )
            except ValueError as exc:
                return (
                    None,
                    "",
                    None,
                    gr.update(value=f"**Error:** {exc}", visible=True),
                    gr.update(interactive=True),
                )

        submit.click(
            fn=lambda: gr.update(interactive=False),
            inputs=[],
            outputs=[submit],
        ).then(
            fn=_handle,
            inputs=[text_in, pdf_in],
            outputs=[summary_out, source_out, json_out, error_box, submit],
        )

    return demo


if __name__ == "__main__":
    build_app().launch()
