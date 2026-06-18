"""Gradio entry point — thin UI over `pipeline.analyse`.

All logic lives in the `sumlens` library; this module only ingests the user's
input, runs the pipeline, and shapes the result for display.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sumlens.ingest import load_pdf, load_text
from sumlens.pipeline import analyse
from sumlens.types import AnalysisConfig, AnalysisResult, Document

_LABEL_COLORS: dict[str, str] = {
    "grounded": "green",
    "weak": "orange",
    "hallucinated": "red",
    "flagged": "#7f1d1d",  # detector-flagged token span inside a sentence (deep red)
}
# Source heatmap: support strength binned into two visible levels (+ neutral = unlabeled).
_SUPPORT_COLORS: dict[str, str] = {
    "strong support": "#f59e0b",  # entailment >= _STRONG_SUPPORT
    "support": "#fde68a",         # floor <= entailment < _STRONG_SUPPORT
}
_STRONG_SUPPORT = 0.80
_MAX_WORDS = 10_000
_MAX_PDF_BYTES = 5 * 1024 * 1024  # 5 MB
_WHY_PLACEHOLDER = "_Click a summary sentence to see why it was scored the way it was._"


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


def _merge_spans(spans: list[tuple[int, int]], length: int) -> list[tuple[int, int]]:
    """Clamp char spans to [0, length], drop empties, sort, and merge overlaps."""
    cleaned = sorted(
        (max(0, s), min(length, e)) for s, e in spans if s < e and s < length and e > 0
    )
    merged: list[tuple[int, int]] = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _split_on_spans(text: str, spans: list[tuple[int, int]]) -> list[tuple[str, bool]]:
    """Split `text` into (segment, is_flagged) pieces at the given char spans."""
    merged = _merge_spans(spans, len(text))
    if not merged:
        return [(text, False)]
    pieces: list[tuple[str, bool]] = []
    cursor = 0
    for start, end in merged:
        if start > cursor:
            pieces.append((text[cursor:start], False))
        pieces.append((text[start:end], True))
        cursor = end
    if cursor < len(text):
        pieces.append((text[cursor:], False))
    return pieces


def _summary_spans(
    result: AnalysisResult, labels: Mapping[str, str]
) -> tuple[list[tuple[str, str]], list[str]]:
    """gr.HighlightedText spans for the summary + a parallel span->sentence-id map.

    Each sentence is coloured by its (possibly re-thresholded) label; detector-flagged
    token spans within it become separate `flagged` spans so the exact fabricated
    phrase stands out. The id map lets the click handler recover which sentence a
    clicked span belongs to even when a sentence is split into several spans.
    """
    verdicts = {v.sentence_id: v for v in result.verdicts}
    spans: list[tuple[str, str]] = []
    span_ids: list[str] = []
    for sentence in result.summary.sentences:
        label = labels.get(sentence.id, "weak")
        verdict = verdicts.get(sentence.id)
        token_spans = verdict.evidence.classifier_token_spans if verdict else []
        pieces = _split_on_spans(sentence.text, token_spans)
        for k, (segment, flagged) in enumerate(pieces):
            text = segment + (" " if k == len(pieces) - 1 else "")
            spans.append((text, "flagged" if flagged else label))
            span_ids.append(sentence.id)
    return spans, span_ids


def _to_highlighted(result: AnalysisResult) -> list[tuple[str, str]]:
    """Summary spans coloured by each sentence's verdict label (the initial view)."""
    labels = {v.sentence_id: v.label for v in result.verdicts}
    return _summary_spans(result, labels)[0]


def _source_spans(
    document: Document, support: Mapping[str, float]
) -> tuple[list[tuple[str, str | None]], list[str]]:
    """gr.HighlightedText spans for the source: each sentence shaded by support strength.

    `support` maps source-sentence id -> entailment (0..1). Sentences absent from it
    get a `None` label (neutral, no highlight). Returns spans + a parallel
    span->source-id map so a clicked source sentence can be identified.
    """
    if not document.sentences:
        return [(document.raw_text, None)], []
    spans: list[tuple[str, str | None]] = []
    span_ids: list[str] = []
    for sentence in document.sentences:
        score = support.get(sentence.id)
        if score is None:
            label: str | None = None
        elif score >= _STRONG_SUPPORT:
            label = "strong support"
        else:
            label = "support"
        spans.append((sentence.text + " ", label))
        span_ids.append(sentence.id)
    return spans, span_ids


def _signal_bar(score: float | None) -> str:
    """A tiny text meter for a 0..1 signal score (None = signal did not run)."""
    if score is None:
        return "—"
    filled = round(score * 10)
    return "█" * filled + "░" * (10 - filled) + f"  {score:.2f}"


def _why_panel_md(result: AnalysisResult, sentence_id: str) -> str:
    """Explain one summary sentence: signal scores, best supporting source, failed claims."""
    verdict = next((v for v in result.verdicts if v.sentence_id == sentence_id), None)
    if verdict is None:
        return _WHY_PLACEHOLDER
    sentence = next((s for s in result.summary.sentences if s.id == sentence_id), None)
    sources = {s.id: s.text for s in result.document.sentences}
    s = verdict.signals
    lines = [
        f"**{verdict.label.upper()}** · fused score `{verdict.fused_score:.2f}`",
        f"> {sentence.text if sentence else sentence_id}",
        "",
        "| Signal | Score |",
        "| --- | --- |",
        f"| A · hallucination classifier | `{_signal_bar(s.classifier)}` |",
        f"| B · NLI entailment | `{_signal_bar(s.nli)}` |",
        f"| C · attribution (gated) | `{_signal_bar(s.attribution)}` |",
        "",
    ]
    support = verdict.evidence.source_support
    if support:
        top_id, top_score = support[0]
        lines += [
            f"**Best supporting source** — {top_score:.0%} entailment:",
            f"> {sources.get(top_id, top_id)}",
        ]
    else:
        lines.append(
            "**No supporting source found** — nothing in the document entails this sentence."
        )
    if verdict.evidence.failed_claims:
        lines += ["", "**Claims that failed verification:**"]
        lines += [f"- {c.text}" for c in verdict.evidence.failed_claims]
    return "\n".join(lines)


def _source_to_summary(result: AnalysisResult) -> dict[str, list[tuple[str, float]]]:
    """Transpose source_support: source-sentence id -> [(summary sentence id, score)] desc."""
    rev: dict[str, list[tuple[str, float]]] = {}
    for verdict in result.verdicts:
        for src_id, score in verdict.evidence.source_support:
            rev.setdefault(src_id, []).append((verdict.sentence_id, score))
    for pairs in rev.values():
        pairs.sort(key=lambda pair: pair[1], reverse=True)
    return rev


def _bidirectional_md(result: AnalysisResult, source_id: str) -> str:
    """Reverse view: which summary sentences a clicked source sentence grounds."""
    sources = {s.id: s.text for s in result.document.sentences}
    summaries = {s.id: s.text for s in result.summary.sentences}
    grounded = _source_to_summary(result).get(source_id, [])
    lines = ["**Source sentence**", f"> {sources.get(source_id, source_id)}", ""]
    if grounded:
        lines.append("Grounds these summary sentences:")
        lines += [f"- ({score:.0%}) {summaries.get(sid, sid)}" for sid, score in grounded]
    else:
        lines.append("_Does not strongly support any summary sentence._")
    return "\n".join(lines)


def _apply_tau(
    result: AnalysisResult | None,
    tau_grounded: float,
    tau_hallucinated: float,
) -> list[tuple[str, str]] | None:
    """Re-label summary sentences from stored fused scores without re-running the model."""
    if result is None:
        return None

    def _label(score: float) -> str:
        if score < tau_hallucinated:
            return "hallucinated"
        if score >= tau_grounded:
            return "grounded"
        return "weak"

    labels = {v.sentence_id: _label(v.fused_score) for v in result.verdicts}
    # Token spans are unchanged by re-thresholding, so the span->id map stays valid.
    return _summary_spans(result, labels)[0]


def _event_index(evt: Any) -> int:
    """Extract a flat int index from a gr.SelectData event (index may be int or seq)."""
    index = evt.index
    return int(index[0] if isinstance(index, (list, tuple)) else index)


def _latin1(text: str) -> str:
    """Strip characters outside Latin-1 so fpdf2 core fonts don't error."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


_PDF_COLORS: dict[str, tuple[int, int, int]] = {
    "grounded": (220, 252, 231),
    "weak": (255, 237, 213),
    "hallucinated": (254, 226, 226),
}


def _export_pdf(result: AnalysisResult | None) -> str | None:
    if result is None:
        return None
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "SumLens Analysis Report", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(
        0, 6,
        _latin1(f"Source: {result.document.source}  |  Model: {result.summary.model_name}"),
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Annotated Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    verdict_map = {v.sentence_id: v for v in result.verdicts}
    pdf.set_font("Helvetica", size=10)
    for sentence in result.summary.sentences:
        verdict = verdict_map.get(sentence.id)
        label = verdict.label if verdict else "weak"
        r, g, b = _PDF_COLORS.get(label, (240, 240, 240))
        pdf.set_fill_color(r, g, b)
        pdf.multi_cell(0, 7, _latin1(sentence.text), fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Legend", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=9)
    for lbl, (r, g, b) in _PDF_COLORS.items():
        pdf.set_fill_color(r, g, b)
        pdf.cell(6, 5, "", fill=True)
        pdf.cell(0, 5, f"  {lbl.capitalize()}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Signal Scores", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=8)
    col_w = [80, 25, 20, 22, 15, 22]
    headers = ["Sentence", "Label", "Fused", "Classifier", "NLI", "Attribution"]
    for w, h in zip(col_w, headers, strict=True):
        pdf.cell(w, 6, h, border=1)
    pdf.ln()
    for sentence in result.summary.sentences:
        v = verdict_map.get(sentence.id)
        if v is None:
            continue
        truncated = sentence.text[:45] + "..." if len(sentence.text) > 45 else sentence.text
        row = [
            _latin1(truncated),
            v.label,
            f"{v.fused_score:.2f}",
            f"{v.signals.classifier:.2f}" if v.signals.classifier is not None else "-",
            f"{v.signals.nli:.2f}" if v.signals.nli is not None else "-",
            f"{v.signals.attribution:.2f}" if v.signals.attribution is not None else "-",
        ]
        for w, cell in zip(col_w, row, strict=True):
            pdf.cell(w, 6, cell, border=1)
        pdf.ln()

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        pdf.output(tmp.name)
    finally:
        tmp.close()
    return tmp.name


def _export_json(result: AnalysisResult | None) -> str | None:
    if result is None:
        return None
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    try:
        tmp.write(result.model_dump_json(indent=2))
    finally:
        tmp.close()
    return tmp.name


def run(text: str, pdf_file: str | None) -> tuple[AnalysisResult, dict[str, Any]]:
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
    return result, result.model_dump()


def build_app() -> Any:
    import gradio as gr

    with gr.Blocks(title="SumLens", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# SumLens — Summary Faithfulness Dashboard\n"
            "Paste text or upload a PDF. SumLens summarises it and flags sentences "
            "that may be hallucinated.\n\n"
            "**Green** = grounded · **Orange** = weakly grounded · **Red** = hallucinated · "
            "**dark red** = the exact flagged phrase.  \n"
            "**Click a summary sentence** to light up its supporting source sentences and "
            "see why it scored that way. **Click a source sentence** to see which summary "
            "sentences it grounds."
        )

        result_state: gr.State = gr.State(value=None)
        summary_ids_state: gr.State = gr.State(value=[])
        source_ids_state: gr.State = gr.State(value=[])

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Source document")
                source_out = gr.HighlightedText(
                    label="Source (shaded by how strongly each sentence supports the selection)",
                    color_map=_SUPPORT_COLORS,
                    combine_adjacent=False,
                    show_legend=True,
                )

            with gr.Column():
                gr.Markdown("### Summary with faithfulness highlights")
                summary_out = gr.HighlightedText(
                    label="Summary (click a sentence to explain it and highlight its source)",
                    color_map=_LABEL_COLORS,
                    combine_adjacent=False,
                    show_legend=True,
                )

        with gr.Accordion("Why this verdict?", open=True):
            why_panel = gr.Markdown(value=_WHY_PLACEHOLDER)

        with gr.Row():
            tau_h_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.30, step=0.05,
                label="τ hallucinated — below this → hallucinated (default 0.30)",
            )
            tau_g_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.70, step=0.05,
                label="τ grounded — above this → grounded (default 0.70)",
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

        with gr.Row():
            submit = gr.Button("Analyse", variant="primary")
            json_dl = gr.DownloadButton("Export JSON", visible=False)
            pdf_dl = gr.DownloadButton("Export PDF", visible=False)

        error_box = gr.Markdown(value="", visible=False)

        with gr.Accordion("Full result (JSON viewer)", open=False):
            json_out = gr.JSON(label="AnalysisResult")

        def _handle(text: str, pdf_file: str | None) -> tuple[Any, ...]:
            try:
                result, payload = run(text, pdf_file)
                summary_spans, summary_ids = _summary_spans(
                    result, {v.sentence_id: v.label for v in result.verdicts}
                )
                source_spans, source_ids = _source_spans(result.document, {})
                json_path = _export_json(result)
                pdf_path = _export_pdf(result)
                return (
                    result,
                    summary_spans,
                    summary_ids,
                    source_spans,
                    source_ids,
                    _WHY_PLACEHOLDER,
                    payload,
                    gr.update(value=json_path, visible=True),
                    gr.update(value=pdf_path, visible=True),
                    gr.update(value="", visible=False),
                    gr.update(interactive=True),
                )
            except ValueError as exc:
                return (
                    None, [], [], [], [], _WHY_PLACEHOLDER, None,
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(value=f"**Error:** {exc}", visible=True),
                    gr.update(interactive=True),
                )

        def _on_summary_select(
            result: AnalysisResult | None, summary_ids: list[str], evt: gr.SelectData
        ) -> tuple[list[tuple[str, str | None]], str]:
            if result is None:
                return [], _WHY_PLACEHOLDER
            idx = _event_index(evt)
            if idx < 0 or idx >= len(summary_ids):
                return _source_spans(result.document, {})[0], _WHY_PLACEHOLDER
            sentence_id = summary_ids[idx]
            verdict = next((v for v in result.verdicts if v.sentence_id == sentence_id), None)
            support = dict(verdict.evidence.source_support) if verdict else {}
            return _source_spans(result.document, support)[0], _why_panel_md(result, sentence_id)

        def _on_source_select(
            result: AnalysisResult | None, source_ids: list[str], evt: gr.SelectData
        ) -> str:
            if result is None or not source_ids:
                return _WHY_PLACEHOLDER
            idx = _event_index(evt)
            if idx < 0 or idx >= len(source_ids):
                return _WHY_PLACEHOLDER
            return _bidirectional_md(result, source_ids[idx])

        # `gr.SelectData` is a string annotation under PEP 563 (from __future__ import
        # annotations) and gradio is imported inside this function, not at module level,
        # so Gradio cannot resolve the string to detect the event-data parameter and
        # passes evt=None. Inject the real class so the click events populate `evt`.
        _on_summary_select.__annotations__["evt"] = gr.SelectData
        _on_source_select.__annotations__["evt"] = gr.SelectData

        submit.click(
            fn=lambda: gr.update(interactive=False),
            inputs=[],
            outputs=[submit],
        ).then(
            fn=_handle,
            inputs=[text_in, pdf_in],
            outputs=[
                result_state, summary_out, summary_ids_state,
                source_out, source_ids_state, why_panel,
                json_out, json_dl, pdf_dl, error_box, submit,
            ],
        )

        summary_out.select(
            fn=_on_summary_select,
            inputs=[result_state, summary_ids_state],
            outputs=[source_out, why_panel],
        )

        source_out.select(
            fn=_on_source_select,
            inputs=[result_state, source_ids_state],
            outputs=[why_panel],
        )

        for slider in (tau_h_slider, tau_g_slider):
            slider.change(
                fn=_apply_tau,
                inputs=[result_state, tau_g_slider, tau_h_slider],
                outputs=[summary_out],
            )

    return demo


if __name__ == "__main__":
    build_app().launch()
