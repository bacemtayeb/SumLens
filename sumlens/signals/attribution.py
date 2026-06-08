"""Signal C — Inseq integrated-gradients source attribution.

Integrated gradients on the *same* model that produced the summary. For each
summary sentence we get token-level attribution back onto the source text, map
each source token to the source sentence containing it, sum the (absolute)
attribution mass per source sentence and normalise. The peak normalised mass is
the sentence's attribution score; the top-k source sentences are its supporting
spans.

The pipeline gates this signal (runs it only on sentences flagged by A or B) for
speed — see `data-model.md` §3. The Inseq computation is isolated in
`_source_token_attributions`, which tests mock at the module boundary.
"""

import re
from functools import lru_cache
from typing import Any

from sumlens.types import AnalysisConfig, Document, Sentence, Summary

_N_STEPS = 40
_TOP_K = 5
_SUBWORD_PREFIX = re.compile(r"^[Ġ▁]")  # GPT-2 'Ġ' and SentencePiece '▁'


def attribute(
    document: Document, summary: Summary, cfg: AnalysisConfig
) -> dict[str, tuple[float, list[str]]]:
    results: dict[str, tuple[float, list[str]]] = {}
    for sentence in summary.sentences:
        token_attrs = _source_token_attributions(document.raw_text, sentence.text, cfg)
        per_source = _aggregate_to_source_sentences(token_attrs, document.sentences)
        if per_source:
            peak = max(per_source.values())
            top_ids = sorted(per_source, key=lambda sid: per_source[sid], reverse=True)[:_TOP_K]
        else:
            peak, top_ids = 0.0, []
        results[sentence.id] = (peak, top_ids)
    return results


def _aggregate_to_source_sentences(
    token_attrs: list[tuple[int, int, float]], source_sentences: list[Sentence]
) -> dict[str, float]:
    """Sum |score| of each source token into its sentence; normalise to sum 1."""
    masses: dict[str, float] = {}
    for start, end, score in token_attrs:
        mid = (start + end) // 2
        for sentence in source_sentences:
            if sentence.char_start <= mid < sentence.char_end:
                masses[sentence.id] = masses.get(sentence.id, 0.0) + abs(score)
                break
    total = sum(masses.values())
    if total <= 0:
        return {}
    return {sid: mass / total for sid, mass in masses.items()}


def _source_token_attributions(
    source_text: str, target_text: str, cfg: AnalysisConfig
) -> list[tuple[int, int, float]]:
    """Run Inseq and return source-token (char_start, char_end, score) records.

    This is the Inseq boundary — mocked in tests, verified against real weights on
    HPC. Token char offsets are reconstructed by walking `source_text` because
    Inseq tokens carry subword strings, not source offsets.
    """
    import numpy as np

    model = _get_attributor(cfg.summariser, cfg.attribution_method)
    out = model.attribute(source_text, target_text, n_steps=_N_STEPS, show_progress=False)
    seq = out.sequence_attributions[0]
    matrix = np.asarray(seq.source_attributions, dtype=float)
    per_token = matrix.sum(axis=tuple(range(1, matrix.ndim))) if matrix.ndim > 1 else matrix

    records: list[tuple[int, int, float]] = []
    cursor = 0
    for token, score in zip(seq.source, per_token, strict=False):
        text = _SUBWORD_PREFIX.sub("", token.token)
        if not text:
            continue
        idx = source_text.find(text, cursor)
        if idx < 0:
            continue
        records.append((idx, idx + len(text), float(score)))
        cursor = idx + len(text)
    return records


@lru_cache(maxsize=1)
def _get_attributor(model_name: str, method: str) -> Any:
    import inseq

    return inseq.load_model(model_name, method)
