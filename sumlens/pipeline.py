"""Pipeline orchestration — Document into a full AnalysisResult.

Stages: summarise -> signal A (classifier) and signal B (NLI) -> gate signal C
(attribution) on the sentences A or B flag as suspicious -> fuse -> calibrate ->
label -> assemble evidence. Each stage is timed into `timings_ms`.

A and B run sequentially here; the data-model calls for them "in parallel", which
is a latency optimisation, not a correctness requirement, so it is deferred.
"""

import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from sumlens.fuse import calibrate, fuse, label
from sumlens.signals.attribution import attribute
from sumlens.signals.classifier import classify
from sumlens.signals.nli import entail, extract_claims
from sumlens.summarise import summarise
from sumlens.types import (
    AnalysisConfig,
    AnalysisResult,
    Claim,
    Document,
    Evidence,
    SentenceVerdict,
    SignalScores,
    Summary,
)

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_FUSION_MODEL_PATH = _MODELS_DIR / "fusion.pkl"
_PLATT_MODEL_PATH = _MODELS_DIR / "platt.pkl"
_C_GATE = 0.5  # run attribution where classifier >= gate (A high) or nli < gate (B low)

T = TypeVar("T")


def analyse(document: Document, cfg: AnalysisConfig) -> AnalysisResult:
    timings: dict[str, int] = {}

    summary = _timed(timings, "summarise", lambda: summarise(document, cfg))
    classifier_out = _timed(timings, "classify", lambda: classify(document, summary, cfg))
    nli_out = _timed(timings, "nli", lambda: entail(extract_claims(summary), document, cfg))

    gated = _gated_summary(summary, classifier_out, nli_out)
    attribution_out = _timed(timings, "attribute", lambda: attribute(document, gated, cfg))

    signals: dict[str, SignalScores] = {}
    evidence_parts: dict[str, tuple[list[tuple[int, int]], list[Claim], list[str]]] = {}
    for sentence in summary.sentences:
        a_score, a_spans = classifier_out[sentence.id]
        b_score, b_failed = nli_out.get(sentence.id, (None, []))
        c_peak, c_top = attribution_out.get(sentence.id, (None, []))
        signals[sentence.id] = SignalScores(classifier=a_score, nli=b_score, attribution=c_peak)
        evidence_parts[sentence.id] = (a_spans, b_failed, c_top)

    fused = _timed(
        timings,
        "fuse",
        lambda: calibrate(fuse(signals, _FUSION_MODEL_PATH), _PLATT_MODEL_PATH),
    )

    verdicts = []
    for sentence in summary.sentences:
        a_spans, b_failed, c_top = evidence_parts[sentence.id]
        score = fused[sentence.id]
        verdicts.append(
            SentenceVerdict(
                sentence_id=sentence.id,
                fused_score=score,
                label=label(score, cfg),
                signals=signals[sentence.id],
                evidence=Evidence(
                    failed_claims=b_failed,
                    top_source_sentence_ids=c_top,
                    classifier_token_spans=a_spans,
                ),
            )
        )

    return AnalysisResult(
        document=document,
        summary=summary,
        verdicts=verdicts,
        config=cfg,
        timings_ms=timings,
    )


def _gated_summary(
    summary: Summary,
    classifier_out: dict[str, tuple[float, list[tuple[int, int]]]],
    nli_out: dict[str, tuple[float, list[Claim]]],
) -> Summary:
    """Keep only sentences A or B flag as suspicious — attribution runs on these."""
    suspicious = []
    for sentence in summary.sentences:
        a_score = classifier_out[sentence.id][0]
        b_score = nli_out.get(sentence.id, (None, []))[0]
        if a_score >= _C_GATE or (b_score is not None and b_score < _C_GATE):
            suspicious.append(sentence)
    return summary.model_copy(update={"sentences": suspicious})


def _timed(timings: dict[str, int], name: str, fn: Callable[[], T]) -> T:
    start = time.perf_counter()
    result = fn()
    timings[name] = int((time.perf_counter() - start) * 1000)
    return result
