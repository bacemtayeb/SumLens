"""Signal A — LettuceDetect hallucination classifier wrapper.

Thin wrapper around `lettucedetect`. For each summary sentence we run the detector
with the source document as context and the sentence as the answer. The score is
the mean of the top-k per-token hallucination probabilities (`output_format=
"tokens"` gives `{token, pred, prob}`); the char spans come from `output_format=
"spans"` (`{start, end, confidence, text}`), which are offsets within the summary
sentence. Two calls because token output carries no char offsets (verified against
real weights).
"""

from functools import lru_cache
from typing import Any

from sumlens.types import AnalysisConfig, Document, Summary

_TOP_K = 3


def classify(
    document: Document, summary: Summary, cfg: AnalysisConfig
) -> dict[str, tuple[float, list[tuple[int, int]]]]:
    detector = _get_detector(cfg.classifier_model)
    results: dict[str, tuple[float, list[tuple[int, int]]]] = {}
    for sentence in summary.sentences:
        tokens = detector.predict(
            context=[document.raw_text],
            question="",
            answer=sentence.text,
            output_format="tokens",
        )
        spans = detector.predict(
            context=[document.raw_text],
            question="",
            answer=sentence.text,
            output_format="spans",
        )
        score = _aggregate(tokens)
        token_spans = [(s["start"], s["end"]) for s in spans]
        results[sentence.id] = (score, token_spans)
    return results


def _aggregate(tokens: list[dict[str, Any]]) -> float:
    """Mean of the top-k token hallucination probabilities (0.0 if no tokens)."""
    probs = sorted((float(t["prob"]) for t in tokens), reverse=True)
    if not probs:
        return 0.0
    top = probs[:_TOP_K]
    return sum(top) / len(top)


@lru_cache(maxsize=1)
def _get_detector(model_path: str) -> Any:
    from lettucedetect.models.inference import HallucinationDetector

    return HallucinationDetector(method="transformer", model_path=model_path)
