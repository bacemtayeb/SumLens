"""Assemble fusion training rows from per-sentence signal outputs + gold labels.

One row per summary sentence: the three signal scores (None if a signal did not
run for that sentence) and the grounded label (1 if grounded, 0 if the RAGTruth
gold marks the sentence hallucinated). This pure function is the testable core of
`scripts/extract_features.py`, which supplies the real signal outputs.
"""

from collections.abc import Mapping

from sumlens.types import Claim, Summary

FIELDNAMES = ["summary_id", "sentence_id", "classifier", "nli", "attribution", "grounded"]


def feature_rows(
    summary: Summary,
    hallucinated_ids: list[str],
    classifier_out: dict[str, tuple[float, list[tuple[int, int]]]],
    nli_out: dict[str, tuple[float, list[Claim]]],
    attribution_out: dict[str, tuple[float, list[str]]],
) -> list[dict[str, object]]:
    hallucinated = set(hallucinated_ids)
    rows: list[dict[str, object]] = []
    for sentence in summary.sentences:
        rows.append(
            {
                "summary_id": summary.id,
                "sentence_id": sentence.id,
                "classifier": _score(classifier_out, sentence.id),
                "nli": _score(nli_out, sentence.id),
                "attribution": _score(attribution_out, sentence.id),
                "grounded": 0 if sentence.id in hallucinated else 1,
            }
        )
    return rows


def _score(signal_out: Mapping[str, tuple[float, object]], sentence_id: str) -> float | None:
    entry = signal_out.get(sentence_id)
    return entry[0] if entry is not None else None
