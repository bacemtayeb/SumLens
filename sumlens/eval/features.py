"""Assemble fusion training rows from per-sentence signal outputs + gold labels.

One row per summary sentence: the signal scores (None if a signal did not run for
that sentence) and the grounded label (1 if grounded, 0 if the RAGTruth gold marks
the sentence hallucinated). Signal C is the generator-agnostic support attribution
(`signals/support.py`), which yields two scalars per sentence: attr_conc (support
concentration) and attr_loo (best-supporter necessity margin). This pure function
is the testable core of `scripts/extract_features.py`.
"""

from collections.abc import Mapping

from sumlens.types import Claim, Summary

FIELDNAMES = [
    "summary_id",
    "sentence_id",
    "classifier",
    "nli",
    "attr_conc",
    "attr_loo",
    "grounded",
]


def feature_rows(
    summary: Summary,
    hallucinated_ids: list[str],
    classifier_out: dict[str, tuple[float, list[tuple[int, int]]]],
    nli_out: dict[str, tuple[float, list[Claim]]],
    support_out: dict[str, tuple[float, float, list[tuple[str, float]]]],
) -> list[dict[str, object]]:
    hallucinated = set(hallucinated_ids)
    rows: list[dict[str, object]] = []
    for sentence in summary.sentences:
        rows.append(
            {
                "summary_id": summary.id,
                "sentence_id": sentence.id,
                "classifier": _at(classifier_out, sentence.id, 0),
                "nli": _at(nli_out, sentence.id, 0),
                "attr_conc": _at(support_out, sentence.id, 0),
                "attr_loo": _at(support_out, sentence.id, 1),
                "grounded": 0 if sentence.id in hallucinated else 1,
            }
        )
    return rows


def _at(
    signal_out: Mapping[str, tuple[object, ...]], sentence_id: str, index: int
) -> float | None:
    entry = signal_out.get(sentence_id)
    return float(entry[index]) if entry is not None else None  # type: ignore[arg-type]
