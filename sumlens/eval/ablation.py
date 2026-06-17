"""Ablation over signal subsets — the report's centrepiece table.

For each non-empty subset of {classifier (A), NLI (B), attr_conc (C), attr_loo (D)}
we fit a fusion LogisticRegression on the train rows (using only that subset's
columns), predict on the test rows, and report detection precision/recall/F1
(positive class = hallucinated) plus the calibration error of the grounding
probability. C and D are the two scalars of the generator-agnostic support
attribution (signals/support.py).

Rows are mappings with keys: classifier, nli, attr_conc, attr_loo (float or
None/""), and grounded (1 grounded / 0 hallucinated). Missing values are imputed.
"""

from collections.abc import Mapping, Sequence
from itertools import combinations

from sumlens.eval.metrics import expected_calibration_error, pr_auc, roc_auc
from sumlens.fuse import fit_fusion

_SIGNALS = ("classifier", "nli", "attr_conc", "attr_loo")
_LETTER = {"classifier": "A", "nli": "B", "attr_conc": "C", "attr_loo": "D"}

Row = Mapping[str, object]


def ablation_table(
    train: Sequence[Row], test: Sequence[Row], impute: float = 0.5
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for size in range(1, len(_SIGNALS) + 1):
        for combo in combinations(_SIGNALS, size):
            rows.append(_evaluate_combo(combo, train, test, impute))
    return rows


def _evaluate_combo(
    combo: tuple[str, ...], train: Sequence[Row], test: Sequence[Row], impute: float
) -> dict[str, object]:
    x_train = [[_feature(r, s, impute) for s in combo] for r in train]
    y_train = [_grounded(r) for r in train]
    model = fit_fusion(x_train, y_train)

    x_test = [[_feature(r, s, impute) for s in combo] for r in test]
    grounded_proba = [float(p) for p in model.predict_proba(x_test)[:, 1]]
    y_test = [_grounded(r) for r in test]

    pred_hallucinated = [1 if p < 0.5 else 0 for p in grounded_proba]
    true_hallucinated = [1 - g for g in y_test]
    precision, recall, f1 = _prf(true_hallucinated, pred_hallucinated)

    # Threshold-free detection metrics (positive class = hallucinated). f1 above
    # is a single fixed-0.5 operating point and is misleading under the heavy
    # hallucination class imbalance; roc_auc/pr_auc are the headline numbers.
    proba_hallucinated = [1.0 - p for p in grounded_proba]

    return {
        "condition": "+".join(_LETTER[s] for s in combo),
        "roc_auc": roc_auc(proba_hallucinated, true_hallucinated),
        "pr_auc": pr_auc(proba_hallucinated, true_hallucinated),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ece": expected_calibration_error(grounded_proba, y_test),
    }


def _feature(row: Row, signal: str, impute: float) -> float:
    value = row.get(signal)
    if value is None or value == "":
        return impute
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value))


def _grounded(row: Row) -> int:
    value = row["grounded"]
    if isinstance(value, (int, float)):
        return int(value)
    return int(str(value))


def _prf(true: list[int], pred: list[int]) -> tuple[float, float, float]:
    tp = sum(1 for t, p in zip(true, pred, strict=True) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(true, pred, strict=True) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true, pred, strict=True) if t == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1
