"""Ablation tests — synthetic separable data, no real models (sklearn only)."""

from sumlens.eval.ablation import ablation_table

_GROUNDED = {"classifier": 0.9, "nli": 0.8, "attr_conc": 0.7, "attr_loo": 0.6, "grounded": 1}
_HALLUCINATED = {"classifier": 0.1, "nli": 0.2, "attr_conc": 0.3, "attr_loo": 0.2, "grounded": 0}
_ROWS = [_GROUNDED, _HALLUCINATED] * 10


def test_ablation_table_conditions_and_scores() -> None:
    table = ablation_table(_ROWS, _ROWS)

    conditions = {row["condition"] for row in table}
    assert conditions == {
        "A", "B", "C", "D",
        "A+B", "A+C", "A+D", "B+C", "B+D", "C+D",
        "A+B+C", "A+B+D", "A+C+D", "B+C+D",
        "A+B+C+D",
    }

    for row in table:
        for key in ("roc_auc", "pr_auc", "precision", "recall", "f1", "ece"):
            assert isinstance(row[key], float)

    fused = next(row for row in table if row["condition"] == "A+B+C+D")
    assert fused["f1"] == 1.0  # perfectly separable -> perfect detection
    assert fused["roc_auc"] == 1.0
    assert fused["pr_auc"] == 1.0


def test_ablation_imputes_missing_attribution() -> None:
    # attr_conc missing ("") on every row -> still runs via imputation
    rows = [{**r, "attr_conc": ""} for r in _ROWS]
    table = ablation_table(rows, rows)
    c_only = next(row for row in table if row["condition"] == "C")
    assert isinstance(c_only["f1"], float)
