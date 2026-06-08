"""Ablation tests — synthetic separable data, no real models (sklearn only)."""

from sumlens.eval.ablation import ablation_table

_GROUNDED = {"classifier": 0.9, "nli": 0.8, "attribution": 0.7, "grounded": 1}
_HALLUCINATED = {"classifier": 0.1, "nli": 0.2, "attribution": 0.3, "grounded": 0}
_ROWS = [_GROUNDED, _HALLUCINATED] * 10


def test_ablation_table_conditions_and_scores() -> None:
    table = ablation_table(_ROWS, _ROWS)

    conditions = {row["condition"] for row in table}
    assert conditions == {"A", "B", "C", "A+B", "A+C", "B+C", "A+B+C"}

    for row in table:
        for key in ("precision", "recall", "f1", "ece"):
            assert isinstance(row[key], float)

    fused = next(row for row in table if row["condition"] == "A+B+C")
    assert fused["f1"] == 1.0  # perfectly separable -> perfect detection


def test_ablation_imputes_missing_attribution() -> None:
    # attribution missing ("") on every row -> still runs via imputation
    rows = [{**r, "attribution": ""} for r in _ROWS]
    table = ablation_table(rows, rows)
    c_only = next(row for row in table if row["condition"] == "C")
    assert isinstance(c_only["f1"], float)
