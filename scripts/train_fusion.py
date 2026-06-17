"""Fit the fusion LogisticRegression + Platt calibrator from a features CSV.

CSV columns: classifier,nli,attribution,grounded
(grounded = 1 if the summary sentence is grounded, 0 if hallucinated). Features come
from a real-model pass over RAGTruth (worklist step 12); this script only fits and
pickles the models into --out-dir (default: models/).
"""

import argparse
import csv
import pickle
from pathlib import Path

from sumlens.fuse import fit_fusion, fit_platt


def _num(value: str, impute: float = 0.5) -> float:
    # A signal column is empty when that signal was off for the run (e.g.
    # attribution is off for RAGTruth). Impute neutral, matching the ablation.
    return impute if value == "" else float(value)


def _read(path: Path) -> tuple[list[list[float]], list[int]]:
    features: list[list[float]] = []
    grounded: list[int] = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            features.append(
                [_num(row["classifier"]), _num(row["nli"]), _num(row["attribution"])]
            )
            grounded.append(int(row["grounded"]))
    return features, grounded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="features CSV path")
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    features, grounded = _read(args.features)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fusion = fit_fusion(features, grounded)
    with (args.out_dir / "fusion.pkl").open("wb") as fh:
        pickle.dump(fusion, fh)

    fused_scores = [float(fusion.predict_proba([x])[0, 1]) for x in features]
    platt = fit_platt(fused_scores, grounded)
    with (args.out_dir / "platt.pkl").open("wb") as fh:
        pickle.dump(platt, fh)

    print(f"wrote {args.out_dir}/fusion.pkl and platt.pkl ({len(grounded)} rows)")


if __name__ == "__main__":
    main()
