"""Produce the ablation table (the report's centrepiece) from features CSVs.

Reads train + test feature CSVs (from scripts/extract_features.py), fits a fusion
model per signal subset, and writes ablation.csv plus a run_manifest.json (git SHA,
config, timestamps, row counts) so any number in the report is reproducible.
"""

import argparse
import csv
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from sumlens.eval.ablation import ablation_table
from sumlens.types import AnalysisConfig

_COLUMNS = ["condition", "precision", "recall", "f1", "ece"]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="train features CSV")
    parser.add_argument("--test", type=Path, required=True, help="test features CSV")
    parser.add_argument("--out", type=Path, default=Path("ablation.csv"))
    args = parser.parse_args()

    train, test = _read(args.train), _read(args.test)
    table = ablation_table(train, test)

    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(table)

    manifest = {
        "git_sha": _git_sha(),
        "timestamp": datetime.now(UTC).isoformat(),
        "n_train": len(train),
        "n_test": len(test),
        "config": AnalysisConfig().model_dump(),
    }
    args.out.with_name("run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.out} ({len(table)} conditions) and run_manifest.json")


if __name__ == "__main__":
    main()
