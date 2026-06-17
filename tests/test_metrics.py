"""Metrics tests — F1, ECE (pure), reliability diagram (writes a file)."""

from pathlib import Path

import pytest

from sumlens.eval.metrics import (
    expected_calibration_error,
    pr_auc,
    reliability_diagram,
    roc_auc,
    sentence_f1,
)


def test_sentence_f1_perfect() -> None:
    preds = {"s1": {"sum-0000", "sum-0002"}}
    golds = {"s1": {"sum-0000", "sum-0002"}}
    assert sentence_f1(preds, golds) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_sentence_f1_partial() -> None:
    # gold {0,1}; pred {0,2} -> tp1 fp1 fn1 -> p=r=0.5, f1=0.5
    result = sentence_f1({"s1": {"sum-0000", "sum-0002"}}, {"s1": {"sum-0000", "sum-0001"}})
    assert result == {"precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_sentence_f1_empty() -> None:
    assert sentence_f1({}, {}) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_ece_perfectly_calibrated() -> None:
    # all confidence 1.0 and all correct -> gap 0
    assert expected_calibration_error([1.0, 1.0], [1, 1], n_bins=10) == pytest.approx(0.0)


def test_ece_known_value() -> None:
    # two points in same bin: conf=(0.8+0.8)/2=0.8, acc=(1+0)/2=0.5 -> ece=0.3
    assert expected_calibration_error([0.8, 0.8], [1, 0], n_bins=10) == pytest.approx(0.3)


def test_ece_empty() -> None:
    assert expected_calibration_error([], []) == 0.0


def test_roc_auc_perfect_separation() -> None:
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0


def test_roc_auc_inverted_is_zero() -> None:
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0


def test_roc_auc_ties_give_half() -> None:
    # all scores equal -> every pair tied -> AUC 0.5
    assert roc_auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5


def test_roc_auc_single_class_returns_zero() -> None:
    assert roc_auc([0.1, 0.9], [1, 1]) == 0.0


def test_pr_auc_perfect_separation() -> None:
    assert pr_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0


def test_pr_auc_floor_is_base_rate() -> None:
    # scores carry no signal (descending but labels random) -> AP near base rate
    assert pr_auc([0.4, 0.3, 0.2, 0.1], [1, 0, 0, 0]) == pytest.approx(1.0)
    # worst ranking: the only positive is last -> precision 1/4 at recall 1
    assert pr_auc([0.4, 0.3, 0.2, 0.1], [0, 0, 0, 1]) == pytest.approx(0.25)


def test_pr_auc_no_positives_returns_zero() -> None:
    assert pr_auc([0.1, 0.9], [0, 0]) == 0.0


def test_reliability_diagram_writes_file(tmp_path: Path) -> None:
    out = tmp_path / "reliability.png"
    reliability_diagram([0.1, 0.4, 0.9, 0.95], [0, 0, 1, 1], out)
    assert out.exists() and out.stat().st_size > 0
