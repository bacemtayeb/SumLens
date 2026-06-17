"""Evaluation metrics — sentence-level F1, calibration error, reliability diagram.

Pure functions (no model dependency). `reliability_diagram` imports matplotlib
lazily so the rest of the module stays importable without it.
"""

from pathlib import Path


def sentence_f1(preds: dict[str, set[str]], golds: dict[str, set[str]]) -> dict[str, float]:
    """Micro precision/recall/F1 over hallucinated-sentence-id sets, across summaries."""
    tp = fp = fn = 0
    for key in set(golds) | set(preds):
        predicted = set(preds.get(key, set()))
        gold = set(golds.get(key, set()))
        tp += len(predicted & gold)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def roc_auc(scores: list[float], labels: list[int]) -> float:
    """Threshold-free ROC-AUC (rank-based, ties averaged). `scores` rank the
    positive class (label 1). Returns 0.0 if either class is absent."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return 0.0
    order = sorted(zip(scores, labels, strict=True), key=lambda p: p[0])
    ranks = [0.0] * len(order)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and order[j][0] == order[i][0]:
            j += 1
        rank = (i + j - 1) / 2 + 1  # 1-based average rank for the tie group
        for k in range(i, j):
            ranks[k] = rank
        i = j
    rank_sum_pos = sum(r for r, (_, label) in zip(ranks, order, strict=True) if label == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def pr_auc(scores: list[float], labels: list[int]) -> float:
    """Average precision (area under precision-recall curve). `scores` rank the
    positive class (label 1). Returns 0.0 if no positives. Better than ROC-AUC
    under heavy class imbalance; floor is the positive base rate."""
    n_pos = sum(labels)
    if not n_pos:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = fp = 0
    ap = 0.0
    prev_recall = 0.0
    for i in order:
        if labels[i] == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / n_pos
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def expected_calibration_error(
    scores: list[float], labels: list[int], n_bins: int = 10
) -> float:
    """ECE: weighted mean gap between bin confidence and bin accuracy."""
    if not scores:
        return 0.0
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for score, label in zip(scores, labels, strict=True):
        idx = min(int(score * n_bins), n_bins - 1)
        bins[idx].append((score, label))

    n = len(scores)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        confidence = sum(s for s, _ in bucket) / len(bucket)
        accuracy = sum(label for _, label in bucket) / len(bucket)
        ece += (len(bucket) / n) * abs(accuracy - confidence)
    return ece


def reliability_diagram(
    scores: list[float], labels: list[int], out_path: Path, n_bins: int = 10
) -> None:
    """Save a reliability plot (bin accuracy vs confidence) to `out_path`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers: list[float] = []
    accuracies: list[float] = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        bucket = [label for s, label in zip(scores, labels, strict=True) if lo <= s < hi]
        if not bucket:
            continue
        centers.append((lo + hi) / 2)
        accuracies.append(sum(bucket) / len(bucket))

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle="--", label="perfect")
    ax.plot(centers, accuracies, marker="o", label="model")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability diagram")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)
