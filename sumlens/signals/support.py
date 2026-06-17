"""Signal C (redesign) — generator-agnostic source attribution from an NLI matrix.

Inseq attribution (`attribution.py`) is gradient-based and needs the *generating*
model, so it is undefined for RAGTruth (external-model summaries). This signal
derives attribution from entailment alone, so it is defined for any (source,
summary) pair. For each summary sentence ``s`` we score entailment against every
source sentence ``j``, ``M[s][j] = P(src_j entails s)``, then collapse the row:

- ``attr_conc(s) = max_j M - mean_j M`` — support concentration. A grounded
  sentence has one sharp supporter; a fabricated one has diffuse, flat-low support.
- ``attr_loo(s)  = top1 - top2`` — necessity margin of the single best supporter.
- top-k source sentence ids — the UI heatmap (generator-free, no token offsets).

Reuses signal B's NLI model and batched call. Pure given the NLI boundary, which
tests mock via `_get_nli`. Consumed by `scripts/extract_features.py`.
"""

from sumlens.signals.nli import _entail_prob, _get_nli
from sumlens.types import AnalysisConfig, Document, Summary

_BATCH_SIZE = 64
_TOP_K = 5


def support_attribution(
    document: Document, summary: Summary, cfg: AnalysisConfig
) -> dict[str, tuple[float, float, list[str]]]:
    """Per summary sentence: (attr_conc, attr_loo, top-k source sentence ids)."""
    sources = document.sentences
    sentences = summary.sentences
    if not sentences or not sources:
        return {s.id: (0.0, 0.0, []) for s in sentences}

    nli = _get_nli(cfg.nli_model)
    pairs = [
        {"text": src.text, "text_pair": sent.text} for sent in sentences for src in sources
    ]
    batched = nli(pairs, top_k=None, batch_size=_BATCH_SIZE)
    n = len(sources)

    results: dict[str, tuple[float, float, list[str]]] = {}
    for i, sentence in enumerate(sentences):
        row = [_entail_prob(scores) for scores in batched[i * n : (i + 1) * n]]
        order = sorted(range(n), key=lambda j: row[j], reverse=True)
        top1 = row[order[0]]
        top2 = row[order[1]] if n > 1 else 0.0
        conc = top1 - sum(row) / n
        loo = top1 - top2
        top_ids = [sources[j].id for j in order[:_TOP_K]]
        results[sentence.id] = (conc, loo, top_ids)
    return results
