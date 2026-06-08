"""Signal B — atomic-claim NLI against the source.

Claims start simple: each summary sentence is split into clauses on coordinating
conjunctions, and each clause is treated as an atomic claim. For each claim we run
NLI against every source sentence and keep the max entailment probability
(SummaC-Conv style). Per summary sentence the score is the min over its claims
(the weakest claim wins); claims whose entailment falls below `_ENTAIL_FAIL` are
returned as the failing claims for the UI.
"""

import re
from collections import defaultdict
from functools import lru_cache
from typing import Any

from sumlens.types import AnalysisConfig, Claim, Document, Summary

_ENTAIL_FAIL = 0.5
_BATCH_SIZE = 64
_CLAUSE = re.compile(r",?\s+(?:and|but|or|however|whereas|while)\s+|\s*;\s*", re.IGNORECASE)


def extract_claims(summary: Summary) -> list[Claim]:
    claims: list[Claim] = []
    for sentence in summary.sentences:
        for n, clause in enumerate(_split_clauses(sentence.text), start=1):
            claims.append(
                Claim(id=f"{sentence.id}-claim-{n}", sentence_id=sentence.id, text=clause)
            )
    return claims


def entail(
    claims: list[Claim], document: Document, cfg: AnalysisConfig
) -> dict[str, tuple[float, list[Claim]]]:
    nli = _get_nli(cfg.nli_model)
    sources = [s.text for s in document.sentences]
    scored: dict[str, list[tuple[Claim, float]]] = defaultdict(list)

    if claims and sources:
        # One batched NLI call over every (source, claim) pair, then reduce
        # max-over-sources per claim. Batching is far faster than per-pair calls
        # on GPU; the resulting scores are identical.
        pairs = [
            {"text": src, "text_pair": claim.text} for claim in claims for src in sources
        ]
        batched = nli(pairs, top_k=None, batch_size=_BATCH_SIZE)
        n = len(sources)
        for i, claim in enumerate(claims):
            prob = max(
                (_entail_prob(scores) for scores in batched[i * n : (i + 1) * n]),
                default=0.0,
            )
            scored[claim.sentence_id].append((claim, prob))
    elif claims:
        for claim in claims:
            scored[claim.sentence_id].append((claim, 0.0))

    results: dict[str, tuple[float, list[Claim]]] = {}
    for sentence_id, claim_scores in scored.items():
        sentence_score = min(p for _, p in claim_scores)
        failed = [c for c, p in claim_scores if p < _ENTAIL_FAIL]
        results[sentence_id] = (sentence_score, failed)
    return results


def _split_clauses(text: str) -> list[str]:
    parts = [p.strip() for p in _CLAUSE.split(text)]
    parts = [p for p in parts if p]
    return parts or [text.strip()]


def _entail_prob(scores: list[dict[str, Any]]) -> float:
    return next((s["score"] for s in scores if "entail" in s["label"].lower()), 0.0)


@lru_cache(maxsize=1)
def _get_nli(model_name: str) -> Any:
    import torch
    from transformers import pipeline

    on_gpu = torch.cuda.is_available()
    kwargs: dict[str, Any] = {"device": 0 if on_gpu else -1}
    if on_gpu:
        # bf16 ~2x faster on L40S/Ada; entailment scores drift negligibly.
        kwargs["torch_dtype"] = torch.bfloat16
    return pipeline("text-classification", model=model_name, **kwargs)
