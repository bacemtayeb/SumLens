"""Summarisation — Document into a Summary via a local transformers pipeline.

The model runs locally (no external inference API). The pipeline is built lazily
and cached so tests can mock `_get_summariser` at the module boundary and never
load weights. The output summary is re-tokenised with NLTK Punkt into sentences
with stable ids `sum-0000`, `sum-0001`, ...
"""

from functools import lru_cache
from typing import Any

from sumlens.ingest import split_sentences
from sumlens.types import AnalysisConfig, Document, Summary


def summarise(document: Document, cfg: AnalysisConfig) -> Summary:
    summariser = _get_summariser(cfg.summariser)
    max_length, min_length = _length_bounds(cfg.summary_target_words)
    output = summariser(
        document.raw_text,
        max_length=max_length,
        min_length=min_length,
        truncation=True,
    )
    text = output[0]["summary_text"].strip()
    return Summary(
        id=f"{document.id}-summary",
        document_id=document.id,
        text=text,
        sentences=split_sentences(text, "sum"),
        model_name=cfg.summariser,
    )


def _length_bounds(target_words: int) -> tuple[int, int]:
    """Words to a token max/min band (~1.3 tokens/word; min at ~60% of target)."""
    return int(target_words * 1.3), int(target_words * 0.6)


@lru_cache(maxsize=1)
def _get_summariser(model_name: str) -> Any:
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model=model_name, device=device)
