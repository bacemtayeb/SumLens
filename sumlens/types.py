"""Core types — the single source of truth for SumLens.

Every module reads and writes these types. The JSON schema of `AnalysisResult`
is the API contract: whatever ends up on disk, in the UI, or in an export is a
view of this object.
"""

from typing import Any, Literal

from pydantic import BaseModel

# IDs are stable strings so they can appear in CSVs and the UI without index drift.


class Sentence(BaseModel):
    id: str                      # "src-0042" or "sum-0003"
    text: str
    char_start: int              # offset in the parent document/summary
    char_end: int


class Document(BaseModel):
    id: str
    raw_text: str                # cleaned, paragraph-segmented
    sentences: list[Sentence]    # tokenised with NLTK Punkt
    source: Literal["pdf", "text"]
    meta: dict[str, Any] = {}    # filename, word count, etc.


class Claim(BaseModel):
    id: str                      # "sum-0003-claim-1"
    sentence_id: str             # which summary sentence this claim came from
    text: str                    # atomic claim, e.g. "The bill allocates €2.4B."


class SignalScores(BaseModel):
    classifier: float | None     # signal A — token-prob aggregated to sentence
    nli: float | None            # signal B — max entailment over source sentences
    attribution: float | None    # signal C — peak attribution mass to source


class Evidence(BaseModel):
    """Per-sentence diagnostic payload — the explanation the UI renders."""

    failed_claims: list[Claim]                       # claims that failed NLI
    top_source_sentence_ids: list[str]               # top-k attributed source spans
    source_support: list[tuple[str, float]] = []     # (source_id, entailment) graded heatmap
    classifier_token_spans: list[tuple[int, int]]    # char offsets in the summary sentence
    notes: list[str] = []                            # human-readable diagnostic strings


class SentenceVerdict(BaseModel):
    sentence_id: str
    fused_score: float           # 0..1, calibrated
    label: Literal["grounded", "weak", "hallucinated"]
    signals: SignalScores
    evidence: Evidence


class Summary(BaseModel):
    id: str
    document_id: str
    text: str
    sentences: list[Sentence]
    model_name: str              # e.g. "facebook/bart-large-cnn"


class AnalysisResult(BaseModel):
    """The full JSON payload exposed by the pipeline. This is the API contract.
    Whatever ends up on disk, in the UI, or in an export is a view of this object."""

    document: Document
    summary: Summary
    verdicts: list[SentenceVerdict]
    config: "AnalysisConfig"
    timings_ms: dict[str, int]   # per-stage timing for the cost/perf section


class AnalysisConfig(BaseModel):
    summariser: str = "facebook/bart-large-cnn"
    summary_target_words: int = 150
    nli_model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    classifier_model: str = "KRLabsOrg/lettucedect-base-modernbert-en-v1"
    attribution_method: Literal["integrated_gradients", "input_x_gradient"] = "integrated_gradients"
    tau_hallucinated: float = 0.30   # below → hallucinated
    tau_grounded: float = 0.70       # above → grounded; in between → weak


AnalysisResult.model_rebuild()
