# SumLens — Data Model & Module Contract

> Lives in `docs/data-model.md`. This is the spine: every module reads and writes
> these types. If you change a type here, every module that touches it changes too.
> Implement this **before** writing any module logic.

---

## 1. Project shape

```
sumlens/
├── pyproject.toml
├── app.py                    # Gradio entry point (thin)
├── sumlens/
│   ├── __init__.py
│   ├── types.py              # the dataclasses below — ONE source of truth
│   ├── ingest.py             # PDF/text → Document
│   ├── summarise.py          # Document → Summary
│   ├── signals/
│   │   ├── __init__.py
│   │   ├── classifier.py     # signal A — LettuceDetect wrapper
│   │   ├── nli.py            # signal B — atomic claims + NLI
│   │   └── attribution.py    # signal C — Inseq integrated gradients
│   ├── fuse.py               # logistic regression + Platt calibration
│   ├── pipeline.py           # orchestrates ingest → summarise → A/B/C → fuse
│   └── eval/
│       ├── ragtruth.py       # loader + evaluation harness
│       ├── aggrefact.py      # secondary benchmark
│       └── metrics.py        # F1, ECE, reliability diagram
├── tests/                    # mirrors sumlens/ — one test file per module
├── notebooks/                # exploration only, not in the import graph
├── scripts/
│   ├── train_fusion.py       # fits the logistic regression on RAGTruth split
│   └── evaluate.py           # produces ablation table + calibration plot
└── study/
    ├── protocol.md           # the pre-registered user-study protocol
    ├── documents/            # the 6 locked study documents + ground truth
    └── results/              # raw response CSVs (gitignored if PII)
```

Keep `sumlens/` a **library**. `app.py` is a thin Gradio wrapper that imports
from it. Tests target the library; the UI is integration-tested separately.

---

## 2. Core types — `sumlens/types.py`

Implement with `pydantic.BaseModel` (free JSON schema + validation) or plain
`@dataclass(frozen=True)`. Pydantic is recommended; the API "contract" is its
schema.

```python
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
    nli_model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli"
    classifier_model: str = "KRLabsOrg/lettucedect-base-modernbert-en-v1"
    attribution_method: Literal["integrated_gradients", "input_x_gradient"] = "integrated_gradients"
    tau_hallucinated: float = 0.30   # below → hallucinated
    tau_grounded: float = 0.70       # above → grounded; in between → weak
```

**Three things this design enforces:**
1. Every module produces or consumes a typed object — no `dict[str, Any]` flowing between modules.
2. The JSON schema of `AnalysisResult` is the JSON the Gradio app exports — one shape, not two.
3. Thresholds (`tau_*`) live in config, not constants. The user study lets us tune them honestly.

---

## 3. Module interfaces — function signatures

These are the only public functions each module exposes. Anything else is private.
Stick to these signatures; if an implementation drifts, point it back here.

### `ingest.py`
```python
def load_pdf(path: Path) -> Document: ...
def load_text(text: str) -> Document: ...
```
Internally: pdfplumber for PDF, NLTK Punkt for sentence splitting, paragraph
heuristic on blank lines. Returns ids `src-0000`, `src-0001`, ...

### `summarise.py`
```python
def summarise(document: Document, cfg: AnalysisConfig) -> Summary: ...
```
HuggingFace `transformers` pipeline, `device=0` if CUDA available. Truncates to
`model.config.max_position_embeddings`. Re-tokenises the output summary with
NLTK Punkt to populate `Summary.sentences` with ids `sum-0000`, `sum-0001`, ...

### `signals/classifier.py`
```python
def classify(document: Document, summary: Summary, cfg: AnalysisConfig)
    -> dict[str, tuple[float, list[tuple[int, int]]]]: ...
# Returns: {sentence_id: (score_in_0_1, [token_spans_in_summary_sentence])}
```
Thin wrapper around `lettucedetect`. Aggregates per-token probs to a per-sentence
score (mean of top-k token probs is a sane default; document this choice).

### `signals/nli.py`
```python
def extract_claims(summary: Summary) -> list[Claim]: ...
def entail(claims: list[Claim], document: Document, cfg: AnalysisConfig)
    -> dict[str, tuple[float, list[Claim]]]: ...
# Returns: {sentence_id: (max_entail_prob, claims_that_failed)}
```
Claim extraction can start simple: split summary sentences into clauses on
conjunctions, treat each clause as a claim. NLI runs each claim × each source
sentence; per claim, take max entailment over source sentences (SummaC's `Conv`
style). Per summary sentence: take min over its claims (weakest claim wins);
list the failing claims for the UI.

### `signals/attribution.py`
```python
def attribute(document: Document, summary: Summary, cfg: AnalysisConfig)
    -> dict[str, tuple[float, list[str]]]: ...
# Returns: {sentence_id: (peak_source_attribution, [top_source_sentence_ids])}
```
Inseq integrated gradients on the **same model that produced the summary**
(this is why we keep the summariser local). 30–50 steps. Aggregates token
attribution to source sentences (sum of token-level scores within a sentence,
normalised). For speed, only run on sentences flagged by A or B above some
gate threshold.

### `fuse.py`
```python
def fuse(signals: dict[str, SignalScores], model_path: Path) -> dict[str, float]: ...
def calibrate(scores: dict[str, float], platt_path: Path) -> dict[str, float]: ...
def label(score: float, cfg: AnalysisConfig) -> Literal["grounded", "weak", "hallucinated"]: ...
```
`model_path` points at a pickled `sklearn.linear_model.LogisticRegression`
trained offline by `scripts/train_fusion.py` on RAGTruth-summarization.
`platt_path` is the calibrator fitted on a held-out split. Both must be loadable
without internet access (commit them under `models/` or use a release asset).

### `pipeline.py`
```python
def analyse(document: Document, cfg: AnalysisConfig) -> AnalysisResult: ...
```
Orchestrates: summarise → run A and B in parallel → gate C on (A ∨ B above
gate) → fuse → label → assemble evidence → return. Times every stage into
`timings_ms`.

---

## 4. Eval harness — `sumlens/eval/`

### `ragtruth.py`
```python
def load_split(split: Literal["train", "test"], task: str = "Summary")
    -> list[tuple[Document, Summary, list[str]]]: ...
# Returns: (doc, summary, list of hallucinated sentence ids per summary)
```
Maps RAGTruth's character-offset span annotations to summary sentence ids using
your own tokeniser. Document this mapping in the report (it's a real eval choice).

### `metrics.py`
```python
def sentence_f1(preds: dict, golds: dict) -> dict[str, float]: ...
def expected_calibration_error(scores: list[float], labels: list[int], n_bins: int = 10) -> float: ...
def reliability_diagram(scores, labels, out_path: Path) -> None: ...
```

### `scripts/evaluate.py`
Outputs **one CSV** with rows = ablation conditions (A, B, C, A+B, A+C, B+C, A+B+C),
columns = precision/recall/F1/ECE on RAGTruth-test and AggreFact-XSUM-FTSOTA.
This CSV is the centrepiece table of the report.

---

## 5. Configuration & reproducibility

- Every script accepts a `--config path/to/yaml`. The YAML deserialises to `AnalysisConfig`.
- Pin every model revision: `model_name@revision_hash` in the config, not just the name.
- Seed everything (`torch.manual_seed`, `numpy.random.seed`, `random.seed`) and log the seed.
- `scripts/evaluate.py` writes a `run_manifest.json` next to its CSV: git SHA, config, seeds, hardware, timestamps. The professor can reproduce any number in the report from that manifest.

---

## 6. Order of implementation

1. `types.py` — write this first, completely. Everything else imports from it.
2. `ingest.py` + tests against a fixture PDF and a fixture string.
3. `summarise.py` + tests with the model **mocked** (so CI passes without weights).
4. `signals/classifier.py` + mocked tests (LettuceDetect mocked at the module boundary).
5. `signals/nli.py` + mocked tests (NLI mocked at the module boundary).
6. `signals/attribution.py` + mocked tests (Inseq mocked at the module boundary).
7. `fuse.py` — implement first with an identity fusion (mean of available signals) so the pipeline runs end-to-end before training is wired up.
8. `pipeline.py` + an integration test that runs the whole thing on a 100-word fixture document with all models mocked.
9. `app.py` (Gradio) — wires `pipeline.analyse` into `gr.Interface`. Use `gr.HighlightedText` for the summary panel.
10. `eval/ragtruth.py` + `scripts/evaluate.py` — only after the pipeline runs end-to-end with mocks.
11. `scripts/train_fusion.py` — fits the LR, pickles it, replaces the identity fusion.
12. Real models swapped in last, one signal at a time, verifying on HPC.

**Rule: never run the real models in tests.** Mock at the module boundary. Real-model runs happen only in `scripts/evaluate.py` on HPC.
