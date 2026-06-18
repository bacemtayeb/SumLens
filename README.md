# SumLens

![CI](https://github.com/bacemtayeb/SumLens/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Explainability dashboard for AI-generated summaries. Upload a document, get an
abstractive summary, and see — sentence by sentence — how strongly each summary
sentence is grounded in the source, with low-confidence sentences flagged as
potential hallucinations.

> Course project · MICS · Principles of Software Development (S2 2025–26).

---

## Contents

- [What SumLens does](#what-sumlens-does)
- [Hardware requirements](#hardware-requirements)
- [Installation](#installation)
- [Running the app](#running-the-app)
- [Usage walkthrough](#usage-walkthrough)
- [Interpreting the results](#interpreting-the-results)
- [Exporting results](#exporting-results)
- [Development](#development)

---

## What SumLens does

SumLens takes a text document (pasted or PDF) and:

1. Summarises it locally using BART (`facebook/bart-large-cnn`) — no external API.
2. Scores each summary sentence against the source using three signals:
   - **Signal A — Classifier:** LettuceDetect flags hallucinated tokens.
   - **Signal B — NLI:** DeBERTa-v3 checks whether atomic claims are entailed by the source.
   - **Signal C — Attribution:** Inseq integrated gradients measure how much each source span influenced each summary sentence.
3. Fuses the signals into a single grounding score (0 = hallucinated, 1 = grounded).
4. Labels each sentence: **grounded**, **weakly grounded**, or **hallucinated**.
5. Displays the result as a colour-coded summary with click-to-highlight source spans.

---

## Hardware requirements

The pipeline loads three large transformer models. Running on CPU is supported but
can take several minutes per document.

| Setup | RAM | VRAM | Expected time |
|-------|-----|------|---------------|
| GPU (recommended) | 16 GB | 8 GB+ | ~30–60 s |
| CPU-only | 16 GB | — | 3–10 min |

Models are downloaded automatically from Hugging Face on first run (~4 GB total).
No paid API key is required.

---

## Installation

Requires **Python 3.11+** and **Git**.

```bash
git clone https://github.com/bacemtayeb/SumLens.git
cd SumLens
python3.11 -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
python -m nltk.downloader punkt punkt_tab
```

---

## Running the app

```bash
python app.py
```

Gradio prints a local URL, e.g.:

```
Running on local URL: http://127.0.0.1:7860
```

Open that URL in your browser. The app runs entirely on your machine — no data
leaves your computer.

---

## Usage walkthrough

### Step 1 — Load a document

You have two options:

- **Paste text** — click the *Paste text* box and type or paste your document
  (up to 10 000 words).
- **Upload a PDF** — click *or upload PDF* and select a file (up to 5 MB).

If you provide both, the PDF takes priority.

### Step 2 — Analyse

Click the **Analyse** button. The button is disabled while the pipeline runs
(typically 30–60 s on GPU, longer on CPU). Both export buttons appear once
analysis is complete.

### Step 3 — Read the summary

The right panel shows the summary with each sentence colour-coded:

| Colour | Label | Meaning |
|--------|-------|---------|
| Green | Grounded | Well-supported by the source |
| Orange | Weakly grounded | Partial support; treat with caution |
| Red | Hallucinated | Low support; likely fabricated or distorted |

### Step 4 — Trace a sentence to the source

Click any summary sentence. The left panel highlights (in yellow) the source
sentences most strongly attributed to it by the model. Click a different summary
sentence to switch the highlight.

### Step 5 — Adjust the thresholds (optional)

Two sliders let you change the decision boundaries without re-running the model:

- **τ hallucinated** (default 0.30) — sentences with a grounding score *below*
  this value are labelled hallucinated.
- **τ grounded** (default 0.70) — sentences with a grounding score *above* this
  value are labelled grounded. Anything in between is weakly grounded.

Move either slider and the summary colours update instantly.

---

## Interpreting the results

- The **grounding score** (0–1) represents how strongly the model believes a
  summary sentence is supported by the source. It is not a probability in a strict
  statistical sense — treat it as a relative risk indicator.
- A **hallucinated** label does not guarantee the sentence is wrong; it means the
  model could not find sufficient evidence in the source text. Always cross-check
  flagged sentences manually.
- The **signal breakdown** (JSON export) shows the individual classifier, NLI, and
  attribution scores for each sentence, which can help diagnose *why* a sentence
  was flagged.

---

## Exporting results

Two download buttons appear after analysis:

- **Export JSON** — downloads the full `AnalysisResult` as a JSON file. The schema
  matches `sumlens/types.py` and round-trips via `AnalysisResult.model_validate()`.
- **Export PDF** — downloads a human-readable PDF containing the colour-annotated
  summary, a legend, and a per-sentence signal-scores table.

---

## Development

### Quality gate (CI enforces on every PR)

```bash
ruff check . && mypy sumlens tests app.py && pytest -q --cov=sumlens --cov-fail-under=70
```

Lint (ruff), strict type-check (mypy), and tests with a ≥ 70 % coverage gate
must pass before any PR is merged.

### Project layout

```
sumlens/
  types.py          # canonical data model (AnalysisResult, etc.)
  ingest.py         # PDF / text → Document
  summarise.py      # BART summarisation
  signals/
    classifier.py   # Signal A — LettuceDetect
    nli.py          # Signal B — DeBERTa NLI
    attribution.py  # Signal C — Inseq attribution
  fuse.py           # logistic-regression fusion + Platt calibration
  pipeline.py       # orchestrates ingest → summarise → signals → fuse
app.py              # Gradio UI entry point
tests/              # pytest suite (all models mocked)
docs/               # requirements, data model, research plan
```

### Documentation

- [`docs/requirements.md`](docs/requirements.md) — functional / non-functional requirements, MoSCoW, user stories, traceability.
- [`docs/data-model.md`](docs/data-model.md) — canonical data types and JSON schema.
- [`docs/research-plan.md`](docs/research-plan.md) — signals, fusion, evaluation methodology.
- [`docs/mockup.html`](docs/mockup.html) — static HTML wireframe of the two-panel dashboard UI (open in any browser).
- [`docs/use-case.puml`](docs/use-case.puml) — PlantUML use-case diagram (UC-01 "Verify a Summary").

---

## License

Released under the [MIT License](LICENSE).
