# SumLens — Requirements Specification

> Phase 2 · Requirements Engineering (Ch. 5)
> Course: Principles of Software Development · MICS · S2 2025–26
> Status: baseline for the 26 May Moodle submission · revision 1

---

## 1. Scope

SumLens is an explainability dashboard for AI-generated summaries. A user uploads a
document, receives an abstractive summary, and sees — sentence by sentence — how
strongly each summary sentence is grounded in the source, with low-confidence
sentences flagged as potential hallucinations.

This document defines functional (FR) and non-functional (NFR) requirements, their
MoSCoW priority, the user stories they trace to, and the verification method for each.
Every requirement is atomic, testable, and uniquely identified so it can be traced
through design, implementation, and the test suite.

---

## 2. Stakeholders & personas

| ID | Persona | Context | Core need |
|----|---------|---------|-----------|
| P1 | Journalist | News org; pre-processing parliamentary reports / press releases | Know which summary parts are safe to quote without a fact-check round-trip |
| P2 | Policy analyst | Think-tank, NGO, ministry; summarising legislation for briefings | See what the model *ignored* — that is where dropped caveats hide |
| P3 | Financial analyst | Reviewing earnings calls / SEC filings under time pressure | Verify that figures in a summary actually appear in the source, fast |

Secondary stakeholder: the development team (maintainability, reproducibility for grading).

---

## 3. Functional requirements

Priority: **M**ust / **S**hould / **C**ould / **W**on't (this iteration).
Verification: U = unit test, I = integration/API test, E = end-to-end UI test, M = manual/demo.

### 3.1 Ingestion
| ID | Requirement | Pri | Traces to | Acceptance criteria | Verif |
|----|-------------|-----|-----------|---------------------|-------|
| FR-01 | Accept a PDF upload up to 5 MB. | M | US-01 | A 5 MB PDF is accepted; a 5.1 MB file is rejected with a clear error. | I |
| FR-02 | Accept pasted plain text up to 10 000 words. | M | US-07 | Text within limit is processed; over-limit input is rejected with a clear message. | I |
| FR-03 | Extract clean, paragraph-segmented text and tokenise it into sentences. | M | US-01 | For a fixture PDF, extracted paragraph and sentence counts match expected values. | U |
| FR-04 | Reject unsupported or corrupt files without crashing. | M | US-08 | Uploading a `.docx` or a truncated PDF returns a 4xx error and a readable message; the service stays up. | I |

### 3.2 Summarisation
| ID | Requirement | Pri | Traces to | Acceptance criteria | Verif |
|----|-------------|-----|-----------|---------------------|-------|
| FR-05 | Produce an abstractive summary using a **locally-run** model (no dependency on a paid/external inference API at runtime). | M | US-01 | A summary is returned for a valid document using only models bundled with the deployment. | I |
| FR-06 | Summary length is configurable within 100–300 words. | S | US-01 | Requesting a 150-word target yields a summary within an accepted tolerance band. | I |

### 3.3 Attribution & flagging
| ID | Requirement | Pri | Traces to | Acceptance criteria | Verif |
|----|-------------|-----|-----------|---------------------|-------|
| FR-07 | Compute token-level attribution scores (summary tokens × source tokens). | M | US-02 | For a known input, the attribution matrix has shape \[summary_tokens × source_tokens]. | U |
| FR-08 | Aggregate token scores to a single confidence score per **summary sentence** (max + mean). | M | US-02 | Given a mock score matrix, per-sentence aggregation returns the expected vector. | U |
| FR-09 | Flag any summary sentence whose peak score is below threshold τ as a potential hallucination. | M | US-03 | Sentences with peak score < τ are flagged; those ≥ τ are not. Parametrised over τ. | U |
| FR-10 | Classify each sentence into grounded / weakly-grounded / hallucinated and expose this in the response. | M | US-03 | Each summary sentence in the API payload carries one of the three labels. | U/I |
| FR-11 | Threshold τ is configurable by the user (default 0.15). | S | US-05 | Changing τ in the UI re-flags sentences without re-running the model. | E |

### 3.4 Visualisation
| ID | Requirement | Pri | Traces to | Acceptance criteria | Verif |
|----|-------------|-----|-----------|---------------------|-------|
| FR-12 | Display source (left) and summary (right) in a two-panel view. | M | US-01 | Both panels render for a processed document. | E |
| FR-13 | On clicking a summary sentence, highlight it and its top-5 attributed source spans simultaneously. | M | US-06 | Clicking a sentence highlights it and ≤5 source spans; clicking another switches the highlight. | E |
| FR-14 | Render per-sentence confidence on a green→red scale. | M | US-01 | Sentence background colour reflects its confidence band. | E |
| FR-15 | Indicate processing progress so the user knows the system is working. | S | US-01 | A progress indicator is visible during a request and clears on completion. | E |

### 3.5 Export
| ID | Requirement | Pri | Traces to | Acceptance criteria | Verif |
|----|-------------|-----|-----------|---------------------|-------|
| FR-16 | Export the annotated result as JSON (sentences, scores, flags). | M | US-04 | Exported JSON validates against the documented schema and round-trips. | U/I |
| FR-17 | Export a rendered PDF of the annotated summary. | S | US-04 | A PDF is produced containing the summary with highlight/flag annotations. | I |

### 3.6 Future scope (declared, not built this iteration)
| ID | Requirement | Pri |
|----|-------------|-----|
| FR-18 | User authentication and per-user document history. | C |
| FR-19 | Side-by-side comparison of multiple attribution methods. | C |
| FR-20 | Multi-language document support. | C |

---

## 4. Non-functional requirements

| ID | Category | Requirement | Pri | Acceptance criteria | Verif |
|----|----------|-------------|-----|---------------------|-------|
| NFR-01 | Performance | Upload → heatmap visible in < 30 s for a 2 000-word document on the demo machine. | M | Timed run on reference hardware stays under 30 s. | M |
| NFR-02 | Reproducibility | The full pipeline runs offline from the repository, with no paid API key. | M | A fresh clone runs end-to-end via documented steps / `docker compose`. | M |
| NFR-03 | Accuracy | Flagging behaviour is evaluated against CNN/DailyMail references; known limitations are documented. | S | An evaluation note with measured behaviour exists in the report. | M |
| NFR-04 | Security/Privacy | No document persistence beyond the request session; no PII logged. | M | Code review confirms no writes to durable storage and no document content in logs. | M |
| NFR-05 | Usability | A user with no ML background completes upload → interpret → export without instructions. | S | A non-expert completes the flow unaided in a usability walkthrough. | M |
| NFR-06 | Maintainability | Ingestion, summariser, and attribution are independently replaceable modules behind stable interfaces. | M | Each can be swapped by changing one module without touching the others. | M |
| NFR-07 | Reliability | A failure in the model layer returns a graceful error, not a server crash. | M | Forcing a model error yields a 5xx with a clean message; service stays up. | I |
| NFR-08 | Code quality | CI enforces lint, type-check, and ≥ 70 % test coverage on `main`. | M | CI fails the build when any gate is violated. | M |

---

## 5. MoSCoW summary

- **Must:** FR-01–05, FR-07–10, FR-12–14, FR-16; NFR-01, 02, 04, 06, 07, 08. → the demonstrable MVP.
- **Should:** FR-06, FR-11, FR-15, FR-17; NFR-03, 05.
- **Could:** FR-18 (auth+history), FR-19 (multi-method), FR-20 (multi-language).
- **Won't (this iteration):** collaborative annotation, real-time streaming, model fine-tuning.

Cut order if the schedule slips: Could → Should → never the Must set.

---

## 6. User stories

- **US-01** — As a *journalist*, I want to upload a 10-page parliamentary report and see a highlighted summary, so I can identify which parts I can cite.
  *AC:* upload succeeds; summary renders; each sentence is colour-coded by confidence.
- **US-02** — As a *policy analyst*, I want to see which source sentences the model ignored, so I can judge whether caveats were dropped.
  *AC:* clicking a summary sentence reveals its top source spans; unattributed source regions are visibly distinguishable.
- **US-03** — As a *financial analyst*, I want figures in the summary flagged when they are not grounded in the source, so I do not repeat an invented number.
  *AC:* a summary sentence containing an ungrounded figure is flagged below τ.
- **US-04** — As a *journalist*, I want to export the annotated summary, so I can share the verdict with my editor.
  *AC:* JSON (Must) and PDF (Should) exports reflect the on-screen flags and highlights.
- **US-05** — As *any user*, I want to adjust the strictness threshold τ, so I can tune sensitivity to my risk tolerance.
  *AC:* changing τ re-flags sentences immediately without re-running the model.
- **US-06** — As *any user*, I want to click a summary sentence and see its supporting source spans, so I can verify it at a glance.
  *AC:* selected sentence + ≤5 source spans highlight together.
- **US-07** — As a *policy analyst*, I want to paste raw text instead of a PDF, so I can check content that is not in a file.
  *AC:* pasted text within limit is processed identically to an uploaded PDF.
- **US-08** — As *any user*, I want a clear message when my input is too large or unsupported, so I am not left guessing.
  *AC:* over-limit / unsupported input returns a readable error and the app stays usable.

---

## 7. Use cases

Primary use case **UC-01 "Verify a summary"** (actor: any persona):
1. Actor submits a document (upload PDF or paste text).
2. System validates input (FR-01/02/04).
3. System summarises (FR-05) and computes attribution (FR-07/08).
4. System flags low-confidence sentences (FR-09/10).
5. System renders the two-panel heatmap (FR-12/13/14).
6. Actor optionally adjusts τ (FR-11) and exports (FR-16/17).

Extensions: 2a invalid input → error (FR-04/08-msg); 3a model failure → graceful error (NFR-07).

> The UML use-case diagram lives in `docs/use-case.puml` (drafted next).

---

## 8. Traceability matrix (story → requirements)

| Story | Satisfied by |
|-------|--------------|
| US-01 | FR-01, FR-03, FR-05, FR-12, FR-14, FR-15 |
| US-02 | FR-07, FR-08, FR-13 |
| US-03 | FR-09, FR-10 |
| US-04 | FR-16, FR-17 |
| US-05 | FR-11 |
| US-06 | FR-13 |
| US-07 | FR-02 |
| US-08 | FR-04 |

Every Must-priority FR traces to at least one story; no orphan requirements.

---

## 9. Assumptions & constraints

- Local model choice (e.g. a DistilBART/BART-class summariser) is fixed at the
  architecture stage; requirements are written model-agnostically so the summariser
  can be swapped (NFR-06).
- Single-session, single-user per request; no concurrency guarantees beyond the demo.
- "Hallucination" here means *low source-attribution confidence*, not verified factual
  falsehood — a limitation stated explicitly in the report (NFR-03).
- Reference hardware for NFR-01 is the demo laptop, documented in the README.
