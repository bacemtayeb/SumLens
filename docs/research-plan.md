# SumLens — Research Plan (Literature, Gap, Contribution)

> Working document. Lives in `docs/research-plan.md`. Read before touching code.

---

## 1. Problem framing

Given a document *D* and an AI-generated summary *S*, we want to:
1. **Detect** which sentences (or tokens) of *S* are not supported by *D*.
2. **Explain** *why* — what claim fails, against which source span — in a form a non-ML user can act on.
3. **Calibrate** the system's confidence so the user can rely on it.

The field has moved from (1) alone toward (1)+(2)+(3) jointly (HART, HuDEx, FACTUM, 2025–26). That's our target.

---

## 2. The five families of faithfulness detection

| Family | Idea | Representative work | Strength | Weakness |
|---|---|---|---|---|
| **Fact-based** | Compare extracted facts (n-grams, AMR, claims) between *S* and *D* | AMRFact (NAACL'24), FactCC | Interpretable | Brittle to paraphrase |
| **Classifier-based** | Fine-tune an encoder on labelled (faithful, hallucinated) pairs to do token/sentence/response classification | **LettuceDetect (2025)**, Luna, RIPA | Strong F1 on benchmark; cheap inference | Black-box; domain-fragile |
| **QA-based** | Generate questions from *S*, check whether *D* answers them consistently | QAGS, QuestEval | Interpretable | Slow; QG model errors compound |
| **NLI-based** | Treat each summary claim as a hypothesis, *D* as premise | **SummaC, FENICE (2024 SOTA on AggreFact)** | Strong, interpretable per-claim | Needs the right granularity; whole-summary NLI fails |
| **Internal-state / attribution** | Use gradients, attention, or activations of the summarizer itself | Inseq, ReDeEP, InterpDetect (2025), SPAD (Dec 2025) | Explains *why*; mechanistic | Expensive; signal can be noisy |

LLM-as-judge is sometimes treated as a sixth family. We skip it: too slow, opaque, and not what the course is testing.

---

## 3. Honest prior art for our original pitch

The first draft of SumLens was "use attribution on the summarizer to flag hallucinations and visualise it." Three pieces of prior art make this insufficient as a contribution on its own:

- **Liu et al. (2023), "On Early Detection of Hallucinations in Factual QA"** — already uses integrated-gradients token attribution as one of three hallucination signals (alongside softmax probabilities and internal activations). Different task (QA), but same method.
- **Inseq (Sarti et al. 2023)** — the toolkit explicitly proposes attribution aggregation across summary sentences as a use case. The pipeline we sketched is essentially what their paper invites users to build.
- **SPAD (Dec 2025)** — already aggregates token attributions by POS tag for hallucination detection. Goes further than our draft.

So "attribution heatmap for summary faithfulness" is a known technique. The contribution has to be in *combining* or *evaluating*, not in the heatmap itself.

---

## 4. Real open gaps (2024–26 literature)

1. **Detection ≠ diagnosis.** HuDEx (2025): *current benchmarks focus on detection but do not extend beyond identification.* No widely-adopted system tells a user *what claim failed against which evidence*.
2. **Domain fragility.** TreatFact (clinical, 2024): methods at ~F1 0.9 on AggreFact drop to ~0.5 on clinical text. Few systems are tested out of domain.
3. **No widely-adopted multi-signal fusion baseline.** Classifier, NLI, and attribution methods are usually compared *against* each other, not fused. Where fusion is tried (FACTUM, Jan 2026; SPAD, Dec 2025) it is recent and partial — typically two signal families, not three.
4. **Calibrated user trust is largely unaddressed.** Apple's evaluation work (2024) shows metrics don't align with human judgement. Almost no system reports a reliability diagram. Almost no system is evaluated with users in the loop.

---

## 5. Proposed contribution

**A three-signal hybrid hallucination detector and explainability dashboard for abstractive summaries, evaluated on RAGTruth-summarization and with a small human-in-the-loop study.**

Three signal layers, each implemented by composing an open-source artifact:

| Signal | What it gives | Implementation |
|---|---|---|
| **A — Encoder classifier** | Per-token hallucination probability | `lettucedetect` (ModernBERT, MIT, RAGTruth-trained) |
| **B — Atomic-claim NLI** | Per-claim entailment against the source | Atomic claim extraction + DeBERTa-v3-mnli or AlignScore, SummaC-granularity (sentence-pair NLI) |
| **C — Source attribution** | Which source spans the summary sentence relied on | `inseq` integrated gradients on the local summariser (BART/PEGASUS) |

**Fusion.** Calibrated logistic regression on (A, B, C) sentence-level scores, trained on a small RAGTruth-summarization split. The probability outputs feed Platt scaling for calibration.

**Dashboard.** For each summary sentence we render:
- a confidence band (from the fused probability),
- the atomic claim(s) that failed NLI (signal B → the "what"),
- the attributed source spans, including absence-of-support (signal C → the "why" / where to look).

This converts a flag into a verification action: *"Claim X is not entailed by source. The model attended to source span Y, which contains no matching figure. Check the source here."*

### Why this is defensible as a contribution at master's level
- Combining all three signal families with an ablation table is, as far as I can find, not in the published literature.
- Atomic-claim NLI surfaces the *what* (FENICE-style) while attribution surfaces the *where* (Inseq-style). Joining them is a small but real technical step.
- A reliability diagram + human study addresses the trust-calibration gap explicitly.
- All components are open-source; the engineering risk is integration, not novel modelling.

### What this is *not*
- Not a new SOTA on AggreFact. We won't beat FENICE on its own benchmark.
- Not a new model architecture. We compose existing ones.
- Not a faithfulness *guarantee*. Attribution shows what the model looked at; it does not certify truth.

---

## 6. Datasets

| Dataset | Role | Why |
|---|---|---|
| **RAGTruth** (Niu et al. 2024) | **Primary.** ~18k responses, span-level annotations, summarization subset, LLM-era models. | The current standard for span-level summarization hallucination work. |
| **AggreFact** (Tang et al. 2023) | Secondary. Aggregates FactCC, FRANK, SummEval, XSumFaith, etc. | Lets us compare to FENICE / SummaC baselines on familiar ground. |
| **XSum-Hallucination / FRANK** | Tertiary. Older, sentence-level. | Available as fallback if RAGTruth access is friction. |
| **TreatFact** (clinical, 2024) | Optional out-of-domain probe. | Honest domain-shift story; literature has flagged this as a real gap. |

V100 access on HPC makes evaluating across all of these realistic in our window.

---

## 7. Evaluation plan

### Automatic (must)
- **Sentence-level F1, precision, recall** on RAGTruth-summarization, for: classifier-only (A), NLI-only (B), attribution-only (C), each pair, and fused (A+B+C). One ablation table.
- **Calibration**: reliability diagram + Expected Calibration Error for fused output, pre- and post-Platt.
- **Generalization probe**: same fused model evaluated on AggreFact-XSUM-FTSOTA without retraining. Report the drop honestly.

### Out-of-domain (should)
- Same model on TreatFact (clinical). Expected drop is large — that *is* the finding.

### Human study (should — high marginal value)
- 8–10 participants (classmates ok), 30 min each.
- 10 documents, half with hallucinated summaries, half without. Within-subjects: half the documents with the dashboard, half with raw source+summary only.
- Outcome: time-to-detect, correctness, self-reported confidence (Likert). Pre-register on a one-page protocol committed to the repo before the study.
- This is the single most differentiating element for the report.

---

## 8. Risks and mitigation

| Risk | Likelihood | Mitigation |
|---|---|---|
| RAGTruth-summarization is mostly newswire; signal C may be noisy on LLM-generated summaries (decoder-only models attribute differently). | Med | Restrict signal C to documents summarised by our local BART/PEGASUS, where Inseq's gradient path is cleaner. Use LettuceDetect's pretrained labels for the LLM-summarised subset. |
| Attribution on V100 is slow at scale. | Med | Run attribution only on summary sentences flagged by A or B above τ; this turns C into a verification, not a primary classifier. ~5–10× speedup. |
| Human study doesn't show an effect (n is small). | Med | Pre-register, report null honestly; a clean null with a thoughtful discussion is fine for the report. |
| Fusion gains are marginal over the strongest single signal. | Low–Med | The ablation table is still the contribution. Marginal gains are interesting if explanation quality is the real benefit, not raw F1. |

---

## 9. What success looks like

1. A working dashboard that, for any uploaded document, returns a summary with flagged sentences and a concrete explanation per flag (atomic claim + attributed source spans).
2. One ablation table on RAGTruth-summarization with A, B, C, and fused, plus calibration numbers.
3. One out-of-domain result on AggreFact and/or TreatFact.
4. A small human-study result with a pre-registered protocol — even if the effect is null.
5. A written critical-limitations section that names what attribution *cannot* tell you and where the system would fail in production.

This is a tight, defensible, master's-grade research project, not a SOTA-chasing exercise.

---

## 10. Key references (for the report's bibliography)

- Tang et al. (2023). *Understanding Factual Errors in Summarization* (AggreFact). ACL.
- Niu et al. (2024). *RAGTruth: A Hallucination Corpus for Trustworthy RAG*. ACL.
- Maynez et al. (2020). *On Faithfulness and Factuality in Abstractive Summarization* (XSum-Hallucination). ACL.
- Pagnoni et al. (2021). *FRANK: A Benchmark for Detecting Factuality Errors in Summarization*. NAACL.
- Laban et al. (2022). *SummaC: Re-visiting NLI-based Models for Inconsistency Detection*. TACL.
- Scirè et al. (2024). *FENICE: Factuality Evaluation of Summarization based on NLI and Claim Extraction*. ACL.
- Kovács et al. (2025). *LettuceDetect: Hallucination Detection Framework for RAG*.
- Sarti et al. (2023). *Inseq: An Interpretability Toolkit for Sequence Generation Models*. ACL Demo.
- Liu et al. (2023). *On Early Detection of Hallucinations in Factual QA*.
- HuDEx (2025), HART (2026), SPAD (2025), FACTUM (2026) — recent diagnosis-oriented work.
- Cao et al. (2021). *Hallucinated but Factual!* — the "extrinsic but true" caveat.
- Apple ML (2024). *Evaluating Evaluation Metrics — The Mirage of Hallucination Detection*.
