"""Pipeline integration test — real orchestration + signal logic, all models mocked.

A ~100-word fixture document runs end-to-end through analyse(). Mocks sit at each
model boundary (summariser, classifier detector, NLI, Inseq attribution) so no
weights load. Exercises the A/B-gated attribution path and the full assembly.
"""

from collections.abc import Callable

import pytest

from sumlens import summarise as summarise_mod
from sumlens.ingest import split_sentences
from sumlens.pipeline import analyse
from sumlens.signals import attribution as attribution_mod
from sumlens.signals import classifier as classifier_mod
from sumlens.signals import nli as nli_mod
from sumlens.signals import support as support_mod
from sumlens.types import AnalysisConfig, AnalysisResult, Document

_RAW = (
    "The parliament met on Monday to discuss the proposed national budget for the "
    "coming fiscal year. Lawmakers from every party debated the spending priorities "
    "for several hours without reaching a clear consensus on the final allocations. "
    "The finance minister presented projections covering health, education, and "
    "transport infrastructure across the regions. Several members raised concerns "
    "about the long term sustainability of the proposed deficit levels. No final "
    "figure for total expenditure was announced to the press by the end of the day."
)

_SUMMARY_TEXT = "The bill passed today. The budget is one trillion euros."

# Signal A: LettuceDetect "tokens" output ({token,pred,prob}) and "spans" output.
_GROUNDED_TOKENS: list[dict[str, object]] = [
    {"token": "a", "pred": 0, "prob": 0.10},
    {"token": "b", "pred": 0, "prob": 0.05},
]
_HALLUCINATED_TOKENS: list[dict[str, object]] = [
    {"token": "a", "pred": 1, "prob": 0.95},
    {"token": "b", "pred": 1, "prob": 0.90},
    {"token": "c", "pred": 1, "prob": 0.85},
]
_HALLUCINATED_SPANS: list[dict[str, object]] = [
    {"start": 0, "end": 3, "confidence": 0.9, "text": "x"},
    {"start": 4, "end": 7, "confidence": 0.9, "text": "y"},
    {"start": 8, "end": 12, "confidence": 0.9, "text": "z"},
]


class _FakeDetector:
    def predict(
        self, *, context: list[str], question: str, answer: str, output_format: str
    ) -> list[dict[str, object]]:
        grounded = answer.startswith("The bill")
        if output_format == "spans":
            return [] if grounded else _HALLUCINATED_SPANS
        return _GROUNDED_TOKENS if grounded else _HALLUCINATED_TOKENS


class _FakeNLI:
    def __call__(
        self, pairs: list[dict[str, str]], top_k: object = None, batch_size: object = None
    ) -> list[list[dict[str, object]]]:
        out: list[list[dict[str, object]]] = []
        for pair in pairs:
            ent = 0.9 if "bill" in pair["text_pair"] else 0.2
            out.append(
                [{"label": "entailment", "score": ent}, {"label": "neutral", "score": 1.0 - ent}]
            )
        return out


def _fake_summariser(model_name: str) -> Callable[..., list[dict[str, str]]]:
    def _pipeline(text: str, **kwargs: object) -> list[dict[str, str]]:
        return [{"summary_text": _SUMMARY_TEXT}]

    return _pipeline


@pytest.fixture
def fixture_document() -> Document:
    return Document(
        id="doc-1",
        raw_text=_RAW,
        sentences=split_sentences(_RAW, "src"),
        source="text",
        meta={"word_count": len(_RAW.split())},
    )


def _install_mocks(monkeypatch: pytest.MonkeyPatch, document: Document) -> None:
    monkeypatch.setattr(summarise_mod, "_get_summariser", _fake_summariser)
    monkeypatch.setattr(classifier_mod, "_get_detector", lambda model_path: _FakeDetector())
    monkeypatch.setattr(nli_mod, "_get_nli", lambda model_name: _FakeNLI())
    # support.py binds its own `_get_nli` at import — patch it there too (ungated heatmap).
    monkeypatch.setattr(support_mod, "_get_nli", lambda model_name: _FakeNLI())

    src = document.sentences

    def _fake_attr(source_text: str, target_text: str, cfg: object) -> list[tuple[int, int, float]]:
        # spread equal mass over the first three source sentences -> peak 1/3
        return [(s.char_start, s.char_start + 4, 0.1) for s in src[:3]]

    monkeypatch.setattr(attribution_mod, "_source_token_attributions", _fake_attr)


def test_analyse_end_to_end(
    monkeypatch: pytest.MonkeyPatch, fixture_document: Document
) -> None:
    _install_mocks(monkeypatch, fixture_document)

    result = analyse(fixture_document, AnalysisConfig())

    assert isinstance(result, AnalysisResult)
    assert len(result.summary.sentences) == 2
    assert len(result.verdicts) == 2

    grounded, flagged = result.verdicts

    # Sentence 0: A low, B high -> not gated (Inseq scalar skipped), grounded. The
    # ungated support heatmap still fires: every source entails it (fake NLI 0.9),
    # so it now carries graded supporters even though Inseq did not run.
    assert grounded.sentence_id == "sum-0000"
    assert grounded.label == "grounded"
    assert grounded.signals.attribution is None  # Inseq scalar gated off
    assert grounded.evidence.top_source_sentence_ids  # support heatmap populated
    assert grounded.evidence.source_support  # graded pairs present
    assert all(score == pytest.approx(0.9) for _, score in grounded.evidence.source_support)
    assert grounded.evidence.failed_claims == []

    # Sentence 1: A high -> gated, Inseq scalar runs, hallucinated. Nothing in the
    # source entails it (fake NLI 0.2 < floor) -> support heatmap is empty: the
    # fabricated sentence correctly surfaces no supporters.
    assert flagged.sentence_id == "sum-0001"
    assert flagged.label == "hallucinated"
    assert flagged.signals.classifier == pytest.approx(0.9)
    assert flagged.signals.nli == pytest.approx(0.2)
    assert flagged.signals.attribution == pytest.approx(1 / 3)  # Inseq scalar, gated
    assert flagged.evidence.classifier_token_spans == [(0, 3), (4, 7), (8, 12)]
    assert flagged.evidence.top_source_sentence_ids == []  # below the support floor
    assert flagged.evidence.source_support == []
    assert len(flagged.evidence.failed_claims) == 1


def test_analyse_records_stage_timings(
    monkeypatch: pytest.MonkeyPatch, fixture_document: Document
) -> None:
    _install_mocks(monkeypatch, fixture_document)
    result = analyse(fixture_document, AnalysisConfig())
    assert set(result.timings_ms) == {
        "summarise", "classify", "nli", "attribute", "support", "fuse"
    }
    assert all(v >= 0 for v in result.timings_ms.values())


def test_analysis_result_round_trips(
    monkeypatch: pytest.MonkeyPatch, fixture_document: Document
) -> None:
    _install_mocks(monkeypatch, fixture_document)
    result = analyse(fixture_document, AnalysisConfig())
    assert AnalysisResult.model_validate_json(result.model_dump_json()) == result
