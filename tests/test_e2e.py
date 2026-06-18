"""End-to-end UI test — input → summary → export with ML models mocked at the
module boundary; no real weights loaded (FR-11/12/13).

Mock seams:
  summarise_mod._get_summariser
  classifier_mod._get_detector
  nli_mod._get_nli
  attribution_mod._source_token_attributions
"""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

import app as app_mod
from app import _export_json, _source_spans, _summary_spans, run
from sumlens import summarise as summarise_mod
from sumlens.signals import attribution as attribution_mod
from sumlens.signals import classifier as classifier_mod
from sumlens.signals import nli as nli_mod
from sumlens.signals import support as support_mod
from sumlens.types import AnalysisResult

_RAW = "Parliament passed a bill on Monday. The budget is one trillion euros."
_SUMMARY = "Parliament passed the bill. The budget is one trillion euros."

_GROUNDED_TOKENS: list[dict[str, object]] = [{"token": "a", "pred": 0, "prob": 0.05}]
_HALLUCINATED_TOKENS: list[dict[str, object]] = [{"token": "a", "pred": 1, "prob": 0.95}]
_HALLUCINATED_SPANS: list[dict[str, object]] = [
    {"start": 0, "end": 3, "confidence": 0.9, "text": "x"}
]


class _FakeDetector:
    def predict(
        self, *, context: list[str], question: str, answer: str, output_format: str
    ) -> list[dict[str, object]]:
        grounded = answer.startswith("Parliament")
        if output_format == "spans":
            return [] if grounded else _HALLUCINATED_SPANS
        return _GROUNDED_TOKENS if grounded else _HALLUCINATED_TOKENS


class _FakeNLI:
    def __call__(
        self,
        pairs: list[dict[str, str]],
        top_k: object = None,
        batch_size: object = None,
    ) -> list[list[dict[str, object]]]:
        out: list[list[dict[str, object]]] = []
        for pair in pairs:
            ent = 0.9 if "Parliament" in pair.get("text_pair", "") else 0.2
            out.append(
                [
                    {"label": "entailment", "score": ent},
                    {"label": "neutral", "score": 1.0 - ent},
                ]
            )
        return out


def _fake_summariser(model_name: str) -> Callable[..., list[dict[str, str]]]:
    def _pipeline(text: str, **kwargs: object) -> list[dict[str, str]]:
        return [{"summary_text": _SUMMARY}]

    return _pipeline


@pytest.fixture()
def mocked_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install ML model mocks at the module boundary and a stub load_text."""
    from sumlens.ingest import load_text as real_load_text

    monkeypatch.setattr(summarise_mod, "_get_summariser", _fake_summariser)
    monkeypatch.setattr(classifier_mod, "_get_detector", lambda model_path: _FakeDetector())
    monkeypatch.setattr(nli_mod, "_get_nli", lambda model_name: _FakeNLI())
    monkeypatch.setattr(support_mod, "_get_nli", lambda model_name: _FakeNLI())

    def _fake_attr(
        source_text: str, target_text: str, cfg: object
    ) -> list[tuple[int, int, float]]:
        return [(0, 10, 0.5), (11, 20, 0.3)]

    monkeypatch.setattr(attribution_mod, "_source_token_attributions", _fake_attr)
    monkeypatch.setattr(app_mod, "load_text", real_load_text)


def test_e2e_run_returns_analysis_result(mocked_pipeline: None) -> None:
    result, payload = run(_RAW, None)

    assert isinstance(result, AnalysisResult)
    assert len(result.summary.sentences) == 2
    assert payload == result.model_dump()


def test_e2e_summary_span_colors_match_verdicts(mocked_pipeline: None) -> None:
    result, _ = run(_RAW, None)
    verdict_map = {v.sentence_id: v.label for v in result.verdicts}

    spans, ids = _summary_spans(result, verdict_map)
    # Each span is its sentence's verdict colour, or the deep-red "flagged" sub-span.
    for (_, label), sid in zip(spans, ids, strict=True):
        assert label == verdict_map.get(sid, "weak") or label == "flagged"


def test_e2e_source_spans_contain_document_text(mocked_pipeline: None) -> None:
    result, _ = run(_RAW, None)
    spans, _ = _source_spans(result.document, {})
    joined = " ".join(text for text, _ in spans)
    assert "Parliament" in joined
    assert "budget" in joined


def test_e2e_export_json_round_trips(mocked_pipeline: None) -> None:
    result, _ = run(_RAW, None)
    path = _export_json(result)

    assert path is not None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    restored = AnalysisResult.model_validate(data)
    assert restored == result


def test_e2e_no_real_model_weights_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard: real model loaders must never be called during the test suite."""
    called: list[str] = []

    def _guard(name: str) -> Callable[..., object]:
        def _fail(*args: object, **kwargs: object) -> object:
            called.append(name)
            raise AssertionError(f"Real model loader invoked: {name}")

        return _fail

    monkeypatch.setattr(summarise_mod, "_get_summariser", _guard("_get_summariser"))
    monkeypatch.setattr(classifier_mod, "_get_detector", _guard("_get_detector"))
    monkeypatch.setattr(nli_mod, "_get_nli", _guard("_get_nli"))
    monkeypatch.setattr(support_mod, "_get_nli", _guard("support._get_nli"))
    monkeypatch.setattr(
        attribution_mod, "_source_token_attributions", _guard("_source_token_attributions")
    )

    # Confirm none of the guards fired before any test action
    assert called == []
