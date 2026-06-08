"""Round-trip every core type through JSON and back.

If `model_validate_json(model_dump_json(x)) == x` holds for every type, the JSON
contract is stable: nothing is lost or coerced across a serialise/deserialise cycle.
"""

import pytest
from pydantic import BaseModel

from sumlens.types import (
    AnalysisConfig,
    AnalysisResult,
    Claim,
    Document,
    Evidence,
    Sentence,
    SentenceVerdict,
    SignalScores,
    Summary,
)

_sentence = Sentence(id="src-0000", text="The bill passed.", char_start=0, char_end=16)

_document = Document(
    id="doc-1",
    raw_text="The bill passed. It allocates funds.",
    sentences=[
        _sentence,
        Sentence(id="src-0001", text="It allocates funds.", char_start=17, char_end=36),
    ],
    source="text",
    meta={"filename": "report.pdf", "word_count": 6},
)

_claim = Claim(id="sum-0003-claim-1", sentence_id="sum-0003", text="The bill allocates €2.4B.")

_signal_scores = SignalScores(classifier=0.81, nli=None, attribution=0.42)

_evidence = Evidence(
    failed_claims=[_claim],
    top_source_sentence_ids=["src-0000", "src-0001"],
    classifier_token_spans=[(0, 4), (5, 9)],
    notes=["claim not entailed by source"],
)

_summary = Summary(
    id="sum-1",
    document_id="doc-1",
    text="The bill allocates €2.4B.",
    sentences=[
        Sentence(id="sum-0000", text="The bill allocates €2.4B.", char_start=0, char_end=25),
    ],
    model_name="facebook/bart-large-cnn",
)

_verdict = SentenceVerdict(
    sentence_id="sum-0000",
    fused_score=0.27,
    label="hallucinated",
    signals=_signal_scores,
    evidence=_evidence,
)

_config = AnalysisConfig()

_result = AnalysisResult(
    document=_document,
    summary=_summary,
    verdicts=[_verdict],
    config=_config,
    timings_ms={"summarise": 1200, "attribute": 3400},
)

_INSTANCES = [
    _sentence,
    _document,
    _claim,
    _signal_scores,
    _evidence,
    _verdict,
    _summary,
    _config,
    _result,
]


@pytest.mark.parametrize("instance", _INSTANCES, ids=lambda x: type(x).__name__)
def test_json_round_trip(instance: BaseModel) -> None:
    restored = type(instance).model_validate_json(instance.model_dump_json())
    assert restored == instance
