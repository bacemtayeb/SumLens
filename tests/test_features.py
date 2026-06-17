"""Feature-row assembly tests — pure, no models."""

from sumlens.eval.features import feature_rows
from sumlens.types import Claim, Sentence, Summary


def _summary() -> Summary:
    return Summary(
        id="r1",
        document_id="a",
        text="Grounded one. Hallucinated two.",
        sentences=[
            Sentence(id="sum-0000", text="Grounded one.", char_start=0, char_end=13),
            Sentence(id="sum-0001", text="Hallucinated two.", char_start=14, char_end=31),
        ],
        model_name="m",
    )


def test_feature_rows_labels_and_missing_signals() -> None:
    classifier_out = {"sum-0000": (0.1, []), "sum-0001": (0.9, [(0, 4)])}
    failed = Claim(id="c", sentence_id="sum-0001", text="x")
    nli_out = {"sum-0000": (0.8, []), "sum-0001": (0.2, [failed])}
    # support attribution: (attr_conc, attr_loo, top_source_ids); sum-0000 absent
    support_out = {"sum-0001": (0.3, 0.15, ["src-0000"])}

    rows = feature_rows(_summary(), ["sum-0001"], classifier_out, nli_out, support_out)

    assert rows == [
        {
            "summary_id": "r1",
            "sentence_id": "sum-0000",
            "classifier": 0.1,
            "nli": 0.8,
            "attr_conc": None,  # C did not run for this sentence
            "attr_loo": None,
            "grounded": 1,
        },
        {
            "summary_id": "r1",
            "sentence_id": "sum-0001",
            "classifier": 0.9,
            "nli": 0.2,
            "attr_conc": 0.3,
            "attr_loo": 0.15,
            "grounded": 0,  # marked hallucinated in gold
        },
    ]
