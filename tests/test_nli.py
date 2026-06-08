"""NLI (signal B) tests — NLI model mocked at the `_get_nli` boundary."""

import pytest

from sumlens.signals import nli as nli_mod
from sumlens.signals.nli import _split_clauses, entail, extract_claims
from sumlens.types import AnalysisConfig, Claim, Document, Sentence, Summary


def _summary(text: str) -> Summary:
    return Summary(
        id="doc-1-summary",
        document_id="doc-1",
        text=text,
        sentences=[Sentence(id="sum-0000", text=text, char_start=0, char_end=len(text))],
        model_name="m",
    )


def test_extract_claims_splits_on_conjunction() -> None:
    claims = extract_claims(_summary("The bill passed and funds were allocated."))
    assert [(c.id, c.text) for c in claims] == [
        ("sum-0000-claim-1", "The bill passed"),
        ("sum-0000-claim-2", "funds were allocated."),
    ]
    assert all(c.sentence_id == "sum-0000" for c in claims)


def test_extract_claims_single_clause() -> None:
    claims = extract_claims(_summary("The bill passed today."))
    assert [(c.id, c.text) for c in claims] == [("sum-0000-claim-1", "The bill passed today.")]


def test_split_clauses_comma_and() -> None:
    assert _split_clauses("X happened, and Y followed") == ["X happened", "Y followed"]
    assert _split_clauses("only one") == ["only one"]


# Entailment lookup: (premise, hypothesis) -> entailment prob.
_TABLE = {
    ("Src A.", "The bill passed"): 0.8,
    ("Src B.", "The bill passed"): 0.4,
    ("Src A.", "funds were allocated"): 0.2,
    ("Src B.", "funds were allocated"): 0.3,
}


class _FakeNLI:
    def __call__(
        self, pairs: list[dict[str, str]], top_k: object = None, batch_size: object = None
    ) -> list[list[dict[str, object]]]:
        out: list[list[dict[str, object]]] = []
        for pair in pairs:
            ent = _TABLE[(pair["text"], pair["text_pair"])]
            out.append(
                [
                    {"label": "entailment", "score": ent},
                    {"label": "neutral", "score": 1.0 - ent},
                    {"label": "contradiction", "score": 0.0},
                ]
            )
        return out


def _document() -> Document:
    return Document(
        id="doc-1",
        raw_text="Src A. Src B.",
        sentences=[
            Sentence(id="src-0000", text="Src A.", char_start=0, char_end=6),
            Sentence(id="src-0001", text="Src B.", char_start=7, char_end=13),
        ],
        source="text",
    )


def test_entail_max_over_sources_min_over_claims(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nli_mod, "_get_nli", lambda model_name: _FakeNLI())
    claims = [
        Claim(id="sum-0000-claim-1", sentence_id="sum-0000", text="The bill passed"),
        Claim(id="sum-0000-claim-2", sentence_id="sum-0000", text="funds were allocated"),
    ]

    result = entail(claims, _document(), AnalysisConfig())

    score, failed = result["sum-0000"]
    # claim-1 max-over-sources = 0.8, claim-2 = 0.3 -> sentence min = 0.3
    assert score == pytest.approx(0.3)
    # only claim-2 (0.3) is below the 0.5 fail threshold
    assert [c.id for c in failed] == ["sum-0000-claim-2"]


def test_entail_empty_claims(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nli_mod, "_get_nli", lambda model_name: _FakeNLI())
    assert entail([], _document(), AnalysisConfig()) == {}
