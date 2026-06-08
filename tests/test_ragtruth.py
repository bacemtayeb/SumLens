"""RAGTruth loader tests — fixture JSONL exercising the span->sentence-id mapping."""

import json
from pathlib import Path

from sumlens.eval.ragtruth import _spans_to_sentence_ids, load_split
from sumlens.types import Sentence, Summary


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


def test_load_split_builds_examples(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "source_info.jsonl",
        [
            {"source_id": "a", "task_type": "Summary", "source_info": "Alpha beta. Gamma delta."},
            {"source_id": "b", "task_type": "QA", "source_info": "ignored QA source"},
        ],
    )
    # response "The budget rose. A lie here." -> sum-0000 (0..16), sum-0001 (17..28)
    # label span 17..21 falls in sum-0001 only.
    _write_jsonl(
        tmp_path / "response.jsonl",
        [
            {
                "id": "r1",
                "source_id": "a",
                "split": "test",
                "model": "bart",
                "response": "The budget rose. A lie here.",
                "labels": [{"start": 17, "end": 21, "label_type": "Evident Conflict"}],
            },
            {"id": "r2", "source_id": "a", "split": "train", "response": "Other.", "labels": []},
            {"id": "r3", "source_id": "b", "split": "test", "response": "QA resp.", "labels": []},
        ],
    )

    examples = load_split("test", task="Summary", data_dir=tmp_path)

    assert len(examples) == 1  # train split and QA task filtered out
    document, summary, hallucinated = examples[0]
    assert document.id == "a"
    assert document.raw_text == "Alpha beta. Gamma delta."
    assert summary.id == "r1"
    assert summary.document_id == "a"
    assert summary.model_name == "bart"
    assert [s.id for s in summary.sentences] == ["sum-0000", "sum-0001"]
    assert hallucinated == ["sum-0001"]


def test_spans_to_sentence_ids_overlap() -> None:
    summary = Summary(
        id="r1",
        document_id="a",
        text="First sentence. Second sentence. Third one.",
        sentences=[
            Sentence(id="sum-0000", text="First sentence.", char_start=0, char_end=15),
            Sentence(id="sum-0001", text="Second sentence.", char_start=16, char_end=32),
            Sentence(id="sum-0002", text="Third one.", char_start=33, char_end=43),
        ],
        model_name="m",
    )
    # span 10..20 straddles sum-0000 and sum-0001; span 35..38 hits sum-0002
    assert _spans_to_sentence_ids(summary, [(10, 20), (35, 38)]) == [
        "sum-0000",
        "sum-0001",
        "sum-0002",
    ]
    assert _spans_to_sentence_ids(summary, []) == []
