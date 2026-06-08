"""RAGTruth loader — primary benchmark.

Reads RAGTruth's `source_info.jsonl` + `response.jsonl`, keeps the requested task
(default "Summary"), and builds typed examples. RAGTruth annotates hallucinations
as character-offset spans in the response; we map each span to the summary
sentence ids it overlaps, using our own punkt tokeniser. That mapping is a real
eval choice — document it in the report.

Field names follow RAGTruth's published schema; verify them against the actual
files on first real run.
"""

import json
from pathlib import Path
from typing import Any, Literal

from sumlens.ingest import split_sentences
from sumlens.types import Document, Summary

_DEFAULT_DATA_DIR = Path("data/ragtruth")


def load_split(
    split: Literal["train", "test"],
    task: str = "Summary",
    data_dir: Path = _DEFAULT_DATA_DIR,
) -> list[tuple[Document, Summary, list[str]]]:
    sources = {r["source_id"]: r for r in _read_jsonl(data_dir / "source_info.jsonl")}
    examples: list[tuple[Document, Summary, list[str]]] = []
    for response in _read_jsonl(data_dir / "response.jsonl"):
        if response.get("split") != split:
            continue
        source = sources.get(response["source_id"])
        if source is None or source.get("task_type") != task:
            continue
        examples.append(_build_example(source, response))
    return examples


def _build_example(
    source: dict[str, Any], response: dict[str, Any]
) -> tuple[Document, Summary, list[str]]:
    document = _document_from_source(source)
    summary = _summary_from_response(response)
    spans = [(label["start"], label["end"]) for label in response.get("labels", [])]
    return document, summary, _spans_to_sentence_ids(summary, spans)


def _document_from_source(source: dict[str, Any]) -> Document:
    text = _source_text(source)
    source_id = str(source["source_id"])
    return Document(
        id=source_id,
        raw_text=text,
        sentences=split_sentences(text, "src"),
        source="text",
        meta={"source_id": source_id},
    )


def _summary_from_response(response: dict[str, Any]) -> Summary:
    text = response["response"]
    return Summary(
        id=str(response["id"]),
        document_id=str(response["source_id"]),
        text=text,
        sentences=split_sentences(text, "sum"),
        model_name=str(response.get("model", "unknown")),
    )


def _spans_to_sentence_ids(summary: Summary, spans: list[tuple[int, int]]) -> list[str]:
    """A summary sentence is hallucinated if any labelled span overlaps its char range."""
    return [
        sentence.id
        for sentence in summary.sentences
        if any(_overlaps(sentence.char_start, sentence.char_end, s, e) for s, e in spans)
    ]


def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and b0 < a1


def _source_text(source: dict[str, Any]) -> str:
    info = source.get("source_info", source.get("source", ""))
    if isinstance(info, str):
        return info
    if isinstance(info, dict):
        for key in ("passages", "text", "document"):
            value = info.get(key)
            if isinstance(value, str):
                return value
        return json.dumps(info)
    return str(info)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
