"""Ingestion — PDF or raw text into a `Document`.

PDF text is extracted with pdfplumber. Text is cleaned and paragraph-segmented on
blank lines, then split into sentences with NLTK Punkt. Sentence ids are stable
`src-0000`, `src-0001`, ... and carry char offsets into `Document.raw_text`.
"""

import re
from pathlib import Path
from typing import Any

import nltk
import pdfplumber

from sumlens.types import Document, Sentence

_BLANK_LINE = re.compile(r"\n\s*\n")
_WHITESPACE = re.compile(r"\s+")


def load_pdf(path: Path) -> Document:
    with pdfplumber.open(path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    raw_text = _clean("\n\n".join(pages))
    meta: dict[str, Any] = {"filename": path.name, "word_count": _word_count(raw_text)}
    return Document(
        id=path.stem,
        raw_text=raw_text,
        sentences=split_sentences(raw_text, "src"),
        source="pdf",
        meta=meta,
    )


def load_text(text: str) -> Document:
    raw_text = _clean(text)
    meta: dict[str, Any] = {"word_count": _word_count(raw_text)}
    return Document(
        id="text",
        raw_text=raw_text,
        sentences=split_sentences(raw_text, "src"),
        source="text",
        meta=meta,
    )


def _clean(text: str) -> str:
    """Collapse each blank-line-delimited paragraph onto one line; join with \\n\\n."""
    paragraphs = []
    for para in _BLANK_LINE.split(text):
        collapsed = _WHITESPACE.sub(" ", para).strip()
        if collapsed:
            paragraphs.append(collapsed)
    return "\n\n".join(paragraphs)


def split_sentences(text: str, id_prefix: str) -> list[Sentence]:
    """NLTK Punkt sentence split with char offsets; ids `{id_prefix}-0000`, ..."""
    if not text:
        return []
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    return [
        Sentence(
            id=f"{id_prefix}-{i:04d}",
            text=text[start:end],
            char_start=start,
            char_end=end,
        )
        for i, (start, end) in enumerate(tokenizer.span_tokenize(text))
    ]


def _word_count(raw_text: str) -> int:
    return len(raw_text.split())
