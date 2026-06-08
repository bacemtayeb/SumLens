"""Ensure NLTK Punkt data is present before ingest tests run (offline-safe once cached)."""

import nltk


def _ensure_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt/english.pickle")
    except LookupError:
        nltk.download("punkt", quiet=True)


_ensure_punkt()
