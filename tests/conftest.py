"""Ensure NLTK Punkt data is present before ingest tests run (offline-safe once cached)."""

import nltk


def _ensure_punkt() -> None:
    for resource, package in [
        ("tokenizers/punkt/english.pickle", "punkt"),
        ("tokenizers/punkt_tab/english/", "punkt_tab"),
    ]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)


_ensure_punkt()
