# SumLens

![CI](https://github.com/bacemtayeb/SumLens/actions/workflows/ci.yml/badge.svg)

Explainability dashboard for AI-generated summaries. Upload a document, get an
abstractive summary, and see — sentence by sentence — how strongly each summary
sentence is grounded in the source, with low-confidence sentences flagged as
potential hallucinations.

> Course project · MICS · Principles of Software Development (S2 2025–26).

## Status

Scaffold. Features land via pull requests — see the open issues and `docs/`.

## Documentation

- [`docs/requirements.md`](docs/requirements.md) — functional / non-functional requirements, MoSCoW, user stories, traceability.
- [`docs/data-model.md`](docs/data-model.md) — canonical data types.
- [`docs/research-plan.md`](docs/research-plan.md) — signals, fusion, evaluation methodology.

## Development

Requires Python 3.11+.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m nltk.downloader punkt punkt_tab
```

### Quality gate (CI enforces this on every PR)

```bash
ruff check . && mypy sumlens tests && pytest -q --cov=sumlens --cov-fail-under=70
```

Lint (ruff), type-check (mypy, strict), and tests with a ≥70% coverage gate must
pass before any PR is merged to `main`.

## License

TBD.
