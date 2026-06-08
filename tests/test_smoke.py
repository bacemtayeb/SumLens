"""Smoke test: the package imports.

Keeps CI (and the coverage gate) green on the initial scaffold, before feature
modules land via their own PRs. Removed in the feat/types PR once real tests exist.
"""

import sumlens


def test_package_imports() -> None:
    assert sumlens.__name__ == "sumlens"
