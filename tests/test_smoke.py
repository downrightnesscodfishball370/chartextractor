from __future__ import annotations

import sys
from pathlib import Path

# pylint: disable=import-error

# Allow running tests without installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_package_imports() -> None:
    import chart_extractor

    assert chart_extractor is not None


def test_extract_curve_is_exposed() -> None:
    from chart_extractor import extract_curve

    assert callable(extract_curve)
