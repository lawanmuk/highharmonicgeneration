import csv
from pathlib import Path

import pytest

import hhg

DATA = Path(__file__).resolve().parent.parent / "HHG_datasets"


@pytest.mark.skipif(not DATA.exists(), reason="HHG_datasets not available")
def test_make_figures_writes_all_outputs(tmp_path):
    written = hhg.make_figures(DATA, tmp_path)
    names = {p.name for p in written}
    assert {
        "linear_intensity_scan.png",
        "collinear_intensity_scan.png",
        "pump_comparison_I=1.5e12.png",
        "bicircular_selection_rule_I=1.5e12.png",
        "harmonic_yields.csv",
    } <= names
    for path in written:
        assert path.stat().st_size > 1000
    rows = list(csv.DictReader((tmp_path / "harmonic_yields.csv").open()))
    assert len(rows) == 9
    assert {r["pulse"] for r in rows} == {"linear", "collinear", "bicircular"}


def test_make_figures_needs_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        hhg.make_figures(tmp_path, tmp_path / "out")
