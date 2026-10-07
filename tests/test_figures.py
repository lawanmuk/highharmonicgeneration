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
        "bicircular_polarization_I=1.5e12.png",
        "spectrogram_I=1.5e12.png",
        "harmonic_yields.csv",
        "harmonic_polarization.csv",
    } <= names
    for path in written:
        assert path.stat().st_size > 1000
    rows = list(csv.DictReader((tmp_path / "harmonic_yields.csv").open()))
    assert len(rows) == 9
    assert {r["pulse"] for r in rows} == {"linear", "collinear", "bicircular"}
    helicity = {
        r["run"]: r for r in csv.DictReader((tmp_path / "harmonic_polarization.csv").open())
    }
    assert len(helicity) == 9
    assert float(helicity["bcp_I=1.5e12"]["H4"]) < -0.9
    assert float(helicity["bcp_I=1.5e12"]["H5"]) > 0.9


def test_make_figures_needs_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        hhg.make_figures(tmp_path, tmp_path / "out")
