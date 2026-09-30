"""Physics regression tests on the TDDFT datasets in HHG_datasets/ (skipped if absent)."""

from pathlib import Path

import numpy as np
import pytest

import hhg

DATA = Path(__file__).resolve().parent.parent / "HHG_datasets"
ALLOWED = [1, 2, 4, 5, 7, 8, 10, 11]  # 3n +- 1
FORBIDDEN = [3, 9, 12]  # 3n; 6 is left out because it overlaps the band-gap emission

needs_data = pytest.mark.skipif(not DATA.exists(), reason="HHG_datasets not available")


def contrast(name):
    omega, spectrum = hhg.hhg_spectrum(hhg.load_current(DATA / f"{name}total_current.dat"))
    allowed = hhg.harmonic_yields(omega, spectrum, hhg.OMEGA_PUMP, ALLOWED)
    forbidden = hhg.harmonic_yields(omega, spectrum, hhg.OMEGA_PUMP, FORBIDDEN)
    return hhg.selection_rule_contrast(allowed, forbidden)


@needs_data
@pytest.mark.parametrize("name", ["bcp_I=1.5e12", "bcp_I=5e12"])
def test_bicircular_obeys_3n_plus_minus_1_rule(name):
    assert contrast(name) < 1e-2


@needs_data
def test_collinear_has_no_such_rule():
    assert contrast("cp_I=1.5e12") > 5e-2


@needs_data
def test_all_current_files_load_on_uniform_grids():
    files = []
    for path in sorted(DATA.glob("*total_current.dat")):
        try:
            hhg.parse_run_name(path.name)
        except ValueError:
            continue  # not a raw current file (e.g. processed spectra)
        files.append(path)
    assert len(files) == 9
    for path in files:
        trace = hhg.load_current(path)
        assert np.isfinite(trace.current).all()
        assert trace.dt == pytest.approx(0.08)


@needs_data
def test_processed_spectrum_file_is_not_mistaken_for_a_current():
    processed = DATA / "HHG_cp_I=1.5e12total_current.dat"
    if not processed.exists():
        pytest.skip("processed file has been moved")
    with pytest.raises(ValueError):
        hhg.load_current(processed)
