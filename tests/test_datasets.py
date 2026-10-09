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
@pytest.mark.parametrize("name", ["bcp_I=1.5e12", "bcp_I=1.5e12_run2"])
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


FIELD_RATIO_5_TO_1P5 = np.sqrt(5.0 / 1.5)


@needs_data
@pytest.mark.parametrize("pulse", ["lp", "cp"])
def test_intensity_labels_match_linear_response(pulse):
    # Early in the pulse the current is linear in the field, so 5e12 vs 1.5e12 W/cm^2
    # must give a current ratio of sqrt(5 / 1.5) = 1.826.
    weak = hhg.load_current(DATA / f"{pulse}_I=1.5e12total_current.dat")
    strong = hhg.load_current(DATA / f"{pulse}_I=5e12total_current.dat")
    for t_max in (20.0, 50.0, 100.0):
        ratio = hhg.early_response_ratio(weak, strong, t_max)
        assert ratio == pytest.approx(FIELD_RATIO_5_TO_1P5, rel=5e-3)


@needs_data
def test_second_bicircular_run_is_also_1p5e12():
    # bcp_I=1.5e12_run2 was originally saved as "bcp_I=5e12". It does not scale like a
    # 5e12 run, and its spectrum matches the 1.5e12 run, which confirms the corrected label.
    first = hhg.load_current(DATA / "bcp_I=1.5e12total_current.dat")
    second = hhg.load_current(DATA / "bcp_I=1.5e12_run2total_current.dat")
    ratio = hhg.early_response_ratio(first, second, 20.0)
    assert abs(ratio - FIELD_RATIO_5_TO_1P5) > 0.5
    omega, s_first = hhg.hhg_spectrum(first)
    _, s_second = hhg.hhg_spectrum(second)
    band = (omega > 0.5 * hhg.OMEGA_PUMP) & (omega < 20 * hhg.OMEGA_PUMP)
    assert np.max(np.abs(s_first[band] - s_second[band])) < 5e-3 * s_first[band].max()


BICIRCULAR_RUNS = ["bcp_I=1.5e12", "bcp_I=1.5e12_run2", "bcp_pumpprobe_"]
PLUS_ONE = [1, 4, 7, 10, 13]  # 3n + 1: follow the fundamental's rotation
MINUS_ONE = [2, 5, 8, 11, 14]  # 3n - 1: follow the second harmonic's rotation
STRONG = [1, 2, 4, 5, 8, 11, 13, 14]  # orders that come out almost fully circular


def polarization(name, orders):
    return hhg.harmonic_polarization(hhg.load_current(DATA / f"{name}total_current.dat"), orders)


@needs_data
@pytest.mark.parametrize("name", BICIRCULAR_RUNS)
def test_bicircular_3n_plus_and_minus_1_rotate_in_opposite_directions(name):
    # In a bicircular field, 3n+1 harmonics carry the helicity of the fundamental and
    # 3n-1 harmonics the opposite one (here: 3n+1 clockwise, 3n-1 counter-clockwise).
    assert np.all(polarization(name, PLUS_ONE).helicity < 0)
    assert np.all(polarization(name, MINUS_ONE).helicity > 0)


@needs_data
@pytest.mark.parametrize("name", BICIRCULAR_RUNS)
def test_strong_bicircular_harmonics_are_nearly_circular(name):
    assert np.all(np.abs(polarization(name, STRONG).helicity) > 0.9)


@needs_data
@pytest.mark.parametrize(
    "name", ["lp_I=1e11", "lp_I=1e12", "lp_I=1.5e12", "lp_I=5e12", "cp_I=1.5e12", "cp_I=5e12"]
)
def test_linear_and_collinear_harmonics_have_no_helicity(name):
    # Below 0.01 for every run except lp_I=1e11, whose high orders sit about seven decades
    # under the fundamental, close to the numerical noise floor (largest |h| there: 0.03).
    assert np.all(np.abs(polarization(name, range(1, 15)).helicity) < 0.05)


CYCLE = 2 * np.pi / hhg.OMEGA_PUMP
PULSE = (1.5 * CYCLE, 4.5 * CYCLE)  # the strong part of the pump pulse


def timing(name, orders):
    trace = hhg.load_current(DATA / f"{name}total_current.dat")
    return hhg.cycle_profile(trace, orders, *PULSE)


@needs_data
@pytest.mark.parametrize("name", ["bcp_I=1.5e12", "bcp_I=1.5e12_run2"])
def test_bicircular_harmonics_are_emitted_three_times_per_cycle(name):
    # The bicircular field is a three-leaved pattern that repeats every third of a cycle.
    # Orders 6 to 12 are left out: there the steady band-gap emission blurs the pattern.
    profile = timing(name, [4, 5, 13, 14])
    assert np.all(profile.bursts() == 3)
    assert np.all(profile.modulation() > 0.3)


@needs_data
def test_bicircular_emission_times_repeat_between_runs():
    first = timing("bcp_I=1.5e12", [4, 5, 13, 14]).burst_phase()
    second = timing("bcp_I=1.5e12_run2", [4, 5, 13, 14]).burst_phase()
    np.testing.assert_allclose(second, first, atol=0.02)


@needs_data
@pytest.mark.parametrize("name", ["cp_I=1.5e12", "cp_I=5e12"])
def test_collinear_high_harmonics_are_emitted_once_per_cycle(name):
    # Adding 2 w_0 to w_0 breaks the symmetry between the two halves of the cycle,
    # so the high orders come out as a single burst per cycle.
    assert np.all(timing(name, [13, 14, 16]).bursts() == 1)
