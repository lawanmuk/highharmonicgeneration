"""The probe's response, separated from the pump background (bcp_pumpprobe minus bcp_I=1.5e12).

The probe is a vector-potential step, A(t) = 8e-4 a.u. for t >= 0, which acts as a delta-like
kick in the field. The pump-probe file starts at the same time as the pump-only run, so the
two currents are equal until the kick and differ afterwards. The difference is the probe's
response. These tests check where that response starts and how big it is.
"""

from pathlib import Path

import numpy as np
import pytest

import hhg

DATA = Path(__file__).resolve().parent.parent / "HHG_datasets"
needs_data = pytest.mark.skipif(not DATA.exists(), reason="HHG_datasets not available")

PUMP_PROBE = "bcp_pumpprobe_total_current.dat"
PUMP_ONLY = "bcp_I=1.5e12total_current.dat"

PROBE_KICK = 8e-4  # vector-potential amplitude of the probe, atomic units
PUMP_INTENSITY = 1.5e12  # W/cm^2
AU_INTENSITY = 3.51e16  # W/cm^2 per (a.u. of field)^2
ONSET_AU = 393.04  # first output sample after the kick, file time (9.51 fs)


def difference():
    probe = hhg.load_current(DATA / PUMP_PROBE)
    pump = hhg.load_current(DATA / PUMP_ONLY)
    n = min(probe.time.size, pump.time.size)
    return probe.time[:n], probe.current[:n], pump.current[:n]


@needs_data
def test_probe_response_is_zero_before_the_kick():
    time, pp, pu = difference()
    before = time < ONSET_AU - 1.0
    assert before.any()
    assert np.abs(pp[before] - pu[before]).max() < 1e-12 * np.abs(pu).max()


@needs_data
def test_probe_response_starts_at_the_kick():
    time, pp, pu = difference()
    diff = np.abs(pp[:, 0] - pu[:, 0])
    onset = time[np.argmax(diff > 1e-9 * np.abs(pu[:, 0]).max())]
    assert onset == pytest.approx(ONSET_AU, abs=0.08)


@needs_data
def test_probe_response_size_matches_a_linear_kick():
    # Linear estimate: the probe's vector potential over the pump's, with the pump's field
    # amplitude from its intensity (E0 = sqrt(I / 3.51e16)) and A0 = E0 / omega_pump.
    time, pp, pu = difference()
    pump_field = np.sqrt(PUMP_INTENSITY / AU_INTENSITY)
    pump_vector = pump_field / hhg.OMEGA_PUMP
    expected = PROBE_KICK / pump_vector
    observed = np.abs(pp - pu).max() / np.abs(pu).max()
    assert expected / 3 < observed < 3 * expected
