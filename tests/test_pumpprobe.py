"""Checks on the pump-probe dataset (bcp_pumpprobe), compared with the pump-only run.

The pump-probe file has the same time step and starts at the same time as the pump-only run,
but it stops early, at about t = 2461 atomic units (12 pump cycles) against 5789 for the full
pump-only run. Over the shared time window the two agree to within a fraction of a percent,
which is the size of the probe's effect. These tests fix those facts so that the probe
analysis in the next steps starts from a checked input.
"""

from pathlib import Path

import numpy as np
import pytest

import hhg

DATA = Path(__file__).resolve().parent.parent / "HHG_datasets"
needs_data = pytest.mark.skipif(not DATA.exists(), reason="HHG_datasets not available")

PUMP_PROBE = "bcp_pumpprobe_total_current.dat"
PUMP_ONLY = "bcp_I=1.5e12total_current.dat"


def load(name):
    return hhg.load_current(DATA / name)


def overlap(a, b):
    n = min(a.time.size, b.time.size)
    return a.time[:n], a.current[:n], b.current[:n]


@needs_data
def test_pump_probe_loads_on_a_uniform_grid():
    trace = load(PUMP_PROBE)
    assert trace.dt == pytest.approx(0.08)
    assert np.isfinite(trace.current).all()


@needs_data
def test_pump_probe_is_shorter_than_pump_only():
    probe, pump = load(PUMP_PROBE), load(PUMP_ONLY)
    assert probe.time[-1] == pytest.approx(2461.2, abs=0.1)
    assert pump.time[-1] > 2 * probe.time[-1]


@needs_data
def test_pump_probe_shares_the_pump_time_axis():
    probe, pump = load(PUMP_PROBE), load(PUMP_ONLY)
    time, _, _ = overlap(probe, pump)
    np.testing.assert_allclose(time, pump.time[: time.size])


@needs_data
def test_pump_probe_agrees_with_pump_only_over_the_shared_window():
    # The probe changes the current by well under 1 % of the pump's peak.
    probe, pump = load(PUMP_PROBE), load(PUMP_ONLY)
    _, pp, pu = overlap(probe, pump)
    difference = np.abs(pp - pu).max()
    assert difference < 1e-2 * np.abs(pu).max()


@needs_data
def test_pump_probe_difference_is_weak_at_low_harmonic_orders():
    # Compare the spectra of the two runs' difference and of the pump, at orders 1 to 5.
    probe, pump = load(PUMP_PROBE), load(PUMP_ONLY)
    _, pp, pu = overlap(probe, pump)
    window = np.hanning(pp.shape[0])
    omega = np.fft.rfftfreq(pp.shape[0], d=probe.dt) * 2 * np.pi
    difference = np.abs(np.fft.rfft((pp - pu)[:, 0] * window))
    reference = np.abs(np.fft.rfft(pu[:, 0] * window))
    for order in range(1, 6):
        k = np.argmin(np.abs(omega - order * hhg.OMEGA_PUMP))
        ratio = difference[k - 3 : k + 4].max() / reference[k - 3 : k + 4].max()
        assert ratio < 1e-2
