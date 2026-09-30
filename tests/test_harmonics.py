import numpy as np
import pytest
from test_spectrum import gaussian_tone, trace_with

import hhg

W0 = hhg.OMEGA_PUMP


def test_yields_pick_out_present_orders():
    trace = trace_with([gaussian_tone(1), gaussian_tone(2), None])
    omega, spectrum = hhg.hhg_spectrum(trace)
    present = hhg.harmonic_yields(omega, spectrum, W0, [1, 2])
    absent = hhg.harmonic_yields(omega, spectrum, W0, [3, 4])
    assert absent.max() < 1e-6 * present.min()


def test_selection_rule_contrast():
    assert hhg.selection_rule_contrast(np.array([1.0, 1.0]), np.array([0.01])) == pytest.approx(
        0.01
    )


def test_invalid_half_width_and_out_of_range_order():
    omega = np.linspace(0, 5 * W0, 500)
    spectrum = np.ones_like(omega)
    with pytest.raises(ValueError):
        hhg.harmonic_yields(omega, spectrum, W0, [1], half_width=0.8)
    with pytest.raises(ValueError, match="outside"):
        hhg.harmonic_yields(omega, spectrum, W0, [20])
