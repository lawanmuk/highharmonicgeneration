import numpy as np
import pytest

import hhg

W0 = hhg.OMEGA_PUMP


def rotating_trace(order=3, sense=+1, ellipticity=1.0, angle=0.0, t_max=2000.0, dt=0.08):
    """A Gaussian burst at harmonic ``order`` rotating counter-clockwise (sense=+1) or
    clockwise (sense=-1), with minor/major axis ratio ``ellipticity``, major axis at ``angle``."""
    time = np.arange(0.0, t_max, dt)
    envelope = np.exp(-(((time - t_max / 2) / 300.0) ** 2))
    phase = order * W0 * time
    major, minor = np.cos(phase), sense * ellipticity * np.sin(phase)
    current = np.zeros((time.size, 3))
    current[:, 0] = envelope * (np.cos(angle) * major - np.sin(angle) * minor)
    current[:, 1] = envelope * (np.sin(angle) * major + np.cos(angle) * minor)
    return hhg.CurrentTrace(time, current)


def peak_values(trace):
    omega, s_plus, s_minus = hhg.circular_components(trace)
    i = np.argmax(s_plus + s_minus)
    return omega[i], s_plus[i], s_minus[i]


def test_counter_clockwise_field_is_all_plus():
    omega, s_plus, s_minus = peak_values(rotating_trace(sense=+1))
    assert omega / W0 == pytest.approx(3.0, abs=0.05)
    assert s_minus < 1e-10 * s_plus


def test_clockwise_field_is_all_minus():
    _, s_plus, s_minus = peak_values(rotating_trace(sense=-1))
    assert s_plus < 1e-10 * s_minus


def test_linear_field_splits_equally():
    _, s_plus, s_minus = peak_values(rotating_trace(ellipticity=0.0, angle=0.4))
    assert s_plus == pytest.approx(s_minus, rel=1e-8)


def test_elliptical_field_gives_expected_ratio():
    # For minor/major ratio e, |J_-|/|J_+| = (1 - e) / (1 + e).
    e = 0.5
    _, s_plus, s_minus = peak_values(rotating_trace(ellipticity=e))
    assert s_minus / s_plus == pytest.approx(((1 - e) / (1 + e)) ** 2, rel=1e-6)


def test_parts_add_up_to_in_plane_spectrum():
    trace = rotating_trace(ellipticity=0.3, angle=1.1)
    omega, s_plus, s_minus = hhg.circular_components(trace)
    _, total = hhg.hhg_spectrum(trace)
    np.testing.assert_allclose(s_plus + s_minus, total, rtol=1e-10, atol=1e-14 * total.max())


def test_rotating_the_pattern_does_not_change_the_parts():
    _, plus_a, minus_a = hhg.circular_components(rotating_trace(ellipticity=0.3, angle=0.0))
    _, plus_b, minus_b = hhg.circular_components(rotating_trace(ellipticity=0.3, angle=0.9))
    np.testing.assert_allclose(plus_b, plus_a, rtol=1e-8, atol=1e-12 * plus_a.max())
    np.testing.assert_allclose(minus_b, minus_a, rtol=1e-8, atol=1e-12 * plus_a.max())


def test_direct_and_fft_transforms_agree():
    trace = rotating_trace(ellipticity=0.3)
    omega_fft, plus_fft, _ = hhg.circular_components(trace)
    omega = np.linspace(2.0, 4.0, 50) * W0
    _, plus_direct, _ = hhg.circular_components(trace, omega)
    np.testing.assert_allclose(
        np.interp(omega, omega_fft, plus_fft), plus_direct, atol=5e-3 * plus_direct.max()
    )
