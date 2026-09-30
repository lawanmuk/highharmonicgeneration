import numpy as np
import pytest

import hhg

W0 = hhg.OMEGA_PUMP


def trace_with(components, t_max=2000.0, dt=0.08):
    time = np.arange(0.0, t_max, dt)
    current = np.stack([f(time) if f else np.zeros_like(time) for f in components], axis=1)
    return hhg.CurrentTrace(time, current)


def gaussian_tone(order, center=1000.0, width=300.0):
    return lambda t: np.cos(order * W0 * t) * np.exp(-(((t - center) / width) ** 2))


def test_window_goes_from_one_to_zero_smoothly():
    time = np.linspace(0.0, 100.0, 1001)
    window = hhg.smooth_step_window(time)
    assert window[0] == pytest.approx(1.0)
    assert window[-1] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(window) <= 1e-12)  # never rises (the original bug rose to 2)
    assert np.all((window >= 0) & (window <= 1))


def test_window_ignores_time_offset():
    time = np.linspace(0.0, 100.0, 101)
    np.testing.assert_allclose(hhg.smooth_step_window(time + 50.0), hhg.smooth_step_window(time))


def test_spectrum_peaks_at_the_driven_harmonic():
    omega, spectrum = hhg.hhg_spectrum(trace_with([gaussian_tone(3), None, None]))
    assert omega[np.argmax(spectrum)] / W0 == pytest.approx(3.0, abs=0.05)


def test_fft_matches_direct_transform():
    trace = trace_with([gaussian_tone(3), gaussian_tone(5), None])
    omega_fft, s_fft = hhg.hhg_spectrum(trace)
    omega = np.linspace(0.5, 8.0, 200) * W0
    _, s_direct = hhg.hhg_spectrum(trace, omega)
    np.testing.assert_allclose(
        np.interp(omega, omega_fft, s_fft), s_direct, atol=5e-3 * s_direct.max()
    )


def test_all_components_contribute():
    only_x = trace_with([gaussian_tone(3), None, None])
    x_and_y = trace_with([gaussian_tone(3), gaussian_tone(3), None])
    _, s1 = hhg.hhg_spectrum(only_x)
    _, s2 = hhg.hhg_spectrum(x_and_y)
    np.testing.assert_allclose(s2, 2 * s1, rtol=1e-10, atol=1e-12 * s1.max())


def test_spectrum_is_rotation_invariant():
    base = trace_with([gaussian_tone(3), gaussian_tone(5), gaussian_tone(7)])
    angle = 0.7
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    rotated = hhg.CurrentTrace(base.time, base.current @ rotation.T)
    _, s1 = hhg.hhg_spectrum(base)
    _, s2 = hhg.hhg_spectrum(rotated)
    np.testing.assert_allclose(s2, s1, rtol=1e-8, atol=1e-12 * s1.max())


def test_fourier_direct_single_component_shape():
    time = np.linspace(0, 10, 101)
    assert hhg.fourier_direct(time, np.sin(time), np.array([0.5, 1.0])).shape == (2,)


def test_invalid_pad_factor():
    with pytest.raises(ValueError):
        hhg.fourier_fft(np.linspace(0, 1, 10), np.zeros(10), pad_factor=0)
