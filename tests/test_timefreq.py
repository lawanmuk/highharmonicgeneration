import numpy as np
import pytest

import hhg

W0 = hhg.OMEGA_PUMP
DT = 0.08
TIME = np.arange(0.0, 3000.0, DT)


def tone(order, time=TIME):
    return np.cos(order * W0 * time)


def burst(order, center, width, time=TIME):
    return np.cos(order * W0 * time) * np.exp(-(((time - center) / width) ** 2))


def test_tone_peaks_at_its_frequency_at_every_time():
    centers = np.linspace(500.0, 2500.0, 5)
    omega, g = hhg.gabor_transform(TIME, tone(5), sigma=80.0, centers=centers)
    peaks = omega[np.argmax(np.abs(g), axis=1)] / W0
    np.testing.assert_allclose(peaks, 5.0, atol=0.05)


def test_burst_is_found_at_its_time():
    centers = np.arange(0.0, 3000.0, 10.0)
    omega, g = hhg.gabor_transform(TIME, burst(9, center=1700.0, width=60.0), 40.0, centers)
    k = np.argmin(np.abs(omega - 9 * W0))
    assert centers[np.argmax(np.abs(g[:, k]))] == pytest.approx(1700.0, abs=10.0)


@pytest.mark.parametrize("sigma", [40.0, 80.0, 160.0])
def test_frequency_resolution_is_set_by_window_width(sigma):
    # |G|^2 of a pure tone is a Gaussian in w with FWHM 2 sqrt(ln 2) / sigma.
    omega, g = hhg.gabor_transform(TIME, tone(5), sigma, [1500.0], pad_factor=16)
    power = np.abs(g[0]) ** 2
    above = omega[power >= power.max() / 2]
    fwhm = above[-1] - above[0]
    assert fwhm == pytest.approx(2 * np.sqrt(np.log(2)) / sigma, rel=0.03)


def test_matches_direct_transform_of_windowed_signal():
    # Pins down the sign and phase convention: same as hhg.fourier_direct.
    signal = burst(3, 1000.0, 300.0) + 0.5 * burst(7, 1200.0, 200.0)
    sigma, center = 70.0, 1123.4
    omega, g = hhg.gabor_transform(TIME, signal, sigma, [center], max_omega=10 * W0)
    window = np.exp(-0.5 * ((TIME - center) / sigma) ** 2)
    direct = hhg.fourier_direct(TIME, signal * window, omega)
    np.testing.assert_allclose(g[0], direct, atol=1e-6 * np.abs(direct).max())


def test_windows_add_up_to_the_full_transform():
    # Gaussians spaced by sigma / 2 sum to sqrt(2 pi) sigma / (sigma / 2) almost exactly,
    # so the sum of all windowed transforms is a scaled copy of the plain transform.
    signal = burst(3, 1000.0, 300.0) + burst(5, 1800.0, 200.0)
    sigma = 60.0
    step = sigma / 2
    centers = np.arange(-6 * sigma, TIME[-1] + 6 * sigma, step)
    omega, g = hhg.gabor_transform(TIME, signal, sigma, centers, max_omega=8 * W0)
    summed = g.sum(axis=0) * step / (np.sqrt(2 * np.pi) * sigma)
    direct = hhg.fourier_direct(TIME, signal, omega)
    np.testing.assert_allclose(summed, direct, atol=1e-6 * np.abs(direct).max())


def test_windows_beyond_the_edges_see_no_signal():
    centers = [-1000.0, 1500.0, TIME[-1] + 1000.0]
    omega, g = hhg.gabor_transform(TIME, tone(4), 50.0, centers)
    assert np.abs(g[[0, 2]]).max() < 1e-30 * np.abs(g[1]).max()


def test_half_window_at_the_edge_gives_half_the_weight():
    # A window centred on the first sample sees half of a Gaussian of a slow constant signal.
    sigma = 50.0
    omega, g = hhg.gabor_transform(TIME, np.ones_like(TIME), sigma, [TIME[0], 1500.0])
    assert abs(g[0, 0]) / abs(g[1, 0]) == pytest.approx(0.5, rel=1e-2)


def test_components_are_transformed_independently():
    both = np.stack([tone(3), tone(5)], axis=1)
    omega, g = hhg.gabor_transform(TIME, both, 80.0, [1000.0, 2000.0])
    _, g_x = hhg.gabor_transform(TIME, tone(3), 80.0, [1000.0, 2000.0])
    assert g.shape == (2, omega.size, 2)
    np.testing.assert_allclose(g[..., 0], g_x)


def test_spectrogram_sums_all_current_components():
    current = np.stack([tone(3), tone(5), 0.5 * tone(7)], axis=1)
    trace = hhg.CurrentTrace(TIME, current)
    centers, omega, s = hhg.spectrogram(trace, 80.0, max_omega=10 * W0)
    _, g = hhg.gabor_transform(TIME, current, 80.0, centers, max_omega=10 * W0)
    np.testing.assert_allclose(s, omega**2 * np.sum(np.abs(g) ** 2, axis=2))
    assert centers[0] == TIME[0]
    assert centers[-1] == pytest.approx(TIME[-1], abs=40.0)
    assert omega.max() <= 10 * W0


def test_sigma_must_be_positive():
    with pytest.raises(ValueError):
        hhg.gabor_transform(TIME, tone(3), 0.0, [1000.0])


# A linear chirp cos(w_a t + beta t^2 / 2) has instantaneous frequency w_a + beta t. For a
# Gaussian window the Gabor transform of it is known exactly:
#     |G(t_0, w)|^2  ~  exp(-sigma^2 (w - w_a - beta t_0)^2 / (1 + beta^2 sigma^4)),
# so the ridge follows the instantaneous frequency and is broadened by sqrt(1 + beta^2 sigma^4).
CHIRP_START, CHIRP_END = 3.0 * W0, 12.0 * W0
BETA = (CHIRP_END - CHIRP_START) / TIME[-1]


def chirp(start=CHIRP_START, beta=BETA, time=TIME):
    return np.cos(start * time + 0.5 * beta * time**2)


@pytest.mark.parametrize("direction", [+1, -1])
def test_ridge_follows_the_instantaneous_frequency(direction):
    # direction -1 sweeps down from 12 w_0 to 3 w_0, so the time axis is not mixed up.
    start, beta = (CHIRP_START, BETA) if direction > 0 else (CHIRP_END, -BETA)
    centers = np.linspace(400.0, TIME[-1] - 400.0, 25)
    omega, g = hhg.gabor_transform(
        TIME, chirp(start, beta), 80.0, centers, max_omega=14 * W0, pad_factor=8
    )
    ridge = omega[np.argmax(np.abs(g), axis=1)]
    np.testing.assert_allclose(ridge / W0, (start + beta * centers) / W0, atol=0.03)


@pytest.mark.parametrize("sigma", [40.0, 80.0, 160.0])
def test_chirp_ridge_is_broadened_as_predicted(sigma):
    omega, g = hhg.gabor_transform(TIME, chirp(), sigma, [1500.0], max_omega=14 * W0, pad_factor=16)
    power = np.abs(g[0]) ** 2
    above = omega[power >= power.max() / 2]
    expected = 2 * np.sqrt(np.log(2)) * np.sqrt(1 + BETA**2 * sigma**4) / sigma
    assert above[-1] - above[0] == pytest.approx(expected, rel=0.03)


def test_each_frequency_is_found_at_its_emission_time():
    # Read the other way round: at fixed w the spectrogram peaks at t = (w - w_a) / beta.
    # This is how the emission time of a harmonic is measured.
    trace = hhg.CurrentTrace(TIME, np.stack([chirp(), np.zeros_like(TIME), np.zeros_like(TIME)], 1))
    centers, omega, s = hhg.spectrogram(
        trace, 80.0, centers=np.arange(0.0, TIME[-1], 5.0), max_omega=14 * W0
    )
    for order in (5.0, 7.5, 10.0):
        k = np.argmin(np.abs(omega - order * W0))
        expected = (omega[k] - CHIRP_START) / BETA
        assert centers[np.argmax(s[:, k])] == pytest.approx(expected, abs=10.0)


CYCLE = 2 * np.pi / W0


def burst_train(bursts, phase, order=11, shift=0.0, n_cycles=8, width=CYCLE / 30):
    """A pump-like fundamental cos(w_0 t + 2 pi shift) and harmonic ``order`` emitted in short
    bursts, ``bursts`` times per cycle, at cycle phase ``phase`` of that fundamental."""
    time = np.arange(0.0, n_cycles * CYCLE, DT)
    harmonic = np.zeros_like(time)
    for k in range(n_cycles * bursts):
        t_burst = (k / bursts + phase - shift) * CYCLE
        harmonic += np.exp(-(((time - t_burst) / width) ** 2))
    current = np.zeros((time.size, 3))
    current[:, 0] = np.cos(W0 * time + 2 * np.pi * shift) + 0.01 * harmonic * np.cos(
        order * W0 * time
    )
    return hhg.CurrentTrace(time, current)


@pytest.mark.parametrize("bursts", [1, 2, 3])
@pytest.mark.parametrize("phase", [0.0, 0.1, 0.27])
def test_cycle_profile_finds_burst_count_and_time(bursts, phase):
    phase = phase % (1 / bursts)
    profile = hhg.cycle_profile(burst_train(bursts, phase), [11], 2 * CYCLE, 6 * CYCLE)
    assert profile.bursts()[0] == bursts
    found = profile.burst_phase()[0]
    # compare on the circle of length 1 / bursts
    difference = (found - phase + 0.5 / bursts) % (1 / bursts) - 0.5 / bursts
    assert abs(difference) < 0.01
    assert profile.modulation()[0] > 0.5


def test_cycle_phase_follows_the_fundamental():
    # Shifting the whole pattern in time must not change the phase found: it is measured
    # against the current's own fundamental, not against the time axis.
    a = hhg.cycle_profile(burst_train(2, 0.1), [11], 2 * CYCLE, 6 * CYCLE)
    b = hhg.cycle_profile(burst_train(2, 0.1, shift=0.37), [11], 2 * CYCLE, 6 * CYCLE)
    assert b.burst_phase()[0] == pytest.approx(a.burst_phase()[0], abs=0.01)


def test_steady_emission_has_no_modulation():
    time = np.arange(0.0, 8 * CYCLE, DT)
    current = np.zeros((time.size, 3))
    current[:, 0] = np.cos(W0 * time) + 0.01 * np.cos(11 * W0 * time)
    profile = hhg.cycle_profile(hhg.CurrentTrace(time, current), [11], 2 * CYCLE, 6 * CYCLE)
    assert profile.modulation()[0] < 0.02


def test_cycle_profile_shapes_and_normalisation():
    profile = hhg.cycle_profile(burst_train(3, 0.1), [9, 11, 13], 2 * CYCLE, 6 * CYCLE, n_bins=24)
    assert profile.intensity.shape == (3, 24)
    np.testing.assert_allclose(profile.intensity.max(axis=1), 1.0)
    assert profile.phase[0] == pytest.approx(0.5 / 24)


def test_cycle_profile_needs_a_full_cycle():
    with pytest.raises(ValueError):
        hhg.cycle_profile(burst_train(1, 0.0), [11], 2 * CYCLE, 2.5 * CYCLE)
