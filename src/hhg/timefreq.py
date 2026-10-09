"""Time-frequency analysis of current traces with a Gabor transform.

The Gabor transform slides a Gaussian window of width ``sigma`` along the signal and
Fourier-transforms each windowed piece,

    G(t_0, w) = integral f(t) exp(-(t - t_0)^2 / (2 sigma^2)) exp(+i w t) dt,

with the same sign convention as :mod:`hhg.spectrum`. The window width sets the trade-off
between time and frequency resolution: a peak in |G|^2 has a full width at half maximum of
2 sqrt(ln 2) / sigma in frequency, so separating neighbouring harmonics (spaced by the pump
frequency w_0) needs sigma well above 1 / w_0, while timing emission within one optical cycle
(2 pi / w_0, about 207 atomic units) needs sigma well below that cycle.

:func:`cycle_profile` uses a short window to follow harmonic emission within the optical
cycle: it folds the time-resolved intensity of each harmonic onto one cycle of the pump.
"""

from dataclasses import dataclass

import numpy as np

from hhg.constants import OMEGA_PUMP
from hhg.io import CurrentTrace
from hhg.spectrum import fourier_fft

CUTOFF = 5.0  # the window is cut at +-CUTOFF sigma, where it has fallen to exp(-12.5)


def gabor_transform(
    time: np.ndarray,
    signal: np.ndarray,
    sigma: float,
    centers: np.ndarray,
    max_omega: float | None = None,
    pad_factor: int = 2,
):
    """Gabor transform of ``signal`` at window centers ``centers`` (atomic units).

    ``signal`` may have shape (n_t,) or (n_t, n_components); the signal is taken as zero
    outside the sampled times. Returns ``(omega, G)`` with ``G`` of shape
    (n_centers, n_omega) or (n_centers, n_omega, n_components). ``max_omega`` drops higher
    frequencies to save memory; ``pad_factor`` works as in :func:`hhg.fourier_fft`.
    """
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    centers = np.atleast_1d(np.asarray(centers, dtype=float))
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    dt = time[1] - time[0]
    squeeze = signal.ndim == 1
    signal = signal.reshape(time.size, -1)

    # zero-pad by half a segment on both sides so every segment fits inside the array
    half = int(np.ceil(CUTOFF * sigma / dt))
    padded = np.zeros((time.size + 2 * half, signal.shape[1]))
    padded[half : half + time.size] = signal
    offsets = np.arange(-half, half + 1)

    omega = None
    out = None
    for k, center in enumerate(centers):
        nearest = int(round((center - time[0]) / dt))
        # centers beyond the trace are clamped to its end; the window is still evaluated at
        # the true center, so the result stays exact (the signal is zero out there)
        nearest = min(max(nearest, 0), time.size - 1)
        index = nearest + offsets  # index into the original grid, may lie outside it
        segment_time = time[0] + index * dt
        window = np.exp(-0.5 * ((segment_time - center) / sigma) ** 2)
        segment = padded[index + half] * window[:, None]
        omega_k, transform = fourier_fft(segment_time, segment, pad_factor)
        if out is None:
            keep = slice(None) if max_omega is None else omega_k <= max_omega
            omega = omega_k[keep]
            out = np.empty((centers.size, omega.size, signal.shape[1]), dtype=complex)
        out[k] = transform[keep]
    return omega, (out[..., 0] if squeeze else out)


def spectrogram(
    trace: CurrentTrace,
    sigma: float,
    centers: np.ndarray | None = None,
    max_omega: float | None = None,
):
    """Time-resolved HHG spectrum S(t_0, w) = w^2 * sum_i |G_i(t_0, w)|^2.

    ``centers`` defaults to steps of sigma / 2 over the whole trace. Returns
    ``(centers, omega, S)`` with ``S`` of shape (n_centers, n_omega). Integrated over the
    centers this gives the HHG spectrum smoothed over a frequency width of about 1 / sigma.
    """
    if centers is None:
        centers = np.arange(trace.time[0], trace.time[-1] + sigma / 4, sigma / 2)
    centers = np.atleast_1d(np.asarray(centers, dtype=float))
    omega, transform = gabor_transform(trace.time, trace.current, sigma, centers, max_omega)
    return centers, omega, omega**2 * np.sum(np.abs(transform) ** 2, axis=2)


def _gabor_at(time, signal, sigma, centers, omega):
    """Gabor transform at one exact frequency (an FFT grid point would be slightly off, and
    that small offset turns into a phase error that grows with time)."""
    dt = time[1] - time[0]
    half = int(np.ceil(CUTOFF * sigma / dt))
    rotated = signal * np.exp(1j * omega * time)
    out = np.empty(centers.size, dtype=complex)
    for k, center in enumerate(centers):
        nearest = int(round((center - time[0]) / dt))
        lo, hi = max(nearest - half, 0), min(nearest + half + 1, time.size)
        window = np.exp(-0.5 * ((time[lo:hi] - center) / sigma) ** 2)
        out[k] = np.sum(rotated[lo:hi] * window) * dt
    return out


@dataclass(frozen=True)
class CycleProfile:
    """Emission of each harmonic order folded onto one optical cycle.

    ``phase`` holds the bin centres in cycles (0 to 1); ``intensity`` has one row per order,
    each scaled to a maximum of 1.
    """

    orders: np.ndarray
    phase: np.ndarray
    intensity: np.ndarray

    def _component(self, max_bursts: int):
        coefficients = np.fft.rfft(self.intensity, axis=1)[:, 1 : max_bursts + 1]
        bursts = 1 + np.argmax(np.abs(coefficients), axis=1)
        picked = coefficients[np.arange(bursts.size), bursts - 1]
        return bursts, picked, np.mean(self.intensity, axis=1)

    def bursts(self, max_bursts: int = 4) -> np.ndarray:
        """Number of emission bursts per cycle: the strongest Fourier component, 1 to max."""
        return self._component(max_bursts)[0]

    def burst_phase(self, max_bursts: int = 4) -> np.ndarray:
        """Cycle phase of the first burst, in [0, 1 / bursts): the emission time."""
        bursts, picked, _ = self._component(max_bursts)
        # the profile's bins sit at phase (i + 1/2) / n_bins, the FFT assumes i / n_bins
        offset = 0.5 / self.phase.size
        return (-np.angle(picked) / (2 * np.pi * bursts) + offset) % (1.0 / bursts)

    def modulation(self, max_bursts: int = 4) -> np.ndarray:
        """Amplitude of the burst pattern relative to the mean: 0 for steady emission."""
        _, picked, mean = self._component(max_bursts)
        return 2 * np.abs(picked) / self.phase.size / mean


def cycle_profile(
    trace: CurrentTrace,
    orders,
    t_start: float,
    t_stop: float,
    sigma: float | None = None,
    n_bins: int = 40,
    omega_0: float = OMEGA_PUMP,
    reference: int = 0,
) -> CycleProfile:
    """Fold the emission of each harmonic ``order`` onto one cycle of the pump.

    The time-resolved intensity at each order (from :func:`spectrogram` with a short window,
    by default a tenth of a cycle) is averaged over ``t_start`` to ``t_stop`` as a function
    of the cycle phase. With so short a window each order is a band about 2.6 orders wide.

    The current itself fixes the phase: phase 0 is where the component of current
    ``reference`` (0 = x) at the pump frequency peaks. That component is extracted with a
    window of half a cycle, so it follows slow changes of the pump phase.
    """
    cycle = 2 * np.pi / omega_0
    if sigma is None:
        sigma = cycle / 10
    if t_stop - t_start < cycle:
        raise ValueError("t_start to t_stop must span at least one cycle")
    orders = np.atleast_1d(np.asarray(orders))
    centers = np.arange(t_start, t_stop, cycle / (4 * n_bins))

    fundamental = _gabor_at(trace.time, trace.current[:, reference], cycle / 2, centers, omega_0)
    # cos(w_0 t + phi) gives G(t, w_0) ~ exp(-i phi), so the cycle phase is w_0 t - arg G
    phase = ((omega_0 * centers - np.angle(fundamental)) / (2 * np.pi)) % 1.0

    _, omega, power = spectrogram(trace, sigma, centers, max_omega=(orders.max() + 3) * omega_0)
    bins = np.minimum((phase * n_bins).astype(int), n_bins - 1)
    counts = np.maximum(np.bincount(bins, minlength=n_bins), 1)
    rows = []
    for order in orders:
        profile = np.bincount(bins, power[:, np.argmin(np.abs(omega - order * omega_0))], n_bins)
        profile = profile / counts
        rows.append(profile / profile.max())
    return CycleProfile(orders, (np.arange(n_bins) + 0.5) / n_bins, np.array(rows))
