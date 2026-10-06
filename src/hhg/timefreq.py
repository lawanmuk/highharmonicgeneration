"""Time-frequency analysis of current traces with a Gabor transform.

The Gabor transform slides a Gaussian window of width ``sigma`` along the signal and
Fourier-transforms each windowed piece,

    G(t_0, w) = integral f(t) exp(-(t - t_0)^2 / (2 sigma^2)) exp(+i w t) dt,

with the same sign convention as :mod:`hhg.spectrum`. The window width sets the trade-off
between time and frequency resolution: a peak in |G|^2 has a full width at half maximum of
2 sqrt(ln 2) / sigma in frequency, so separating neighbouring harmonics (spaced by the pump
frequency w_0) needs sigma well above 1 / w_0, while timing emission within one optical cycle
(2 pi / w_0, about 207 atomic units) needs sigma well below that cycle.
"""

import numpy as np

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
