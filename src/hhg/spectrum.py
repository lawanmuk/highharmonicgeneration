"""Windowed Fourier transforms and HHG spectra.

The emitted HHG intensity is proportional to the squared dipole acceleration,

    S(w) = w^2 * sum_i |J_i(w)|^2,

where J_i(w) is the Fourier transform of the i-th current component. Before transforming,
the current is multiplied by the smooth-step window 1 - 3x^2 + 2x^3 (x = t / t_max), which
goes from 1 to 0 with zero slope at both ends and suppresses truncation artefacts.
"""

import numpy as np

from hhg.io import CurrentTrace


def smooth_step_window(time: np.ndarray) -> np.ndarray:
    """1 - 3x^2 + 2x^3 with x = (t - t_0) / (t_max - t_0): 1 at the start, 0 at the end."""
    time = np.asarray(time, dtype=float)
    x = (time - time[0]) / (time[-1] - time[0])
    return 1.0 - 3.0 * x**2 + 2.0 * x**3


def fourier_direct(time: np.ndarray, signal: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """F(w) = sum_t signal(t) exp(i w t) dt on an arbitrary frequency grid.

    ``signal`` may have shape (n_t,) or (n_t, n_components). Exact but O(n_t * n_omega);
    works in blocks of frequencies to keep memory small.
    """
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    omega = np.asarray(omega, dtype=float)
    dt = time[1] - time[0]
    squeeze = signal.ndim == 1
    signal = signal.reshape(time.size, -1)
    out = np.empty((omega.size, signal.shape[1]), dtype=complex)
    for start in range(0, omega.size, 64):
        block = omega[start : start + 64]
        out[start : start + 64] = np.exp(1j * np.outer(block, time)) @ signal
    out *= dt
    return out[:, 0] if squeeze else out


def fourier_fft(time: np.ndarray, signal: np.ndarray, pad_factor: int = 4):
    """Same transform as :func:`fourier_direct`, evaluated with a zero-padded FFT.

    Returns ``(omega, F)`` for omega = 0 ... pi/dt. ``pad_factor`` sets how finely the
    spectrum is sampled (a factor 4 gives a grid four times finer than 2 pi / t_max).
    """
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if pad_factor < 1:
        raise ValueError("pad_factor must be at least 1")
    dt = time[1] - time[0]
    n = int(2 ** np.ceil(np.log2(time.size * pad_factor)))
    # numpy's FFT uses exp(-i w t); conj of the transform of a real signal gives exp(+i w t)
    transform = np.conj(np.fft.rfft(signal, n=n, axis=0)) * dt
    omega = 2.0 * np.pi * np.fft.rfftfreq(n, d=dt)
    phase = np.exp(1j * omega * time[0])
    if transform.ndim == 2:
        phase = phase[:, None]
    return omega, transform * phase


def current_transform(trace: CurrentTrace, omega: np.ndarray | None = None, window: bool = True):
    """Fourier transform J_i(w) of each current component, optionally windowed.

    With ``omega=None`` the transform is computed by FFT on its natural grid; otherwise it is
    evaluated exactly at the given frequencies (Hartree). Returns ``(omega, J)`` with ``J`` of
    shape ``(n_omega, 3)``.
    """
    current = trace.current * smooth_step_window(trace.time)[:, None] if window else trace.current
    if omega is None:
        return fourier_fft(trace.time, current)
    omega = np.asarray(omega, dtype=float)
    return omega, fourier_direct(trace.time, current, omega)


def hhg_spectrum(trace: CurrentTrace, omega: np.ndarray | None = None, window: bool = True):
    """HHG spectrum S(w) = w^2 * sum_i |J_i(w)|^2 of a current trace.

    With ``omega=None`` the spectrum is computed by FFT on its natural grid; otherwise the
    exact transform is evaluated at the given frequencies (Hartree). Returns ``(omega, S)``.
    """
    omega, transform = current_transform(trace, omega, window)
    return omega, omega**2 * np.sum(np.abs(transform) ** 2, axis=1)
