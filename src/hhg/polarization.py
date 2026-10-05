"""Circular-polarization analysis of HHG spectra.

The in-plane current transform is split into two counter-rotating circular parts,

    J_+(w) = (J_x(w) - i J_y(w)) / sqrt(2),    J_-(w) = (J_x(w) + i J_y(w)) / sqrt(2),

using the same transform convention as :mod:`hhg.spectrum`, F(w) = integral f(t) exp(+i w t) dt.
With this convention J_+ holds emission whose polarization rotates counter-clockwise, from
+x towards +y, and J_- holds emission rotating clockwise. The two parts are orthogonal, so

    S_+(w) + S_-(w) = w^2 (|J_x(w)|^2 + |J_y(w)|^2),

the in-plane part of :func:`hhg.hhg_spectrum`. A linearly polarized harmonic splits equally
between the two.

Per harmonic order, :func:`harmonic_polarization` turns the integrated yields Y_+ and Y_-
into a helicity, (Y_+ - Y_-) / (Y_+ + Y_-), and a signed ellipticity (minor/major axis ratio).
"""

from dataclasses import dataclass

import numpy as np

from hhg.constants import OMEGA_PUMP
from hhg.harmonics import harmonic_yields
from hhg.io import CurrentTrace
from hhg.spectrum import current_transform


def circular_components(trace: CurrentTrace, omega: np.ndarray | None = None, window: bool = True):
    """Counter-clockwise and clockwise HHG spectra ``S_+`` and ``S_-``.

    Returns ``(omega, s_plus, s_minus)`` with S_+- = w^2 |J_+-(w)|^2. ``omega`` and
    ``window`` work as in :func:`hhg.hhg_spectrum`. The z component of the current, normal
    to the polarization plane, is not included.
    """
    omega, transform = current_transform(trace, omega, window)
    j_x, j_y = transform[:, 0], transform[:, 1]
    j_plus = (j_x - 1j * j_y) / np.sqrt(2.0)
    j_minus = (j_x + 1j * j_y) / np.sqrt(2.0)
    return omega, omega**2 * np.abs(j_plus) ** 2, omega**2 * np.abs(j_minus) ** 2


@dataclass(frozen=True)
class HarmonicPolarization:
    """Circular yields per harmonic order and the polarization numbers derived from them.

    ``yield_plus`` and ``yield_minus`` are the counter-clockwise and clockwise yields
    integrated around each order, as in :func:`hhg.harmonic_yields`.
    """

    orders: np.ndarray
    yield_plus: np.ndarray
    yield_minus: np.ndarray

    @property
    def total(self) -> np.ndarray:
        """In-plane yield, Y_+ + Y_-."""
        return self.yield_plus + self.yield_minus

    @property
    def helicity(self) -> np.ndarray:
        """(Y_+ - Y_-) / (Y_+ + Y_-): +1 counter-clockwise, -1 clockwise, 0 linear."""
        return (self.yield_plus - self.yield_minus) / self.total

    @property
    def ellipticity(self) -> np.ndarray:
        """Signed minor/major axis ratio, (sqrt Y_+ - sqrt Y_-) / (sqrt Y_+ + sqrt Y_-).

        The sign follows the helicity. The two are related by h = 2 e / (1 + e^2).
        """
        root_plus, root_minus = np.sqrt(self.yield_plus), np.sqrt(self.yield_minus)
        return (root_plus - root_minus) / (root_plus + root_minus)


def harmonic_polarization(
    trace: CurrentTrace,
    orders,
    omega_0: float = OMEGA_PUMP,
    half_width: float = 0.25,
    window: bool = True,
) -> HarmonicPolarization:
    """Helicity and ellipticity of each harmonic ``order`` of ``trace``.

    The circular spectra from :func:`circular_components` are integrated over
    +-``half_width`` harmonic orders around each order of ``omega_0``.
    """
    omega, s_plus, s_minus = circular_components(trace, window=window)
    orders = np.atleast_1d(np.asarray(orders))
    return HarmonicPolarization(
        orders=orders,
        yield_plus=harmonic_yields(omega, s_plus, omega_0, orders, half_width),
        yield_minus=harmonic_yields(omega, s_minus, omega_0, orders, half_width),
    )
