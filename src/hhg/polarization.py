"""Circular-polarization analysis of HHG spectra.

The in-plane current transform is split into two counter-rotating circular parts,

    J_+(w) = (J_x(w) - i J_y(w)) / sqrt(2),    J_-(w) = (J_x(w) + i J_y(w)) / sqrt(2),

using the same transform convention as :mod:`hhg.spectrum`, F(w) = integral f(t) exp(+i w t) dt.
With this convention J_+ holds emission whose polarization rotates counter-clockwise, from
+x towards +y, and J_- holds emission rotating clockwise. The two parts are orthogonal, so

    S_+(w) + S_-(w) = w^2 (|J_x(w)|^2 + |J_y(w)|^2),

the in-plane part of :func:`hhg.hhg_spectrum`. A linearly polarized harmonic splits equally
between the two.
"""

import numpy as np

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
