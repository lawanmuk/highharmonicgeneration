"""Harmonic-resolved analysis of HHG spectra."""

import numpy as np


def harmonic_yields(
    omega: np.ndarray,
    spectrum: np.ndarray,
    omega_0: float,
    orders,
    half_width: float = 0.25,
) -> np.ndarray:
    """Integrated intensity in a window of +-``half_width`` (in harmonic orders) around each order.

    ``omega`` must be sorted and uniformly spaced.
    """
    omega = np.asarray(omega, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    if not 0 < half_width <= 0.5:
        raise ValueError("half_width must be in (0, 0.5]")
    order_axis = omega / omega_0
    d_order = order_axis[1] - order_axis[0]
    yields = []
    for n in np.atleast_1d(orders):
        inside = np.abs(order_axis - n) <= half_width
        if not inside.any():
            raise ValueError(f"order {n} is outside the spectrum")
        yields.append(np.sum(spectrum[inside]) * d_order)
    return np.array(yields)


def selection_rule_contrast(yields_allowed: np.ndarray, yields_forbidden: np.ndarray) -> float:
    """Ratio of the mean forbidden to the mean allowed harmonic yield (small = rule obeyed)."""
    return float(np.mean(yields_forbidden) / np.mean(yields_allowed))
