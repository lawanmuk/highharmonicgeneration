"""Consistency checks between runs."""

import numpy as np

from hhg.io import CurrentTrace


def early_response_ratio(reference: CurrentTrace, other: CurrentTrace, t_max: float) -> float:
    """RMS ratio of the in-plane current of ``other`` to ``reference`` for times up to ``t_max``.

    While the pulse is still weak the response is linear in the field, so two runs that
    differ only in peak intensity give a ratio equal to the field ratio sqrt(I_other / I_ref).
    This checks intensity labels without access to the simulation input. Both runs must use
    the same time step.
    """
    if not np.isclose(reference.dt, other.dt):
        raise ValueError("runs use different time steps")
    if t_max <= reference.time[0]:
        raise ValueError("t_max must be later than the first time step")
    ref = reference.current[reference.time <= t_max, :2]
    oth = other.current[other.time <= t_max, :2]
    n = min(len(ref), len(oth))
    norm = np.sum(ref[:n] ** 2)
    if norm == 0:
        raise ValueError("reference current is zero before t_max")
    return float(np.sqrt(np.sum(oth[:n] ** 2) / norm))
