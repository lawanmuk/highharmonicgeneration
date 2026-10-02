"""Reading time-dependent total-current files.

The files follow the ``total_current`` layout written by real-time TDDFT codes such as
Octopus: comment lines starting with ``#``, then one row per time step with the columns

    Iter, t, I(1), I(2), I(3), ...

Time and current are in atomic units. Only the time and the three Cartesian current
components are used here.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COL_TIME = 1
COL_CURRENT = slice(2, 5)

_NAME = re.compile(
    r"^(?P<pulse>lp|cp|bcp)_"
    r"(?:I=(?P<intensity>[0-9.eE+-]+)(?:_(?P<run_tag>[a-z0-9]+))?|(?P<tag>[a-z]+)_?)"
    r"total_current"
)

PULSE_NAMES = {"lp": "linear", "cp": "collinear", "bcp": "bicircular"}


@dataclass(frozen=True)
class CurrentTrace:
    """Total current J(t) with shape (n_t, 3) on a uniform time grid (atomic units)."""

    time: np.ndarray
    current: np.ndarray
    name: str = ""

    def __post_init__(self):
        if self.time.ndim != 1 or self.time.size < 2:
            raise ValueError("time must be 1D with at least two points")
        if self.current.shape != (self.time.size, 3):
            raise ValueError(
                f"current must have shape ({self.time.size}, 3), got {self.current.shape}"
            )
        steps = np.diff(self.time)
        if not np.allclose(steps, steps[0], rtol=1e-6, atol=1e-12):
            raise ValueError("time steps are not uniform")

    @property
    def dt(self) -> float:
        return float(self.time[1] - self.time[0])


def load_current(path) -> CurrentTrace:
    """Load a total-current file into a :class:`CurrentTrace`."""
    path = Path(path)
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] < 5:
        raise ValueError(
            f"{path.name}: expected at least 5 columns (Iter, t, Jx, Jy, Jz), got {data.shape[1]}"
        )
    return CurrentTrace(data[:, COL_TIME], data[:, COL_CURRENT].copy(), name=path.name)


@dataclass(frozen=True)
class RunInfo:
    """What a dataset file name says about the run."""

    pulse: str
    intensity_w_cm2: float | None
    tag: str | None = None

    @property
    def label(self) -> str:
        text = PULSE_NAMES[self.pulse]
        if self.intensity_w_cm2 is not None:
            text += f", I = {self.intensity_w_cm2:.2g} W/cm$^2$"
        if self.tag:
            text += f" ({self.tag})"
        return text


def parse_run_name(name: str) -> RunInfo:
    """Parse a dataset file name into pulse type, intensity and an optional tag.

    Accepted forms: ``lp_I=1.5e12total_current.dat``, ``bcp_I=1.5e12_run2total_current.dat``
    (a further run at the same intensity) and ``bcp_pumpprobe_total_current.dat``.
    """
    match = _NAME.match(Path(name).name)
    if match is None:
        raise ValueError(f"cannot parse run name: {name}")
    intensity = match["intensity"]
    return RunInfo(
        pulse=match["pulse"],
        intensity_w_cm2=float(intensity) if intensity else None,
        tag=match["run_tag"] or match["tag"],
    )
