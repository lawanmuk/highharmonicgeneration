"""High-harmonic generation spectra from time-dependent current traces.

All quantities are in atomic units unless a name says otherwise.
"""

from hhg.constants import AU_TO_FS, HARTREE_TO_EV, OMEGA_PUMP
from hhg.diagnostics import early_response_ratio
from hhg.figures import make_figures
from hhg.harmonics import harmonic_yields, selection_rule_contrast
from hhg.io import CurrentTrace, RunInfo, load_current, parse_run_name
from hhg.polarization import circular_components
from hhg.spectrum import (
    current_transform,
    fourier_direct,
    fourier_fft,
    hhg_spectrum,
    smooth_step_window,
)

__all__ = [
    "AU_TO_FS",
    "HARTREE_TO_EV",
    "OMEGA_PUMP",
    "CurrentTrace",
    "RunInfo",
    "circular_components",
    "current_transform",
    "early_response_ratio",
    "fourier_direct",
    "fourier_fft",
    "harmonic_yields",
    "hhg_spectrum",
    "load_current",
    "make_figures",
    "parse_run_name",
    "selection_rule_contrast",
    "smooth_step_window",
]

__version__ = "0.1.0"
