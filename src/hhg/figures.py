"""Regenerate every figure and the harmonic-yield table from the current datasets.

Usage:  hhg-figures --data HHG_datasets --out figures
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hhg.constants import OMEGA_PUMP  # noqa: E402
from hhg.harmonics import harmonic_yields  # noqa: E402
from hhg.io import PULSE_NAMES, load_current, parse_run_name  # noqa: E402
from hhg.spectrum import hhg_spectrum  # noqa: E402

# Validated reference palette: categorical for pulse types, one-hue blue ramp for intensity.
CATEGORICAL = {"lp": "#2a78d6", "cp": "#eb6834", "bcp": "#1baf7a"}
BLUE_RAMP = ["#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
MAX_ORDER = 25
YIELD_ORDERS = range(1, 15)

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "text.color": INK,
    "legend.frameon": False,
    "legend.labelcolor": INK,
    "lines.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _fmt(intensity: float) -> str:
    """1.5e12 -> '1.5e12' (for file names)."""
    return f"{intensity:.2g}".replace("e+", "e")


def _legend_below(ax, n_items, title=None):
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=n_items,
        title=title,
        title_fontsize=10,
    )


def _runs(data_dir: Path) -> dict:
    """Map run key (file stem prefix) to (RunInfo, path) for every raw current file."""
    runs = {}
    for path in sorted(data_dir.glob("*total_current.dat")):
        try:
            info = parse_run_name(path.name)
        except ValueError:
            continue
        runs[path.name.replace("total_current.dat", "").rstrip("_")] = (info, path)
    return runs


def _spectrum(path: Path):
    omega, spectrum = hhg_spectrum(load_current(path))
    keep = omega <= MAX_ORDER * OMEGA_PUMP
    return omega[keep], spectrum[keep]


def _spectrum_axes(title: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_yscale("log")
    ax.set_xlim(0, MAX_ORDER)
    ax.set_xticks(range(0, MAX_ORDER + 1, 5))
    ax.set_xticks(range(MAX_ORDER + 1), minor=True)
    ax.grid(True, which="minor", axis="x", color=GRID, linewidth=0.5)
    ax.grid(True, which="major", axis="both", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(r"Harmonic order $N = \omega / \omega_0$")
    ax.set_ylabel(r"$\omega^2\,|J(\omega)|^2$ (arb. units)")
    ax.set_title(title, loc="left")
    return fig, ax


def _intensity_scan(runs, pulse, out_dir):
    selected = sorted(
        (info.intensity_w_cm2, path)
        for info, path in runs.values()
        if info.pulse == pulse and info.intensity_w_cm2
    )
    if not selected:
        return None
    # spread the colors over the whole ramp so neighbouring intensities stay distinguishable
    picks = np.linspace(0, len(BLUE_RAMP) - 1, len(selected)).round().astype(int)
    ramp = [BLUE_RAMP[i] for i in picks]
    with plt.rc_context(STYLE):
        fig, ax = _spectrum_axes(f"{PULSE_NAMES[pulse].capitalize()} pump: intensity scan")
        for (intensity, path), color in zip(selected, ramp, strict=False):
            omega, spectrum = _spectrum(path)
            ax.plot(omega / OMEGA_PUMP, spectrum, color=color, label=f"{intensity:.2g} W/cm$^2$")
        _legend_below(ax, len(selected), title="Peak intensity")
        path = out_dir / f"{PULSE_NAMES[pulse]}_intensity_scan.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path


def _pump_comparison(runs, intensity, out_dir):
    selected = [
        (info, path)
        for info, path in runs.values()
        if info.intensity_w_cm2 == intensity and info.tag is None
    ]
    order = {"lp": 0, "cp": 1, "bcp": 2}
    selected.sort(key=lambda item: order[item[0].pulse])
    with plt.rc_context(STYLE):
        fig, ax = _spectrum_axes(f"Pump configurations at {intensity:.2g} W/cm$^2$")
        for info, path in selected:
            omega, spectrum = _spectrum(path)
            orders = omega / OMEGA_PUMP
            ax.plot(orders, spectrum, color=CATEGORICAL[info.pulse], label=PULSE_NAMES[info.pulse])
        _legend_below(ax, len(selected))
        path = out_dir / f"pump_comparison_I={_fmt(intensity)}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path


def _selection_rule(runs, key, out_dir):
    if key not in runs:
        return None
    info, source = runs[key]
    omega, spectrum = _spectrum(source)
    orders = np.array(list(YIELD_ORDERS))
    yields = harmonic_yields(omega, spectrum, OMEGA_PUMP, orders)
    yields = yields / yields.max()
    forbidden = orders % 3 == 0
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(
            orders[~forbidden],
            yields[~forbidden],
            width=0.7,
            color=CATEGORICAL["bcp"],
            edgecolor=SURFACE,
            linewidth=2,
            label=r"$N = 3n \pm 1$ (allowed)",
        )
        ax.bar(
            orders[forbidden],
            yields[forbidden],
            width=0.7,
            color=AXIS,
            hatch="///",
            edgecolor=MUTED,
            linewidth=0.8,
            label=r"$N = 3n$ (forbidden)",
        )
        ax.set_yscale("log")
        ax.set_xticks(orders)
        ax.grid(True, axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("Harmonic order N")
        ax.set_ylabel("Integrated yield (normalised)")
        ax.set_title(
            f"Bicircular selection rule, I = {info.intensity_w_cm2:.2g} W/cm$^2$", loc="left"
        )
        ax.annotate(
            "band-gap\nemission",
            (6, yields[orders == 6][0]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK_2,
        )
        _legend_below(ax, 2)
        path = out_dir / f"bicircular_selection_rule_I={_fmt(info.intensity_w_cm2)}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path


def write_yield_table(runs, out_dir: Path) -> Path:
    """CSV of integrated harmonic yields, one row per run: the table view of the figures."""
    path = out_dir / "harmonic_yields.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "pulse", "intensity_w_cm2", *[f"H{n}" for n in YIELD_ORDERS]])
        for key, (info, source) in runs.items():
            omega, spectrum = _spectrum(source)
            yields = harmonic_yields(omega, spectrum, OMEGA_PUMP, YIELD_ORDERS)
            writer.writerow(
                [
                    key,
                    PULSE_NAMES[info.pulse],
                    info.intensity_w_cm2 or "",
                    *[f"{y:.4e}" for y in yields],
                ]
            )
    return path


def make_figures(data_dir, out_dir) -> list[Path]:
    """Create all figures and the yield table; return the paths written.

    The bicircular runs are not plotted as an intensity scan: the 1.5e12 and 5e12 W/cm^2
    files give spectra that agree to 0.2 %, which needs checking against the simulation input
    before the two can be presented as different intensities.
    """
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    runs = _runs(data_dir)
    if not runs:
        raise FileNotFoundError(f"no *total_current.dat files found in {data_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [
        _intensity_scan(runs, "lp", out_dir),
        _intensity_scan(runs, "cp", out_dir),
        _pump_comparison(runs, 1.5e12, out_dir),
        _selection_rule(runs, "bcp_I=1.5e12", out_dir),
        write_yield_table(runs, out_dir),
    ]
    return [p for p in written if p is not None]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Regenerate the HHG figures and yield table.")
    parser.add_argument("--data", default="HHG_datasets", help="folder with *total_current.dat")
    parser.add_argument("--out", default="figures", help="output folder (default: figures)")
    args = parser.parse_args(argv)
    for path in make_figures(args.data, args.out):
        print(path)


if __name__ == "__main__":
    main()
