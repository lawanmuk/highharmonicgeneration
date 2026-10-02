# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `hhg.early_response_ratio`: checks intensity labels from the linear early-time response, without the simulation input.
- Dataset tests confirming the linear and collinear intensity labels and the corrected bicircular label.
- Run-name parser accepts a tag after the intensity, for example `bcp_I=1.5e12_run2`.
- Pinned `requirements.txt` (generated with `uv pip compile --universal`) for reproducible installs on Python 3.10 and newer.

### Fixed
- `bcp_I=5e12total_current.dat` was mislabelled: it is a second run at 1.5e12 W/cm² and is now `bcp_I=1.5e12_run2total_current.dat`. Found from the linear early-time response and confirmed by the author.
- README grammar in the Author section.

### Changed
- README data table and note describe the renamed bicircular run.

## [0.1.0] - 2026-09-30

### Added
- `hhg` Python package with a `pyproject.toml` build definition: current-file reader, run-name parser, smooth-step window, direct and FFT transforms, HHG spectrum over all current components, harmonic yields and selection-rule contrast.
- `hhg-figures` command that regenerates every figure and `figures/harmonic_yields.csv`.
- pytest suite, including physics regression tests on the datasets (bicircular 3n +- 1 selection rule).
- GitHub Actions CI (lint, tests on Python 3.10 to 3.13, figure regeneration, pip-audit) and Dependabot.

### Fixed
- Window function rose to 2 at the end of the trace instead of falling to 0 (`1 - x^2 + 2x^3` instead of `1 - 3x^2 + 2x^3`), which added a noise floor and distorted harmonic 7 by a factor of about 75.
- Only the x component of the current was used; the spectrum now sums x, y and z.
- The Fourier sum was multiplied by the energy step instead of the time step.
- The processing script looked for data files without their `.dat` extension in the wrong folder.
- `current2HHG.m` overwrote its `filename` and `V` arguments, and two offsets wrote to the same output file.

### Changed
- Figures in `HHG_plots/` (made with the faulty window) replaced by regenerated ones in `figures/`.
- Processed MATLAB spectra moved from the data folders to `results/matlab/`; original scripts moved to `legacy/`.

### Removed
- Duplicate copy of `lp_I=1.5e12total_current` in the script folder.
