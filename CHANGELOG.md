# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Test that the time and frequency spreads of the Gabor transform satisfy the Gaussian limit (product 1/2) at four window widths, and a README paragraph on the resolution trade-off.
- `hhg.gabor_transform` and `hhg.spectrogram`: time-frequency analysis with a sliding Gaussian window, with tests for the sign convention, time and frequency resolution, edge handling and window summation.
- Tests that the Gabor transform recovers a linear chirp: the ridge follows the instantaneous frequency in both sweep directions, its width matches the exact result, and each frequency is found at its emission time.
- `hhg-figures` now also writes `spectrogram_I=1.5e12.png`, the time-resolved emission for the linear, collinear and bicircular pumps; README section on time-resolved emission.
- `hhg.cycle_profile` and `CycleProfile`: folds the emission of each harmonic onto one pump cycle and gives the number of bursts per cycle, the emission time (cycle phase) and the strength of the burst pattern, with tests on synthetic burst trains. Dataset tests: bicircular harmonics are emitted three times per cycle, at the same cycle phase in both runs, and collinear high harmonics once per cycle.

## [0.2.0] - 2026-10-06

### Added
- `hhg.harmonic_polarization` and `HarmonicPolarization`: circular yields, helicity and signed ellipticity for each harmonic order, with tests on circular, linear and elliptical fields.
- Dataset tests for the bicircular helicity rule: 3n+1 and 3n-1 harmonics rotate in opposite directions in all three bicircular runs, and linear and collinear runs show no helicity.
- `hhg-figures` now also writes `bicircular_polarization_I=1.5e12.png` (circular spectra and helicity per harmonic) and `harmonic_polarization.csv`; README section on harmonic polarization.
- `hhg.circular_components`: splits the in-plane HHG spectrum into counter-clockwise (S+) and clockwise (S-) circular parts, with tests for the sign convention, linear and elliptical fields, rotation invariance and S+ + S- = in-plane spectrum.
- `hhg.current_transform`: the windowed Fourier transform of each current component, shared by `hhg_spectrum` and the polarization analysis.
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

[Unreleased]: https://github.com/lawanmuk/highharmonicgeneration/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/lawanmuk/highharmonicgeneration/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/lawanmuk/highharmonicgeneration/releases/tag/v0.1.0
