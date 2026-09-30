# Contributing

## Setup

```bash
git clone https://github.com/lawanmuk/highharmonicgeneration.git
cd highharmonicgeneration
python -m venv .venv
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
ruff format src tests
ruff check src tests
pytest
```

All three must pass; CI runs the same checks.

## Guidelines

- Put reusable logic in `src/hhg/`, not in scripts, and add a test for it in `tests/`.
- Use atomic units inside the package; convert to eV, fs or harmonic order only for plotting.
- New datasets go in `HHG_datasets/` with the naming pattern `<lp|cp|bcp>_I=<intensity>total_current.dat`; processed output does not belong there.
- After changing the analysis, regenerate the figures with `hhg-figures` and commit them.
- For a bug fix, add a test that fails before the fix and passes after it.
- Use short commit messages with a type prefix: `feat:`, `fix:`, `test:`, `docs:`, `ci:`, `chore:`.
- Add a line to `CHANGELOG.md` under "Unreleased".
