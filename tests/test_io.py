import numpy as np
import pytest

import hhg

HEADER = """################################################################################
# HEADER
# Iter             t                  I(1)                I(2)                I(3)
################################################################################
"""


def write_current(path, n=50, dt=0.08, columns=5):
    rows = np.zeros((n, columns))
    rows[:, 0] = np.arange(n)
    rows[:, 1] = np.arange(n) * dt
    rows[:, 2] = np.sin(rows[:, 1])
    rows[:, 3] = np.cos(rows[:, 1])
    path.write_text(HEADER + "\n".join(" ".join(f"{v:.12e}" for v in r) for r in rows) + "\n")
    return rows


def test_load_current_reads_header_file(tmp_path):
    path = tmp_path / "lp_I=1e12total_current.dat"
    rows = write_current(path)
    trace = hhg.load_current(path)
    assert trace.current.shape == (50, 3)
    assert trace.dt == pytest.approx(0.08)
    np.testing.assert_allclose(trace.current[:, 0], rows[:, 2])
    assert trace.name == path.name


def test_load_current_rejects_too_few_columns(tmp_path):
    path = tmp_path / "bad.dat"
    write_current(path, columns=4)
    with pytest.raises(ValueError, match="at least 5 columns"):
        hhg.load_current(path)


def test_non_uniform_time_is_rejected():
    time = np.array([0.0, 0.1, 0.3])
    with pytest.raises(ValueError, match="uniform"):
        hhg.CurrentTrace(time, np.zeros((3, 3)))


def test_wrong_current_shape_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        hhg.CurrentTrace(np.arange(4.0), np.zeros((4, 2)))


@pytest.mark.parametrize(
    ("name", "pulse", "intensity", "tag"),
    [
        ("lp_I=1e11total_current.dat", "lp", 1e11, None),
        ("cp_I=1.5e12total_current.dat", "cp", 1.5e12, None),
        ("bcp_I=5e12total_current.dat", "bcp", 5e12, None),
        ("bcp_I=1.5e12_run2total_current.dat", "bcp", 1.5e12, "run2"),
        ("bcp_pumpprobe_total_current.dat", "bcp", None, "pumpprobe"),
    ],
)
def test_parse_run_name(name, pulse, intensity, tag):
    info = hhg.parse_run_name(name)
    assert (info.pulse, info.intensity_w_cm2, info.tag) == (pulse, intensity, tag)
    assert info.label


def test_parse_run_name_rejects_unknown():
    with pytest.raises(ValueError):
        hhg.parse_run_name("something_else.dat")
