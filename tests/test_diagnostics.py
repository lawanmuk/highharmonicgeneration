import numpy as np
import pytest

import hhg


def make_trace(scale=1.0, dt=0.08, n=2000):
    time = np.arange(n) * dt
    current = np.zeros((n, 3))
    current[:, 0] = scale * np.sin(0.03 * time) * np.minimum(time / 100.0, 1.0)
    current[:, 1] = scale * np.cos(0.05 * time) * np.minimum(time / 100.0, 1.0)
    return hhg.CurrentTrace(time, current)


def test_scaled_run_gives_scale_factor():
    ratio = hhg.early_response_ratio(make_trace(), make_trace(scale=1.826), t_max=50.0)
    assert ratio == pytest.approx(1.826)


def test_runs_of_different_length_are_compared_on_common_times():
    ratio = hhg.early_response_ratio(make_trace(n=2000), make_trace(n=3000), t_max=50.0)
    assert ratio == pytest.approx(1.0)


def test_different_time_steps_are_rejected():
    with pytest.raises(ValueError, match="time steps"):
        hhg.early_response_ratio(make_trace(), make_trace(dt=0.05), t_max=50.0)


def test_t_max_before_start_is_rejected():
    with pytest.raises(ValueError, match="t_max"):
        hhg.early_response_ratio(make_trace(), make_trace(), t_max=-1.0)
