"""Manual extrinsic correction: identity, composition and the module's own consistency check."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.extrinsic_correction import _self_check, compose_delta, is_zero_delta  # noqa: E402


def test_module_self_check():
    _self_check()


def test_zero_delta_is_identity():
    rvec, tvec = np.array([[0.1], [-2.2], [2.1]]), np.array([[0.2], [1.2], [3.1]])
    assert is_zero_delta(None) and is_zero_delta({})
    r, t = compose_delta(rvec, tvec, {})
    assert np.allclose(r, rvec) and np.allclose(t, tvec)


def test_dz_moves_the_camera_along_its_view_axis():
    rvec, tvec = np.array([[0.1], [-2.2], [2.1]]), np.array([[0.2], [1.2], [3.1]])
    _, t = compose_delta(rvec, tvec, {"dz": 0.5})
    assert abs(np.linalg.norm(t - tvec) - 0.5) < 1e-9
