"""Self-contained regression checks for dataset_export.py's confidence/
visibility/interpolation-mask threading and the sync export gate.

Run: python -m core.test_dataset_export
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from core.dataset_export import (
    JOINT_NAMES, N_JOINTS,
    _interpolate_window_gaps, load_trial_confidence, load_trial_visibility, load_trial_3d_source,
    sync_gate_reason,
)


def test_sync_gate_reason():
    # Explicitly resolved sync (needs_review=False) -> allowed.
    assert sync_gate_reason({"sync_needs_review": False}) is None

    # needs_review True -> blocked.
    assert sync_gate_reason({"sync_needs_review": True}) is not None

    # Missing field entirely (older/never-refined trial) -> fail-safe blocked.
    assert sync_gate_reason({}) is not None

    # needs_review True but manually approved -> allowed.
    assert sync_gate_reason({
        "sync_needs_review": True, "manual_review_approved": True,
    }) is None

    # Approval alone (no needs_review key) does NOT bypass on its own unless
    # explicitly True — only True lets it through regardless of needs_review.
    assert sync_gate_reason({"manual_review_approved": True}) is None

    # approved=False must not accidentally allow it through.
    assert sync_gate_reason({
        "sync_needs_review": True, "manual_review_approved": False,
    }) is not None


def test_interpolate_window_gaps_mask():
    # (T=5, J=1, D=1) — a single short interior gap at frame 2.
    window = np.array([[[1.0]], [[2.0]], [[np.nan]], [[4.0]], [[5.0]]])
    valid = np.array([True, True, False, True, True])

    result = _interpolate_window_gaps(window, valid, max_gap=2)
    assert result is not None
    filled, interp_mask = result
    assert np.isclose(filled[2, 0, 0], 3.0), "gap must be linearly interpolated"
    assert interp_mask.shape == (5, 1)
    assert list(interp_mask[:, 0]) == [False, False, True, False, False]

    # Gap longer than max_gap -> window rejected (None).
    valid_long_gap = np.array([True, False, False, False, True])
    assert _interpolate_window_gaps(window, valid_long_gap, max_gap=1) is None

    # No gaps at all -> mask is all False, array returned unchanged.
    all_valid = np.array([True, True, True, True, True])
    clean_window = np.array([[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]]])
    filled2, mask2 = _interpolate_window_gaps(clean_window, all_valid)
    assert not mask2.any()
    assert np.array_equal(filled2, clean_window)


def test_load_confidence_and_visibility():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "pose2d_rtmpose.csv"
        row = {"frame": [0, 1]}
        for name in JOINT_NAMES:
            row[f"{name}_X"] = [0.0, 0.0]
            row[f"{name}_Y"] = [0.0, 0.0]
            row[f"{name}_conf"] = [0.9, 0.2]
            row[f"{name}_valid"] = [True, False]
        pd.DataFrame(row).to_csv(p, index=False)

        conf = load_trial_confidence(p)
        vis = load_trial_visibility(p)
        assert conf.shape == (2, N_JOINTS)
        assert vis.shape == (2, N_JOINTS)
        assert np.allclose(conf[0], 0.9) and np.allclose(conf[1], 0.2)
        assert vis[0].all() and not vis[1].any()

        # A pre-existing CSV without _conf/_valid columns must raise, not
        # silently default to "fully confident and visible".
        p_old = Path(d) / "pose2d_rtmpose_old.csv"
        old_row = {"frame": [0, 1]}
        for name in JOINT_NAMES:
            old_row[f"{name}_X"] = [0.0, 0.0]
            old_row[f"{name}_Y"] = [0.0, 0.0]
        pd.DataFrame(old_row).to_csv(p_old, index=False)
        try:
            load_trial_confidence(p_old)
            raise AssertionError("expected ValueError for a pre-confidence-tracking CSV")
        except ValueError as e:
            assert "predates confidence" in str(e)


def test_load_trial_3d_source():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "synced_60hz.csv"
        row = {"synced_video_frame": [0, 1]}
        for name in JOINT_NAMES:
            row[f"{name}_X"] = [0.0, 0.0]
            row[f"{name}_Y"] = [0.0, 0.0]
            row[f"{name}_Z"] = [0.0, 0.0]
            row[f"{name}_Source"] = ["c3d", "interpolated"]
        pd.DataFrame(row).to_csv(p, index=False)

        source = load_trial_3d_source(p)
        assert source.shape == (2, N_JOINTS)
        assert (source[0] == "c3d").all()
        assert (source[1] == "interpolated").all()

        # A synced_60hz.csv without _Source columns (predates propagation)
        # must raise, not silently default to "c3d" for every frame.
        p_old = Path(d) / "synced_60hz_old.csv"
        old_row = {"synced_video_frame": [0, 1]}
        for name in JOINT_NAMES:
            old_row[f"{name}_X"] = [0.0, 0.0]
            old_row[f"{name}_Y"] = [0.0, 0.0]
            old_row[f"{name}_Z"] = [0.0, 0.0]
        pd.DataFrame(old_row).to_csv(p_old, index=False)
        try:
            load_trial_3d_source(p_old)
            raise AssertionError("expected ValueError for a pre-provenance-tracking CSV")
        except ValueError as e:
            assert "predates provenance" in str(e)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
