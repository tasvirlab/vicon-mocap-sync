"""Deterministic synthetic checks of calibration geometry, label matching and clock alignment (no research data)."""
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.pnp_solver import reprojection_rms, solve_at_offset  # noqa: E402
from core.timestamp_sync import compute_timestamp_sync, parse_xcp_start_time  # noqa: E402
from core.vicon_io import interp_marker_at_time, resolve_labels  # noqa: E402

K = np.array([[1500.0, 0, 960.0], [0, 1500.0, 540.0], [0, 0, 1.0]])
DIST = np.zeros((1, 5))
RVEC = np.array([[0.1], [0.2], [0.05]])
TVEC = np.array([[0.2], [-0.1], [4.0]])


def _scene(n=14):
    rng = np.random.default_rng(0)
    obj = np.column_stack([rng.uniform(-0.6, 0.6, n), rng.uniform(-0.6, 0.6, n), rng.uniform(0.0, 1.8, n)])
    img = cv2.projectPoints(obj, RVEC, TVEC, K, DIST)[0].reshape(-1, 2)
    return obj, img


def test_pnp_recovers_known_pose():
    obj, img = _scene()
    rvec, tvec, rms, mask = solve_at_offset(obj, img, K, DIST)
    assert mask.all() and rms < 1e-3
    assert np.abs(rvec - RVEC).max() < 1e-3 and np.abs(tvec - TVEC).max() < 1e-3


def test_pnp_rejects_a_gross_outlier():
    obj, img = _scene()
    img[3] += 120.0
    rvec, tvec, rms, mask = solve_at_offset(obj, img, K, DIST)
    assert not mask[3] and mask.sum() == len(obj) - 1
    assert np.abs(tvec - TVEC).max() < 1e-2


def test_reprojection_rms_is_zero_at_truth_and_grows_with_error():
    obj, img = _scene()
    assert reprojection_rms(obj, img, RVEC, TVEC, K, DIST) < 1e-9
    assert reprojection_rms(obj, img, RVEC, TVEC + np.array([[0.05], [0], [0]]), K, DIST) > 1.0


def test_interpolation_between_frames_and_at_edges():
    frames = np.arange(5)
    data = np.column_stack([frames * 0.1, np.zeros(5), np.zeros(5)])
    assert np.allclose(interp_marker_at_time(data, frames, 100.0, 0.015), [0.15, 0, 0])
    assert np.allclose(interp_marker_at_time(data, frames, 100.0, -1.0), data[0])
    assert np.allclose(interp_marker_at_time(data, frames, 100.0, 9.0), data[-1])
    data[2, 0] = np.nan
    assert np.isnan(interp_marker_at_time(data, frames, 100.0, 0.015)).all()


def test_resolve_labels():
    labels = ["LSHO", "rsho", "LKNE:marker", "LANK"]
    assert resolve_labels(labels, ["LSHO", "RSHO", "LKNE"]) == ["LSHO", "rsho", "LKNE:marker"]
    for bad in (["MISSING"], ["L"]):
        try:
            resolve_labels(labels, bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_timestamp_sync_video_starts_first():
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    r = compute_timestamp_sync(t0, t0 + timedelta(seconds=2), 60.0, 100.0, 1800, 3000)
    assert abs(r["offset_seconds"] - 2.0) < 1e-9
    assert r["sync_start_video_frame"] == 120 and r["sync_start_vicon_frame"] == 0
    assert r["sync_end_video_frame"] - r["sync_start_video_frame"] == 1680


def test_timestamp_sync_vicon_starts_first_and_no_overlap_raises():
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    r = compute_timestamp_sync(t0 + timedelta(seconds=1), t0, 60.0, 100.0, 600, 3000)
    assert r["sync_start_video_frame"] == 0 and r["sync_start_vicon_frame"] == 100
    try:
        compute_timestamp_sync(t0, t0 + timedelta(seconds=100), 60.0, 100.0, 600, 3000)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-overlapping recordings")


def test_xcp_start_time_is_converted_to_utc():
    xml = '<Cameras><Camera><Capture START_TIME="2026-06-25 13:25:38.495 ST"/></Camera></Cameras>'
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.xcp"
        p.write_text(xml)
        dt = parse_xcp_start_time(p, 9.0)
    assert dt == datetime(2026, 6, 25, 4, 25, 38, 495000, tzinfo=timezone.utc)
