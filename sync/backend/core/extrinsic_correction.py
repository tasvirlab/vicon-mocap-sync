"""
extrinsic_correction.py
========================
Manual spatial correction for a camera's extrinsic pose (rvec/tvec), applied
on top of whatever the calibration backend produced.

Why this exists: intrinsic/extrinsic PnP solves can leave a small but
systematic whole-skeleton offset or tilt (residual R,t error) that shows up as
"the skeleton is close but doesn't quite sit on the body" in the Overlay
preview. Unlike per-joint click noise, this is a single rigid 6-DOF error —
correcting it once and saving it into extrinsics.json fixes every task/trial
for that subject in one step, since intrinsic/extrinsic are session-wide
constants (the camera didn't move between tasks).

The correction is composed in VIEW SPACE (the camera's own frame) so the six
numbers map directly to what the user sees on screen: dx/dy shift the
skeleton left-right/up-down, dz shifts it closer/farther (changing apparent
scale), and droll/dpitch/dyaw tilt it — exactly the degrees of freedom a human
eyeballing the live preview needs to nudge.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# A delta this small is not a real correction — treat it as "no delta" so a
# preview request with all-zero fields (the common case) skips the extra
# Rodrigues round-trip.
_EPS = 1e-9


def is_zero_delta(delta: Optional[dict]) -> bool:
    if not delta:
        return True
    return all(abs(float(delta.get(k, 0.0))) < _EPS
               for k in ("dx", "dy", "dz", "droll", "dpitch", "dyaw"))


def compose_delta(rvec: np.ndarray, tvec: np.ndarray, delta: dict
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a view-space rigid correction to an existing (rvec, tvec).

    delta: {dx, dy, dz} in metres, {droll, dpitch, dyaw} in degrees — all
    default 0.0 if missing.

    Returns:
        (new_rvec, new_tvec), each (3, 1) float64 — same shape convention as
        core/overlay_pipeline.py's _load_extrinsics().
    """
    if is_zero_delta(delta):
        return rvec, tvec

    droll  = np.radians(float(delta.get("droll", 0.0)))
    dpitch = np.radians(float(delta.get("dpitch", 0.0)))
    dyaw   = np.radians(float(delta.get("dyaw", 0.0)))
    d_t    = np.array([[float(delta.get("dx", 0.0))],
                        [float(delta.get("dy", 0.0))],
                        [float(delta.get("dz", 0.0))]], dtype=np.float64)

    # View-space rotation, camera frame (OpenCV convention: X right, Y down,
    # Z forward). Composed as three explicit elementary rotations rather than
    # one combined Rodrigues vector — the latter is only a valid small-angle
    # approximation, and the UI sends an accumulated running total (which can
    # grow well past "small") rather than tiny per-click increments.
    cr, sr = np.cos(droll),  np.sin(droll)
    cp, sp = np.cos(dpitch), np.sin(dpitch)
    cy, sy = np.cos(dyaw),   np.sin(dyaw)
    R_roll  = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])   # about Z
    R_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])   # about X
    R_yaw   = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])   # about Y
    d_R = R_yaw @ R_pitch @ R_roll

    R, _ = cv2.Rodrigues(rvec)
    new_R = d_R @ R
    new_t = d_R @ tvec + d_t

    new_rvec, _ = cv2.Rodrigues(new_R)
    return new_rvec.reshape(3, 1), new_t.reshape(3, 1)


def _derived_fields(rvec: np.ndarray, tvec: np.ndarray) -> dict:
    """
    Recompute the informational fields the calibration backend writes alongside
    rvec/tvec (R, T_cam_vicon, camera_centre_vicon_m) so they stay consistent
    with a corrected pose. The sync backend itself only ever reads rvec/tvec (see
    _load_extrinsics in overlay_pipeline.py) — this is purely so the file
    doesn't silently go stale for any other tool that reads it later.
    """
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3]  = t
    camera_centre = (-R.T @ t)
    return {
        "R": R.flatten().tolist(),
        "T_cam_vicon": T.flatten().tolist(),
        "camera_centre_vicon_m": camera_centre.tolist(),
    }


def save_correction(extrinsics_path: Path, delta: dict) -> dict:
    """
    Load extrinsics.json, permanently compose `delta` into rvec/tvec, recompute
    the derived fields, and write the file back in place.

    Every consumer of this file (run_overlay, preview_single_frame, and any
    downstream tool) reads straight from disk, so this is the single place a
    manual spatial correction needs to be applied to affect every task for
    the subject that shares this extrinsics.json.

    Returns the updated extrinsics dict (for the API response to confirm).
    """
    with open(extrinsics_path, encoding="utf-8") as f:
        data = json.load(f)

    rvec = np.asarray(data["rvec"],   dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(data["tvec_m"], dtype=np.float64).reshape(3, 1)
    new_rvec, new_tvec = compose_delta(rvec, tvec, delta)

    data["rvec"]    = new_rvec.reshape(3).tolist()
    data["tvec_m"]  = new_tvec.reshape(3).tolist()
    data.update(_derived_fields(new_rvec, new_tvec))

    history = data.setdefault("manual_spatial_corrections", [])
    history.append({k: float(delta.get(k, 0.0))
                     for k in ("dx", "dy", "dz", "droll", "dpitch", "dyaw")})

    with open(extrinsics_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved manual spatial correction to {extrinsics_path}: {delta}")
    return data


def _self_check():
    """Zero delta is a no-op; dz shifts a point directly ahead of the camera
    by exactly dz in depth; droll=90 deg rotates the image-plane axes into
    each other, matching a quarter-turn camera roll."""
    rvec0 = np.zeros((3, 1))
    tvec0 = np.zeros((3, 1))

    r, t = compose_delta(rvec0, tvec0, {})
    assert np.allclose(r, rvec0) and np.allclose(t, tvec0)

    r, t = compose_delta(rvec0, tvec0, {"dz": 2.0})
    assert np.allclose(t.flatten(), [0, 0, 2.0])

    r, t = compose_delta(rvec0, tvec0, {"droll": 90.0})
    R, _ = cv2.Rodrigues(r)
    # A point straight right of the camera (X axis) should land straight down
    # (Y axis) after a 90 deg roll about the viewing (Z) axis.
    assert np.allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-9)

    print("extrinsic_correction self-check OK")


if __name__ == "__main__":
    _self_check()
