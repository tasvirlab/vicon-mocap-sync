"""
extrinsic_body.py
=================
Body-marker extrinsic calibration.

The user clicks N body joints across multiple video frames in the browser.
For each frame, corresponding Vicon 3D positions are fetched by linear
interpolation at the synchronised timestamp.
solvePnPRansac + LM refinement produces the camera-to-Vicon transform.

Output files:
  extrinsics.json       -- rvec, tvec_m, R, T_cam_vicon, camera_centre_vicon_m
  extrinsics.yaml       -- same, human-readable
  extrinsic_report.txt  -- summary report
  clicks.json           -- raw click data (re-solve without re-clicking)
  reprojection_check/   -- check_XXXXXX.jpg per accepted frame
                          (green circle = clicked, red cross = reprojected)

Joint sets
----------
  6-joint (no hips):   LSHO, RSHO, LKNE, RKNE, LANK, RANK
  8-joint (with hips): LSHO, RSHO, LHIP, RHIP, LKNE, RKNE, LANK, RANK
"""

from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .vicon_io import (
    read_vicon_trajectories, resolve_labels,
    interp_marker_at_time, rotate_frame,
)
from .pnp_solver import find_sync_and_solve, per_point_residuals

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[int, str], None]

JOINTS_6 = ["LSHO", "RSHO", "LKNE", "RKNE", "LANK", "RANK"]
JOINTS_8 = ["LSHO", "RSHO", "LHIP", "RHIP", "LKNE", "RKNE", "LANK", "RANK"]

JOINT_DESCRIPTIONS = {
    "LSHO": "L. Shoulder (acromion)",    "RSHO": "R. Shoulder (acromion)",
    "LHIP": "L. Hip (greater troch.)",   "RHIP": "R. Hip (greater troch.)",
    "LKNE": "L. Knee (lat. condyle)",    "RKNE": "R. Knee (lat. condyle)",
    "LANK": "L. Ankle (lat. malleolus)", "RANK": "R. Ankle (lat. malleolus)",
}

# BGR colour palette for the check images
_POINT_COLORS = [
    (0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0),
    (255, 0, 0), (255, 0, 255), (0, 128, 255), (128, 0, 255),
]


def get_joint_list(joint_count: int) -> List[str]:
    """Return ordered joint labels for the requested count (6 or 8)."""
    return JOINTS_8 if joint_count == 8 else JOINTS_6


def extract_video_frame(video_path: str, frame_idx: int,
                         rotate: str = "none") -> Tuple[bytes, int, float]:
    """
    Extract a single frame from a video.

    Args:
        video_path: Path to video file.
        frame_idx:  0-based frame index.
        rotate:     'none' | 'ccw90' | 'cw90' | '180'

    Returns:
        (jpeg_bytes, total_frames, fps)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = float(cap.get(cv2.CAP_PROP_FPS)) or 60.0
    idx   = max(0, min(frame_idx, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Cannot read frame {idx} from video.")
    frame = rotate_frame(frame, rotate)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes(), total, fps


def _load_intrinsics(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load K and dist from JSON or NPZ intrinsic file."""
    p = Path(path)
    if p.suffix == ".npz":
        d = np.load(str(p))
        return d["camera_matrix"].astype(np.float64), d["dist_coeffs"].astype(np.float64)
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    cm = d.get("camera_matrix") or d.get("K")
    K  = np.array(cm["matrix_3x3"] if isinstance(cm, dict) else cm,
                  dtype=np.float64)
    dc = (d.get("dist_coefficients") or d.get("dist_coefficients")
          or d.get("dist_coefs") or d.get("distortion_coefficients"))
    if isinstance(dc, dict):
        dc = dc.get("vector", [])
    dist = np.array(dc, dtype=np.float64).reshape(1, -1)
    return K, dist


def run_body_extrinsic(
    video_path:          str,
    vicon_path:          str,
    intrinsic_path:      str,
    clicks:              Dict[str, List[List[float]]],
    joint_count:         int,
    output_dir:          str,
    progress_cb:         Optional[ProgressCallback] = None,
    sync_offset_hint_s:  Optional[float] = None,
    rotate:              str = "none",
) -> dict:
    """
    Run body-marker extrinsic calibration.

    Args:
        video_path:         Path to calibration video.
        vicon_path:         Path to cleaned Vicon CSV.
        intrinsic_path:     Path to intrinsic_params JSON or NPZ.
        clicks:             {str(frame_idx): [[x, y], ...]} — N pts per frame.
        joint_count:        6 or 8.
        output_dir:         Parent directory to write timestamped results into.
        progress_cb:        Optional (percent, step_str) callback.
        sync_offset_hint_s: Wall-clock XCP sync hint (seconds).
        rotate:             Frame rotation ('none' | 'ccw90' | 'cw90' | '180').

    Returns:
        Result dict compatible with the ExtrinsicResult frontend type.
    """
    def _cb(pct: int, step: str):
        if progress_cb:
            progress_cb(pct, step)

    joints   = get_joint_list(joint_count)
    n_joints = len(joints)

    _cb(5, "Loading intrinsic parameters…")
    K, dist = _load_intrinsics(intrinsic_path)
    logger.info(f"Intrinsics: fx={K[0,0]:.1f} fy={K[1,1]:.1f}")

    _cb(15, "Loading Vicon marker data…")
    vicon    = read_vicon_trajectories(vicon_path)
    resolved = resolve_labels(vicon["labels"], joints)
    logger.info(f"Body joints resolved: {dict(zip(joints, resolved))}")

    # Build accepted dict: {int(frame_idx): [(x, y), ...]}
    accepted = {}
    for fidx_str, pts in clicks.items():
        if len(pts) == n_joints:
            accepted[int(fidx_str)] = [tuple(p) for p in pts]

    if len(accepted) < 3:
        raise ValueError(
            f"Need ≥3 fully-clicked frames (got {len(accepted)}). "
            f"Each frame needs exactly {n_joints} joints clicked."
        )

    _cb(25, "Reading video frame rate…")
    cap     = cv2.VideoCapture(video_path)
    fps_rgb = float(cap.get(cv2.CAP_PROP_FPS)) or 60.0
    ok, probe = cap.read()
    cap.release()
    if ok:
        probe = rotate_frame(probe, rotate)
        ih, iw = probe.shape[:2]
    else:
        iw, ih = 0, 0

    hint_info = (f"XCP hint {sync_offset_hint_s:+.3f} s ±2 s"
                 if sync_offset_hint_s is not None else "full grid-search")
    _cb(35, f"Solving ({len(accepted)} frames, {hint_info})…")

    best = find_sync_and_solve(
        accepted, fps_rgb, vicon, resolved, K, dist, verbose=False,
        offset_hint_s=sync_offset_hint_s,
    )
    best["label_order"] = resolved

    _cb(80, "Exporting results…")
    return _export(
        best, K, dist, joints, resolved, vicon,
        accepted, clicks, video_path, vicon_path,
        fps_rgb, (iw, ih), rotate, output_dir, _cb,
    )


# ── Output ───────────────────────────────────────────────────────────────────

def _write_yaml(path: Path, data: dict):
    """Minimal YAML serialiser (no extra dependency)."""
    def fmt(v):
        if isinstance(v, float): return f"{v:.10g}"
        if isinstance(v, list):  return "[" + ", ".join(fmt(x) for x in v) + "]"
        return str(v)

    def dump(o, ind=0):
        lines = []
        for k, v in o.items():
            pad = "  " * ind
            if isinstance(v, dict):
                lines.append(f"{pad}{k}:")
                lines.extend(dump(v, ind + 1))
            else:
                lines.append(f"{pad}{k}: {fmt(v)}")
        return lines

    with path.open("w", encoding="utf-8") as f:
        f.write(f"# extrinsic_body  {datetime.now().isoformat()}\n\n")
        f.write("\n".join(dump(data)) + "\n")


def _export(best, K, dist, joints, resolved, vicon,
            accepted, clicks_raw, video_path, vicon_path,
            fps_rgb, img_size, rotate, output_dir, cb):
    """Save all output files. Returns API result dict."""
    rvec = best["rvec"].reshape(3)
    tvec = best["tvec"].reshape(3)
    R, _ = cv2.Rodrigues(rvec)
    T    = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = tvec
    cam_centre = -R.T @ tvec   # camera position in Vicon world (metres)
    iw, ih     = img_size

    payload = {
        "rvec":                   rvec.tolist(),
        "tvec_m":                 tvec.tolist(),
        "R":                      R.flatten().tolist(),
        "T_cam_vicon":            T.flatten().tolist(),
        "camera_centre_vicon_m":  cam_centre.tolist(),
        "rms_reprojection_px":    round(float(best["rms"]), 6),
        "sync_offset_s":          round(float(best["offset_s"]), 6),
        "n_correspondences":      int(best["n_points"]),
        "n_inliers":              int(best.get("n_inliers", best["n_points"])),
        "resolved_labels":        resolved,
        "source_video":           str(video_path),
        "source_vicon_csv":       str(vicon_path),
        "rgb_fps":                fps_rgb,
        "image_width":            iw,
        "image_height":           ih,
        "units":                  "metres",
        "convention":             "X_cam = R @ X_vicon + t   (Vicon world -> camera)",
        "date":                   datetime.now().isoformat(),
    }

    ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = Path(output_dir) / f"extrinsic_body_{ts}"
    out.mkdir(parents=True, exist_ok=True)

    with (out / "extrinsics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    _write_yaml(out / "extrinsics.yaml", payload)

    with (out / "extrinsic_report.txt").open("w", encoding="utf-8") as f:
        f.write("Vicon -> Camera Extrinsic Calibration (Body Markers)\n")
        f.write("=" * 52 + "\n\n")
        f.write(f"Date            : {payload['date']}\n")
        f.write(f"Video           : {Path(str(video_path)).name}\n")
        f.write(f"Vicon CSV       : {Path(str(vicon_path)).name}\n")
        f.write(f"RGB fps         : {fps_rgb:.3f}\n")
        f.write(f"Joints          : {', '.join(joints)}\n")
        f.write(f"Correspondences : {best['n_points']}\n")
        f.write(f"Sync offset     : {best['offset_s']:+.4f} s\n")
        rms = best['rms']
        qual = "[GOOD]" if rms < 3 else "[OK]" if rms < 5 else "[POOR — recheck clicks]"
        f.write(f"RMS reproj err  : {rms:.4f} px {qual}\n")
        f.write(f"\nCamera centre (m): "
                f"[{cam_centre[0]:.3f}, {cam_centre[1]:.3f}, {cam_centre[2]:.3f}]  "
                f"dist={np.linalg.norm(cam_centre):.3f} m\n")
        f.write("\nT_cam_vicon (4x4):\n")
        for row in T:
            f.write("  " + "  ".join(f"{v:10.5f}" for v in row) + "\n")

    with (out / "clicks.json").open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in clicks_raw.items()}, f, indent=2)

    cb(88, "Saving reprojection check images…")
    _render_checks(out, accepted, best, resolved, vicon, fps_rgb,
                   K, dist, video_path, rotate)

    files = {
        "extrinsics_json":    str(out / "extrinsics.json"),
        "extrinsics_yaml":    str(out / "extrinsics.yaml"),
        "extrinsic_report":   str(out / "extrinsic_report.txt"),
        "clicks_json":        str(out / "clicks.json"),
        "reprojection_check": str(out / "reprojection_check"),
    }

    return {
        "method":          "BODY_MARKERS",
        "quality":         _quality(rms),
        "rms_px":          round(float(rms), 6),
        "n_inliers":       int(best.get("n_inliers", best["n_points"])),
        "n_points":        int(best["n_points"]),
        "sync_offset_s":   round(float(best["offset_s"]), 4),
        "sync_method":     best.get("sync_method", "GRID_SEARCH"),
        "xcp_hint_s":      (round(best["xcp_hint_s"], 4)
                            if best.get("xcp_hint_s") is not None else None),
        "joint_count":     len(joints),
        "joints":          joints,
        "camera_matrix": {
            "fx": round(float(K[0,0]), 4), "fy": round(float(K[1,1]), 4),
            "cx": round(float(K[0,2]), 4), "cy": round(float(K[1,2]), 4),
            "matrix_3x3": K.tolist(),
        },
        "distortion_coefficients": {"vector": dist.ravel().tolist()},
        "rvec":     rvec.tolist(),
        "tvec_mm":  [round(v * 1000, 4) for v in tvec.tolist()],
        "R_matrix": R.tolist(),
        "world_to_cam_4x4": T.tolist(),
        "cam_to_world_4x4": np.linalg.inv(T).tolist(),
        "camera_position_world_mm": {
            "x": round(float(cam_centre[0]) * 1000, 3),
            "y": round(float(cam_centre[1]) * 1000, 3),
            "z": round(float(cam_centre[2]) * 1000, 3),
            "distance_from_origin_m": round(float(np.linalg.norm(cam_centre)), 4),
        },
        "euler_angles_deg": _euler(R),
        "per_marker_errors": _per_errors(best, joints, K, dist),
        "output_dir":  str(out),
        "files":       files,
    }


_INVERSE_ROTATE = {
    "ccw90": "cw90",
    "cw90":  "ccw90",
    "180":   "180",
    "none":  "none",
}


def _render_checks(out: Path, accepted: dict, best: dict, resolved: list,
                   vicon: dict, fps_rgb: float, K, dist,
                   video_path: str, rotate: str, max_render: int = 20):
    """
    Draw clicked (green) vs reprojected (red) markers and save in LANDSCAPE.

    Strategy (robust regardless of whether OpenCV auto-applies rotation metadata):
      1. Read raw frame from video.
      2. Apply the SAME rotation the user used in the app → frame is now in
         the exact coordinate space where clicks were placed.
      3. Draw clicked points and reprojected points DIRECTLY (no coord transform).
      4. Rotate the annotated frame BACK to landscape using the inverse rotation.
      5. Save the landscape frame — always landscape output.
    """
    rp_dir = out / "reprojection_check"
    rp_dir.mkdir(parents=True, exist_ok=True)
    cap  = cv2.VideoCapture(str(video_path))
    rvec = best["rvec"]
    tvec = best["tvec"]

    for n, (fidx, pts) in enumerate(sorted(accepted.items())):
        if n >= max_render:
            break
        t_vic = fidx / fps_rgb + best["offset_s"]
        obj   = np.asarray(
            [interp_marker_at_time(vicon["data"][lab],
                                   vicon["frames"], vicon["fps"], t_vic)
             for lab in resolved], dtype=np.float64)
        if np.any(np.isnan(obj)):
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue

        # Step 2: bring frame into the same space as the user's clicks
        frame = rotate_frame(frame, rotate)

        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)

        # Step 3: draw directly — no coordinate transformation needed
        for i, ((cx, cy), (px, py)) in enumerate(zip(pts, proj)):
            col = _POINT_COLORS[i % len(_POINT_COLORS)]
            cv2.circle(frame, (int(cx), int(cy)), 10, (0, 255, 0), 2)
            cv2.drawMarker(frame, (int(px), int(py)),
                           (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
            cv2.line(frame, (int(cx), int(cy)), (int(px), int(py)),
                     (255, 0, 0), 1)
            cv2.putText(frame, resolved[i], (int(cx) + 8, int(cy) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

        cv2.putText(frame, "green=clicked  red=reprojected",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Step 4: save in display orientation (landscape — same as user clicked on)
        cv2.imwrite(str(rp_dir / f"check_{fidx:06d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    cap.release()
    logger.info(f"Saved {min(len(accepted), max_render)} landscape reprojection "
                f"check images to {rp_dir}")


def _quality(rms: float) -> str:
    if rms < 1.0: return "EXCELLENT"
    if rms < 3.0: return "GOOD"
    if rms < 5.0: return "ACCEPTABLE"
    return "POOR — recalibrate"


def _euler(R: np.ndarray) -> dict:
    sy = float(np.sqrt(R[0,0]**2 + R[1,0]**2))
    if sy > 1e-6:
        rx = float(np.degrees(np.arctan2(R[2,1], R[2,2])))
        ry = float(np.degrees(np.arctan2(-R[2,0], sy)))
        rz = float(np.degrees(np.arctan2(R[1,0], R[0,0])))
    else:
        rx = float(np.degrees(np.arctan2(-R[1,2], R[1,1])))
        ry = float(np.degrees(np.arctan2(-R[2,0], sy)))
        rz = 0.0
    return {"rx": round(rx, 3), "ry": round(ry, 3), "rz": round(rz, 3)}


def _per_errors(best: dict, joints: list, K, dist) -> list:
    """Compute per-joint mean reprojection error across all accepted frames."""
    if best.get("obj") is None or best.get("img") is None:
        return []
    try:
        errs = per_point_residuals(
            best["obj"], best["img"], best["rvec"], best["tvec"], K, dist)
        n = len(joints)
        result = []
        for i, err in enumerate(errs):
            result.append({
                "marker":    joints[i % n],
                "error_px":  round(float(err), 4),
            })
        return result
    except Exception:
        return []
