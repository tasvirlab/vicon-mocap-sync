"""
extrinsic_calibration.py
========================
Full extrinsic calibration pipeline:
  load intrinsics → detect ChArUco corners in video frame → load Vicon markers
  → build 2D/3D correspondences → solvePnP(Ransac) → validate → export.

The board corners observed in the video frame provide 2D image coordinates;
Vicon markers physically attached to the board corners provide 3D world coords.
"""

from __future__ import annotations
import cv2
import json
import logging
import math
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from itertools import permutations as _perms
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .vicon_loader import load_vicon_markers
from .vicon_io import rotate_frame as _rotate_frame

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, str], None]

_DICT_MAP = {
    "DICT_4X4_50":         cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100":        cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50":         cv2.aruco.DICT_5X5_50,
    "DICT_6X6_250":        cv2.aruco.DICT_6X6_250,
    "DICT_7X7_250":        cv2.aruco.DICT_7X7_250,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

_PNP_METHODS = {
    "ITERATIVE": cv2.SOLVEPNP_ITERATIVE,
    "EPNP":      cv2.SOLVEPNP_EPNP,
    "SQPNP":     cv2.SOLVEPNP_SQPNP,
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ExtrinsicResult:
    rvec:          np.ndarray
    tvec:          np.ndarray
    R:             np.ndarray
    world_to_cam:  np.ndarray
    cam_to_world:  np.ndarray
    rms_px:        float = 0.0
    n_inliers:     int   = 0
    n_points:      int   = 0
    method:        str   = "ITERATIVE"
    inlier_mask:   Optional[np.ndarray] = None

    @property
    def camera_position_world(self) -> np.ndarray:
        return (-self.R.T @ self.tvec.ravel()).ravel()

    @property
    def euler_angles_deg(self) -> Tuple[float, float, float]:
        R = self.R
        sy = float(np.sqrt(R[0, 0]**2 + R[1, 0]**2))
        if sy > 1e-6:
            rx = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
            ry = np.degrees(np.arctan2(-R[2, 0], sy))
            rz = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        else:
            rx = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
            ry = np.degrees(np.arctan2(-R[2, 0], sy))
            rz = 0.0
        return float(rx), float(ry), float(rz)

    def quality_label(self) -> str:
        if self.rms_px < 1.0:
            return "EXCELLENT"
        elif self.rms_px < 3.0:
            return "GOOD"
        elif self.rms_px < 8.0:
            return "ACCEPTABLE"
        else:
            return "POOR — recalibrate"


def _build_extrinsic(rvec, tvec, rms, n_inliers, n_points,
                     method, inlier_mask=None) -> ExtrinsicResult:
    R, _ = cv2.Rodrigues(rvec)
    W2C = np.eye(4, dtype=np.float64)
    W2C[:3, :3] = R
    W2C[:3,  3] = tvec.ravel()
    return ExtrinsicResult(
        rvec=rvec.reshape(3, 1).astype(np.float64),
        tvec=tvec.reshape(3, 1).astype(np.float64),
        R=R, world_to_cam=W2C, cam_to_world=np.linalg.inv(W2C),
        rms_px=rms, n_inliers=n_inliers, n_points=n_points,
        method=method, inlier_mask=inlier_mask,
    )


# ── Main pipeline function ────────────────────────────────────────────────────

def run_extrinsic_calibration(
    video_path:     str,
    vicon_path:     str,
    intrinsic_path: str,
    output_dir:     str,
    config:         dict,
    progress_cb:    Optional[ProgressCallback] = None,
) -> dict:
    """
    Run the full extrinsic calibration pipeline.

    Args:
        video_path:     Path to calibration video.
        vicon_path:     Path to Vicon CSV or C3D file.
        intrinsic_path: Path to intrinsic_params.json or .npz.
        output_dir:     Directory to write results into.
        config:         Dict with keys: board, sync, solver, export.
        progress_cb:    Optional (percent, step) callback.

    Returns:
        Result dict compatible with ExtrinsicResult schema.
    """
    def _cb(pct: int, step: str):
        if progress_cb:
            progress_cb(pct, step)

    board_cfg      = config.get("board", {})
    sync_cfg       = config.get("sync", {})
    solver_cfg     = config.get("solver", {})
    rigid_body_cfg = config.get("rigid_body_pose")
    rotate         = config.get("rotate", "none")

    # ── Load intrinsics ───────────────────────────────────────────────────
    _cb(5, "Loading intrinsic parameters…")
    K, D, w_cal, h_cal = _load_intrinsics(intrinsic_path)
    logger.info(
        f"Intrinsic K loaded — fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, "
        f"cx={K[0,2]:.2f}, cy={K[1,2]:.2f}, "
        f"calib_frame={w_cal}×{h_cal}"
    )

    # ── Build ChArUco board ───────────────────────────────────────────────
    _cb(10, "Building ChArUco board…")
    dict_name  = board_cfg.get("dictionary", "DICT_6X6_250")
    aruco_dict = cv2.aruco.getPredefinedDictionary(_DICT_MAP.get(dict_name, cv2.aruco.DICT_6X6_250))
    board = cv2.aruco.CharucoBoard(
        (board_cfg.get("cols", 9), board_cfg.get("rows", 6)),
        board_cfg.get("square_length_mm", 25.0) / 1000.0,
        board_cfg.get("marker_length_mm", 18.5) / 1000.0,
        aruco_dict,
    )
    det_params     = cv2.aruco.DetectorParameters()
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_det    = cv2.aruco.CharucoDetector(board, charuco_params, det_params)

    # ── Extract the reference video frame + detect ChArUco ───────────────
    _cb(15, "Searching for ChArUco board in video…")
    video_frame_idx = int(sync_cfg.get("video_frame", 0))

    # outer_corner_mapping: Kabsch board→Vicon orient, then direct world solvePnP.
    # marker_mapping:       direct solvePnP using Vicon markers at inner corners.
    # Only marker_mapping needs the scanner to seek specific corner IDs.
    marker_mapping:       Optional[dict] = config.get("marker_mapping")
    outer_corner_mapping: Optional[dict] = config.get("outer_corner_mapping")
    required_corner_ids:  Optional[set]  = None
    if marker_mapping and not outer_corner_mapping:
        required_corner_ids = {int(k) for k in marker_mapping.keys()}

    # For outer_corner_mapping: scan the entire video and pick the frame with
    # the MOST detected corners.  More spread-out corners dramatically improve
    # the depth estimate from solvePnP.  Only skip the full scan when the user
    # has explicitly pinned a frame (video_frame_idx != 0).
    prefer_max = bool(outer_corner_mapping or rigid_body_cfg) and (video_frame_idx == 0)

    frame, img_size, found_frame_idx, charuco_corners, charuco_ids = \
        _find_charuco_frame(video_path, video_frame_idx, charuco_det,
                            required_ids=required_corner_ids,
                            prefer_max_corners=prefer_max,
                            rotate=rotate)

    if found_frame_idx != video_frame_idx and not prefer_max:
        logger.warning(
            f"Board not found at frame {video_frame_idx}; "
            f"using nearest detected frame {found_frame_idx}"
        )
    ids_list = charuco_ids.ravel().tolist() if charuco_ids is not None else []
    ids_str  = ", ".join(str(i) for i in sorted(ids_list))
    _cb(25, f"ChArUco detected at frame {found_frame_idx} ({len(charuco_corners)} corners: IDs {ids_str})")
    logger.info(
        f"Detected {len(charuco_corners)} ChArUco corners in frame {found_frame_idx} — "
        f"corner IDs: [{ids_str}]  "
        f"(use these IDs in the Vicon Marker Mapping panel)"
    )
    video_frame_idx = found_frame_idx

    # ── Adapt K to the extrinsic video frame's decoded dimensions ─────────
    # charuco_corners are in the decoded frame's pixel space.  K must be in
    # the same space for solvePnP to give correct R, t.
    h_frame, w_frame = frame.shape[:2]   # actual decoded dimensions
    img_size = (w_frame, h_frame)        # overwrite coded size with actual
    if (w_cal != w_frame or h_cal != h_frame) and w_cal > 0:
        logger.info(
            f"Resolution mismatch — intrinsic calibration: {w_cal}×{h_cal}, "
            f"extrinsic video frame: {w_frame}×{h_frame}. "
            "Adapting K to match extrinsic frame…"
        )
        K = _adapt_K_to_frame(K, w_cal, h_cal, w_frame, h_frame)
    else:
        logger.info(f"K resolution matches extrinsic frame ({w_frame}×{h_frame}) — no adaptation needed.")

    # Get 3D board-space coords for detected corners
    obj_pts_board, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
    if obj_pts_board is None or len(obj_pts_board) < 4:
        raise ValueError("Board corner matching failed — too few correspondences.")

    img_pts = img_pts.reshape(-1, 2)

    # ── Load Vicon markers (skipped in rigid_body_pose mode) ─────────────
    _cb(35, "Loading Vicon marker data…")
    vicon_frame_idx = int(sync_cfg.get("vicon_frame", 0))
    # rigid_body_pose supplies the board transform directly — no CSV needed.
    _needs_csv = (rigid_body_cfg is None) or bool(outer_corner_mapping) or bool(marker_mapping)
    if _needs_csv:
        if not vicon_path or not Path(vicon_path).exists():
            raise ValueError(
                "Vicon CSV/C3D is required for outer_corner_mapping and marker_mapping modes. "
                "For rigid_body_pose mode, the Vicon file is not used."
            )
        markers, vicon_fps = load_vicon_markers(vicon_path)
        if not markers:
            raise ValueError("No markers found in Vicon file.")
    else:
        markers = {}
        logger.info("rigid_body_pose mode: board transform is in config — Vicon CSV skipped.")

    # ── Build 3D correspondences and solve ───────────────────────────────
    _cb(45, "Building 3D correspondence set…")
    obj_pts_board_mm = obj_pts_board.reshape(-1, 3) * 1000.0  # m → mm

    if rigid_body_cfg:
        # ── Rigid-body mode: Vicon-tracked board pose → direct world solvePnP ─
        # Retroreflective markers sit on the board BORDER or BACK so they don't
        # occlude ChArUco corners.  Vicon Nexus tracks the board as a Rigid Body
        # and provides its full 6-DOF pose.  Supply translation_mm and
        # rotation_matrix (or rvec) in config (exported from Nexus at the
        # calibration frame, or captured live via the Vicon DataStream SDK).
        _cb(50, "Computing board-to-world transform from Vicon rigid body pose…")
        extr, obj_pts_world, img_pts_mapped, labels = _rigid_body_calibration(
            rigid_body_cfg, K, D, obj_pts_board_mm, img_pts, solver_cfg,
        )
    elif outer_corner_mapping:
        # ── Kabsch board→Vicon, then direct world-space solvePnP ──────────────
        # Works even when board outer corners are outside the video frame.
        _cb(50, "Fitting board→Vicon transform via Kabsch…")
        extr, obj_pts_world, img_pts_mapped, labels = _outer_corner_calibration(
            board_cfg, markers, vicon_frame_idx,
            outer_corner_mapping, K, D, obj_pts_board_mm, img_pts,
            solver_cfg,
        )
    elif marker_mapping:
        # ── Inner-corner solvePnP (Vicon markers placed at inner corners) ──
        obj_pts_world, img_pts_mapped, labels = _build_vicon_correspondences(
            marker_mapping, markers, vicon_frame_idx, charuco_corners, charuco_ids, board
        )
        logger.info(f"Correspondences: {len(obj_pts_world)} 3D↔2D pairs")
        _cb(55, "Running solvePnP (Vicon-world coords)…")
        extr = _solve_pnp(obj_pts_world, img_pts_mapped, K, D, solver_cfg)
    else:
        # ── Board-only (no Vicon mapping — board-relative result) ─────────
        obj_pts_world = obj_pts_board_mm
        img_pts_mapped = img_pts
        labels = [f"corner_{i}" for i in range(len(obj_pts_world))]
        logger.info(f"Correspondences: {len(obj_pts_world)} 3D↔2D pairs (board-local)")
        _cb(55, "Running solvePnP (board-local coords)…")
        extr = _solve_pnp(obj_pts_world, img_pts_mapped, K, D, solver_cfg)

    # Per-point errors
    proj_pts, _ = cv2.projectPoints(
        obj_pts_world.reshape(-1, 1, 3), extr.rvec, extr.tvec, K, D
    )
    errors = np.linalg.norm(proj_pts.reshape(-1, 2) - img_pts_mapped, axis=1)
    logger.info(
        f"Extrinsic solved — RMS={extr.rms_px:.4f} px  "
        f"[{extr.quality_label()}]  method={extr.method}"
    )

    # ── Overlay visualization ─────────────────────────────────────────────
    _cb(70, "Generating overlay visualization…")
    overlay = _draw_overlay(frame.copy(), obj_pts_world, img_pts_mapped,
                            proj_pts.reshape(-1, 2), extr, K, D)

    # ── Export ────────────────────────────────────────────────────────────
    _cb(80, "Exporting results…")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir   = Path(output_dir) / f"extrinsic_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pos = extr.camera_position_world
    rx, ry, rz = extr.euler_angles_deg
    D_vec = D.ravel()

    result_data = {
        "method":  extr.method,
        "quality": extr.quality_label(),
        # Frame dimensions used for solvePnP — needed by the sync backend to adapt K
        # for overlay videos at a different resolution or orientation.
        "calibration_frame_width":  w_frame,
        "calibration_frame_height": h_frame,
        "reprojection": {
            "rms_px":    round(extr.rms_px, 6),
            "n_inliers": extr.n_inliers,
            "n_points":  extr.n_points,
            "per_marker": [
                {"marker": lbl, "error_px": round(float(e), 4)}
                for lbl, e in zip(labels, errors)
            ],
        },
        # Intrinsic parameters — included so this file is self-contained for projection
        "camera_matrix": {
            "fx": round(float(K[0, 0]), 6), "fy": round(float(K[1, 1]), 6),
            "cx": round(float(K[0, 2]), 6), "cy": round(float(K[1, 2]), 6),
            "matrix_3x3": K.tolist(),
        },
        "distortion_coefficients": {
            "k1": round(float(D_vec[0]), 8),
            "k2": round(float(D_vec[1]), 8) if len(D_vec) > 1 else 0.0,
            "vector": D_vec.tolist(),
        },
        # Extrinsic parameters
        "rvec":  [round(v, 8) for v in extr.rvec.ravel().tolist()],
        "tvec_mm": [round(v, 6) for v in extr.tvec.ravel().tolist()],
        "R_matrix": [[round(v, 8) for v in row] for row in extr.R.tolist()],
        "world_to_cam_4x4": extr.world_to_cam.tolist(),
        "cam_to_world_4x4": extr.cam_to_world.tolist(),
        "camera_position_world_mm": {
            "x": round(float(pos[0]), 3),
            "y": round(float(pos[1]), 3),
            "z": round(float(pos[2]), 3),
        },
        "euler_angles_deg": {"rx": round(rx, 3), "ry": round(ry, 3), "rz": round(rz, 3)},
    }

    np.savez_compressed(
        str(out_dir / "extrinsic_params.npz"),
        # Intrinsic parameters included for self-contained projection
        camera_matrix=K,
        dist_coeffs=D,
        # Frame dimensions used during solvePnP (adapted K is valid for this size)
        calibration_frame_width=np.array([w_frame]),
        calibration_frame_height=np.array([h_frame]),
        # Extrinsic parameters
        rvec=extr.rvec, tvec=extr.tvec, R=extr.R,
        world_to_cam=extr.world_to_cam, cam_to_world=extr.cam_to_world,
        rms_px=np.array([extr.rms_px]),
    )
    with open(out_dir / "extrinsic_params.json", "w") as f:
        json.dump(result_data, f, indent=2)

    cv2.imwrite(str(out_dir / "overlay_frame.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])

    _cb(88, "Generating reprojection plot…")
    _save_reprojection_plot(errors, labels,
                             config.get("rms_warn_px", 3.0),
                             config.get("rms_bad_px", 8.0),
                             out_dir / "reprojection_plot.png")
    if len(img_pts_mapped) >= 4:
        _save_error_heatmap(errors, img_pts_mapped, img_size,
                             out_dir / "error_heatmap.png")

    _save_extrinsic_report(extr, errors, labels, config,
                            video_path, vicon_path, video_frame_idx,
                            vicon_frame_idx, out_dir)

    files = {
        "extrinsic_params_npz":  str(out_dir / "extrinsic_params.npz"),
        "extrinsic_params_json": str(out_dir / "extrinsic_params.json"),
        "overlay_frame":         str(out_dir / "overlay_frame.jpg"),
        "reprojection_plot":     str(out_dir / "reprojection_plot.png"),
        "error_heatmap":         str(out_dir / "error_heatmap.png"),
        "alignment_report":      str(out_dir / "alignment_report.txt"),
    }

    return {
        "method":   extr.method,
        "quality":  extr.quality_label(),
        "rms_px":   round(extr.rms_px, 6),
        "n_inliers": extr.n_inliers,
        "n_points":  extr.n_points,
        "calibration_frame_width":  w_frame,
        "calibration_frame_height": h_frame,
        "camera_matrix": {
            "fx": round(float(K[0, 0]), 6), "fy": round(float(K[1, 1]), 6),
            "cx": round(float(K[0, 2]), 6), "cy": round(float(K[1, 2]), 6),
            "matrix_3x3": K.tolist(),
        },
        "distortion_coefficients": {
            "vector": D_vec.tolist(),
        },
        "rvec":      [round(v, 8) for v in extr.rvec.ravel().tolist()],
        "tvec_mm":   [round(v, 6) for v in extr.tvec.ravel().tolist()],
        "R_matrix":  [[round(v, 8) for v in row] for row in extr.R.tolist()],
        "world_to_cam_4x4": extr.world_to_cam.tolist(),
        "cam_to_world_4x4": extr.cam_to_world.tolist(),
        "camera_position_world_mm": {
            "x": round(float(pos[0]), 3),
            "y": round(float(pos[1]), 3),
            "z": round(float(pos[2]), 3),
            "distance_from_origin_m": round(float(np.linalg.norm(pos)) / 1000.0, 4),
        },
        "euler_angles_deg": {"rx": round(rx, 3), "ry": round(ry, 3), "rz": round(rz, 3)},
        "per_marker_errors": [
            {"marker": lbl, "error_px": round(float(e), 4)}
            for lbl, e in zip(labels, errors)
        ],
        # Diagnostics for the Vicon Marker Mapping panel
        "detected_frame_idx":   found_frame_idx,
        "detected_corner_ids":  sorted(ids_list),
        "output_dir": str(out_dir),
        "files": files,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_intrinsics(path: str) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Load K and D from an intrinsic file (NPZ or JSON).

    Handles two JSON formats:
      - Flat format: camera_matrix is a raw 3×3 list,
        dist key is 'dist_coefficients'.
      - Nested format: camera_matrix is a dict with
        'matrix_3x3', dist key is 'distortion_coefficients.vector'.

    Returns:
        (K, D, calib_w, calib_h) — calib_w/h are 0×0 when not stored.
    """
    p = Path(path)
    if p.suffix == ".npz":
        data = np.load(str(p))
        K = data["camera_matrix"]
        D = data["dist_coeffs"]
        calib_w = calib_h = 0
        if "image_size" in data:
            calib_w, calib_h = int(data["image_size"][0]), int(data["image_size"][1])
        return K, D, calib_w, calib_h

    elif p.suffix == ".json":
        with open(p, encoding="utf-8") as f:
            d = json.load(f)

        # Camera matrix — raw list or nested dict
        cm = d.get("camera_matrix") or d.get("K")
        if cm is None:
            raise ValueError(f"No 'camera_matrix' key in {p.name}")
        K = np.array(cm["matrix_3x3"] if isinstance(cm, dict) else cm,
                     dtype=np.float64)
        if K.ndim == 1:
            K = K.reshape(3, 3)

        # Distortion — try all known key names
        dc = (d.get("dist_coefficients")        # flat format
              or d.get("distortion_coefficients") # nested dict
              or d.get("dist_coefs"))
        if dc is None:
            raise ValueError(f"No distortion key in {p.name}")
        D = np.array(dc["vector"] if isinstance(dc, dict) else dc,
                     dtype=np.float64)

        # Image size — from whichever key is present
        info  = d.get("calibration_info", {})
        calib_w = int(d.get("image_width",  info.get("image_width",
                     d.get("calibration_frame_width",  0))))
        calib_h = int(d.get("image_height", info.get("image_height",
                     d.get("calibration_frame_height", 0))))
        return K, D, calib_w, calib_h

    else:
        raise ValueError(f"Unsupported intrinsic file format: {p.suffix}")


def _adapt_K_to_frame(
    K: np.ndarray,
    w_cal: int,
    h_cal: int,
    w_frame: int,
    h_frame: int,
) -> np.ndarray:
    """Scale (and if necessary rotate) K from calibration resolution to frame resolution.

    Handles two cases:
      1. Same orientation (both landscape or both portrait): pure proportional scale.
      2. Orientation change (landscape ↔ portrait): swap fx/fy, remap cx/cy, then scale.
         Assumes a 90° CW rotation (standard for iPhone portrait videos where
         the encoded stream is landscape with a ``rotate=90`` metadata tag).

    Args:
        K:       (3, 3) camera matrix calibrated at (w_cal × h_cal).
        w_cal:   Calibration frame width  (pixels).
        h_cal:   Calibration frame height (pixels).
        w_frame: Target extrinsic video frame width  (decoded by cv2).
        h_frame: Target extrinsic video frame height (decoded by cv2).

    Returns:
        K_adapted: (3, 3) camera matrix valid for (w_frame × h_frame) frames.
    """
    if w_cal <= 0 or h_cal <= 0:
        logger.warning(
            "adapt_K_to_frame: calibration dimensions unknown — "
            "K used as-is.  Re-run intrinsic calibration and re-export."
        )
        return K.astype(np.float64).copy()

    K = K.astype(np.float64).copy()

    cal_landscape   = w_cal   >= h_cal
    frame_landscape = w_frame >= h_frame

    if cal_landscape == frame_landscape:
        # Same orientation — proportional scale
        sx = w_frame / w_cal
        sy = h_frame / h_cal
        K[0, 0] *= sx   # fx
        K[0, 2] *= sx   # cx
        K[1, 1] *= sy   # fy
        K[1, 2] *= sy   # cy
        logger.info(
            f"K scaled {w_cal}×{h_cal} → {w_frame}×{h_frame} "
            f"(sx={sx:.4f}, sy={sy:.4f})"
        )
    else:
        # Orientation change — assume 90° CW (landscape → portrait).
        # Under 90° CW:  u_P = (H_L−1−v_L)·sx,  v_P = u_L·sy
        #   fx_P = fy_L · sx,   cx_P = (H_L − cy_L) · sx
        #   fy_P = fx_L · sy,   cy_P = cx_L · sy
        sx = w_frame / h_cal   # portrait width  ÷ landscape height
        sy = h_frame / w_cal   # portrait height ÷ landscape width
        fx_new = K[1, 1] * sx   # fy → fx
        fy_new = K[0, 0] * sy   # fx → fy
        cx_new = (h_cal - K[1, 2]) * sx
        cy_new = K[0, 2] * sy
        K = np.array(
            [[fx_new, 0.0, cx_new],
             [0.0, fy_new, cy_new],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        logger.info(
            f"K adapted {w_cal}×{h_cal}(landscape) → {w_frame}×{h_frame}(portrait) "
            f"via 90° CW: fx={fx_new:.2f}, fy={fy_new:.2f}, "
            f"cx={cx_new:.2f}, cy={cy_new:.2f}"
        )

    return K


def _find_charuco_frame(
    video_path:          str,
    start_frame:         int,
    charuco_det,
    search_window:       int = 30,
    required_ids:        Optional[set] = None,
    prefer_max_corners:  bool = False,
    rotate:              str = "none",
) -> Tuple[np.ndarray, Tuple[int, int], int, np.ndarray, np.ndarray]:
    """Find the best video frame for ChArUco-based calibration.

    Search strategy
    ---------------
    Pass 1 — local ±search_window frames around start_frame (fast).
    Pass 2 — full-video coarse scan every 15 frames (fallback).

    *prefer_max_corners* (True when outer_corner_mapping is active):
        Scan the entire video and return the frame with the MOST detected
        corners.  More spread-out corners give a far more stable depth
        estimate from solvePnP than the first frame that happens to show ≥4
        clustered corners.  The scan still runs coarsely (every 15 frames) to
        keep latency reasonable.

    *required_ids* (set of int, set when marker_mapping is active):
        Prefer frames where ALL required corner IDs are visible.  Falls back
        to the best partial match (most IDs present) if no perfect frame exists.

    Returns
    -------
    (frame_bgr, (w, h), actual_frame_idx, charuco_corners, charuco_ids)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Note: CAP_PROP_FRAME_WIDTH/HEIGHT are the *coded* (stream) dimensions.
    # The actual decoded frame may differ if cv2 applies the rotation metadata tag.
    # We derive w, h from the actual frame below after the first successful read.
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Trackers for the two selection modes
    best_partial: Optional[Tuple] = None  # (n_hits,   frame, idx, corners, ids)
    best_maxcorn: Optional[Tuple] = None  # (n_corners, frame, idx, corners, ids)

    def _try_frame(idx: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            return None
        frame = _rotate_frame(frame, rotate)
        nonlocal w, h
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for img in (gray, clahe.apply(gray)):
            corners, ids, _, _ = charuco_det.detectBoard(img)
            if corners is not None and len(corners) >= 4:
                return frame, idx, corners, ids
        return None

    def _covers_required(ids_array) -> int:
        if required_ids is None:
            return 0
        return len(required_ids & set(ids_array.ravel().tolist()))

    def _update_maxcorn(frame, found_idx, corners, ids):
        nonlocal best_maxcorn
        n = len(corners)
        if best_maxcorn is None or n > best_maxcorn[0]:
            best_maxcorn = (n, frame, found_idx, corners, ids)

    # ── Pass 1: local window around start_frame ───────────────────────────
    local_candidates: List[int] = [start_frame]
    for delta in range(1, search_window + 1):
        if start_frame + delta < total:
            local_candidates.append(start_frame + delta)
        if start_frame - delta >= 0:
            local_candidates.append(start_frame - delta)

    for idx in local_candidates:
        result = _try_frame(idx)
        if result is None:
            continue
        frame, found_idx, corners, ids = result

        if prefer_max_corners:
            _update_maxcorn(frame, found_idx, corners, ids)
            # Don't stop early — scan everything for max-corners mode

        elif required_ids is None:
            # Fast-path: first valid frame is good enough
            cap.release()
            logger.info(f"ChArUco board found at frame {found_idx} (local search)")
            return frame, (w, h), found_idx, corners, ids

        else:
            n_hits = _covers_required(ids)
            if n_hits == len(required_ids):
                cap.release()
                logger.info(
                    f"ChArUco board found at frame {found_idx} (local search) — "
                    f"all {len(required_ids)} required corner IDs present."
                )
                return frame, (w, h), found_idx, corners, ids
            if best_partial is None or n_hits > best_partial[0]:
                best_partial = (n_hits, frame, found_idx, corners, ids)

    # ── Pass 2: full-video coarse scan (every 15 frames) ─────────────────
    if not prefer_max_corners:
        logger.warning(
            f"Board not found near frame {start_frame} (±{search_window}). "
            "Falling back to full-video scan (every 15 frames)…"
        )
    else:
        logger.info(
            "Scanning full video for frame with most ChArUco corners "
            "(prefer_max_corners mode — coarse scan every 15 frames)…"
        )

    coarse_stride = 15
    for idx in range(0, total, coarse_stride):
        result = _try_frame(idx)
        if result is None:
            continue
        frame, found_idx, corners, ids = result

        if prefer_max_corners:
            _update_maxcorn(frame, found_idx, corners, ids)

        elif required_ids is None:
            cap.release()
            logger.info(
                f"ChArUco board found at frame {found_idx} during full-video scan. "
                f"Set 'Video frame #' to {found_idx} to use this frame directly next time."
            )
            return frame, (w, h), found_idx, corners, ids

        else:
            n_hits = _covers_required(ids)
            if n_hits == len(required_ids):
                cap.release()
                detected_ids_str = ", ".join(str(i) for i in sorted(ids.ravel().tolist()))
                logger.info(
                    f"ChArUco board found at frame {found_idx} (full scan) with ALL "
                    f"{len(required_ids)} required corner IDs present. "
                    f"Set 'Video frame #' to {found_idx} to use this frame directly next time. "
                    f"Detected IDs: [{detected_ids_str}]"
                )
                return frame, (w, h), found_idx, corners, ids
            if best_partial is None or n_hits > best_partial[0]:
                best_partial = (n_hits, frame, found_idx, corners, ids)

    cap.release()

    # ── Return best max-corners frame ─────────────────────────────────────
    if prefer_max_corners and best_maxcorn is not None:
        n_best, frame, found_idx, corners, ids = best_maxcorn
        ids_str = ", ".join(str(i) for i in sorted(ids.ravel().tolist()))
        logger.info(
            f"Best frame for calibration: frame {found_idx} with {n_best} ChArUco corners. "
            f"Set 'Video frame #' to {found_idx} to reuse it directly next time. "
            f"Corner IDs: [{ids_str}]"
        )
        return frame, (w, h), found_idx, corners, ids

    # ── Return best partial match (required_ids mode) ─────────────────────
    if best_partial is not None:
        n_hits, frame, found_idx, corners, ids = best_partial
        detected_ids_sorted = sorted(ids.ravel().tolist())
        missing_ids = sorted(required_ids - set(ids.ravel().tolist()))
        logger.warning(
            f"No video frame found with ALL required corner IDs {sorted(required_ids)}. "
            f"Best frame {found_idx} has {n_hits}/{len(required_ids)} required IDs. "
            f"Detected IDs: {detected_ids_sorted}. "
            f"Missing IDs: {missing_ids}. "
            "Options: (1) update the Vicon Marker Mapping panel to use the detected IDs, "
            "or (2) re-record the calibration video with the full board clearly visible."
        )
        return frame, (w, h), found_idx, corners, ids

    raise ValueError(
        f"ChArUco board not detected anywhere in the video "
        f"(searched ±{search_window} frames around frame {start_frame}, "
        f"then every {coarse_stride} frames across all {total} frames). "
        "Ensure the board is fully visible in the video and the board "
        "parameters (dictionary, cols, rows, square/marker size) match "
        "the physical board used."
    )


def _extract_frame(video_path: str, frame_idx: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not ret:
        raise ValueError(f"Cannot read frame {frame_idx} from video.")
    return frame, (w, h)


def _kabsch_rigid_transform(
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate rotation R and translation t (Kabsch / SVD method).

    Finds R, t such that:  pts_dst ≈ (R @ pts_src.T).T + t

    Args:
        pts_src: (N, 3) source points (board-local coords, mm).
        pts_dst: (N, 3) target points (Vicon world coords, mm).

    Returns:
        R: (3, 3) rotation matrix.
        t: (3,)  translation vector.
    """
    c_src = pts_src.mean(axis=0)
    c_dst = pts_dst.mean(axis=0)
    H = (pts_src - c_src).T @ (pts_dst - c_dst)
    U, _S, Vt = np.linalg.svd(H)
    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = c_dst - R @ c_src
    return R, t


# Keys expected in outer_corner_mapping and their board-local column-index (0-indexed)
_OUTER_CORNER_KEYS: List[str] = ["top_left", "top_right", "bottom_left", "bottom_right"]


def _outer_corner_calibration(
    board_cfg:        dict,
    markers:          Dict[str, np.ndarray],
    vicon_frame_idx:  int,
    outer_mapping:    dict,
    K:                np.ndarray,
    D:                np.ndarray,
    obj_pts_board_mm: np.ndarray,
    img_pts_all:      np.ndarray,
    solver_cfg:       dict,
) -> Tuple["ExtrinsicResult", np.ndarray, np.ndarray, List[str]]:
    """Vicon-world calibration via board-outer-corner Kabsch + direct world solvePnP.

    Step 1 — Kabsch rigid-body fit:
        Tries all 24 permutations of the four board canonical outer corners
        against the 4 Vicon marker positions.  The permutation with the lowest
        residual is used → board-to-Vicon transform (R_bv, t_bv).  This also
        auto-detects board orientation (portrait / landscape / 180° etc.).

    Step 2 — Direct world-space solvePnP:
        All detected inner corners are transformed to Vicon world via (R_bv, t_bv)
        and fed into a fresh solvePnP.  This gives the optimizer full freedom to
        find the correct camera distance (avoids inheriting the wrong depth from
        a board-only step that was computed on a small cluster of 4 corners).

    Args:
        board_cfg:         Board settings dict (cols, rows, square_length_mm).
        markers:           Vicon marker data {name: (N_frames, 3) mm}.
        vicon_frame_idx:   Vicon frame to read marker positions from.
        outer_mapping:     {corner_key: vicon_marker_name}
                           corner_key ∈ {"top_left", "top_right", "bottom_left", "bottom_right"}
        K, D:              Camera intrinsic matrix and distortion.
        obj_pts_board_mm:  (N, 3) detected inner corners in board-local mm.
        img_pts_all:       (N, 2) corresponding image pixel positions.
        solver_cfg:        solvePnP solver settings dict.

    Returns:
        (extr_world, obj_pts_world, img_pts_mapped, labels) where obj_pts_world
        are inner-corner positions in Vicon world space.
    """
    sq_mm = float(board_cfg.get("square_length_mm", 40.0))
    cols  = int(board_cfg.get("cols", 7))
    rows  = int(board_cfg.get("rows", 11))
    W, H  = cols * sq_mm, rows * sq_mm

    # Collect valid Vicon positions (order matches _OUTER_CORNER_KEYS)
    vicon_pts_used: List[np.ndarray] = []
    labels: List[str] = []
    available = list(markers.keys())

    for key in _OUTER_CORNER_KEYS:
        marker_name = outer_mapping.get(key)
        if not marker_name:
            continue
        if marker_name not in markers:
            logger.warning(
                f"outer_corner_mapping: '{marker_name}' (key='{key}') not in Vicon data. "
                f"Available: {available[:10]}{'…' if len(available) > 10 else ''}"
            )
            continue
        xyz = markers[marker_name][vicon_frame_idx]
        if np.any(np.isnan(xyz)):
            logger.warning(
                f"outer_corner_mapping: '{marker_name}' has NaN at Vicon frame {vicon_frame_idx}."
            )
            continue
        vicon_pts_used.append(xyz)
        labels.append(f"{marker_name} ({key})")

    n_valid = len(vicon_pts_used)
    if n_valid < 3:
        raise ValueError(
            f"outer_corner_mapping: only {n_valid} valid outer corners found "
            f"(need ≥ 3). Check marker names match the Vicon CSV and vicon_frame is valid. "
            f"Available markers: {available}"
        )

    vicon_arr = np.array(vicon_pts_used, dtype=np.float64)

    # All four canonical outer corners of the board in board-local mm (Z=0 plane)
    board_corners_canonical = np.array([
        [0, 0, 0],  # canonical index 0: top-left (portrait)
        [W, 0, 0],  # canonical index 1: top-right (portrait)
        [0, H, 0],  # canonical index 2: bottom-left (portrait)
        [W, H, 0],  # canonical index 3: bottom-right (portrait)
    ], dtype=np.float64)

    # Auto-detect board orientation by trying all permutations of corner assignment.
    # For n_valid=4 this is 24 trials; for n_valid=3 it is 24 as well (P(4,3)).
    # The permutation with the lowest Kabsch residual is selected.
    best_R:         Optional[np.ndarray] = None
    best_t:         Optional[np.ndarray] = None
    best_board_arr: Optional[np.ndarray] = None
    best_perm:      Optional[tuple]      = None
    best_err:       float                = np.inf

    for perm in _perms(range(4), n_valid):
        board_arr = board_corners_canonical[list(perm)]
        R_try, t_try = _kabsch_rigid_transform(board_arr, vicon_arr)
        errs     = np.linalg.norm((R_try @ board_arr.T).T + t_try - vicon_arr, axis=1)
        mean_err = float(errs.mean())
        if mean_err < best_err:
            best_err       = mean_err
            best_R, best_t = R_try, t_try
            best_board_arr = board_arr.copy()
            best_perm      = perm

    # Human-readable log of the winning assignment
    _cname = {
        0: "(0,0,0)",
        1: f"({W:.0f},0,0)",
        2: f"(0,{H:.0f},0)",
        3: f"({W:.0f},{H:.0f},0)",
    }
    perm_str = "  |  ".join(
        f"{labels[i]} → board{_cname[best_perm[i]]}" for i in range(n_valid)
    )
    n_perms = math.perm(4, n_valid)  # P(4, n_valid) = 4!/(4-n_valid)!
    logger.info(
        f"Kabsch orientation auto-selected "
        f"(best of {n_perms} permutations, mean residual = {best_err:.2f} mm):\n  {perm_str}"
    )

    # Warn if residual is still large after best-fit orientation — likely a
    # square_length_mm misconfiguration.  Estimate the actual square size from
    # the ratio of Vicon distances to board distances.
    if best_err > 20.0:
        vicon_span = float(np.mean([
            np.linalg.norm(vicon_arr[i] - vicon_arr[j])
            for i in range(n_valid) for j in range(i + 1, n_valid)
        ]))
        board_span = float(np.mean([
            np.linalg.norm(best_board_arr[i] - best_board_arr[j])
            for i in range(n_valid) for j in range(i + 1, n_valid)
        ]))
        apparent_sq = sq_mm * (vicon_span / board_span) if board_span > 0 else sq_mm
        logger.warning(
            f"Kabsch residual {best_err:.1f} mm after orientation auto-detect — "
            f"likely a square size mismatch. "
            f"Configured: {sq_mm:.1f} mm/sq. "
            f"Vicon distances imply ≈{apparent_sq:.1f} mm/sq. "
            f"Update 'square_length_mm' in Settings to the measured value."
        )

    R_bv, t_bv = best_R, best_t
    board_arr   = best_board_arr

    fit_errs = np.linalg.norm((R_bv @ board_arr.T).T + t_bv - vicon_arr, axis=1)
    logger.info(
        f"Kabsch fit ({n_valid} outer corners): "
        f"mean={fit_errs.mean():.2f} mm  max={fit_errs.max():.2f} mm"
    )

    # ── Transform inner corners to Vicon world ────────────────────────────
    # Use the Kabsch board→world transform to place every detected inner corner
    # into Vicon world coordinates (mm).
    obj_pts_world = (R_bv @ obj_pts_board_mm.T).T + t_bv   # (N, 3) mm

    # ── Run solvePnP directly in world space ──────────────────────────────
    # A fresh solvePnP on world-space points gives the optimizer full freedom
    # to find the correct camera distance.  This avoids inheriting the
    # potentially-wrong depth from the board-only step 1 (which is unreliable
    # when only a small cluster of corners is visible in that frame).
    logger.info(
        f"Running direct solvePnP in Vicon-world coords "
        f"({len(obj_pts_world)} inner corners)…"
    )
    extr_world = _solve_pnp(obj_pts_world, img_pts_all.reshape(-1, 2), K, D, solver_cfg)
    extr_world.method = "KABSCH"   # keep the KABSCH label for the UI

    logger.info(
        f"Outer-corner calibration RMS (direct world solvePnP, "
        f"{len(obj_pts_world)} inner corners): {extr_world.rms_px:.4f} px  "
        f"[{extr_world.quality_label()}]"
    )

    # ── Correct planar-target PnP reflection ambiguity ────────────────────
    # A flat board has two geometrically-valid solvePnP solutions: camera in
    # front of the board or reflected behind it.  Both give identical reprojection
    # error, so the default solver picks arbitrarily.
    #
    # Fix: run IPPE in board-LOCAL frame (Z=0 — required by the algorithm) to get
    # both solutions explicitly, pick the one whose camera position is on the
    # correct (front) side of the board, then convert that pose to Vicon world
    # frame using the Kabsch transform (R_bv, t_bv).
    #
    # Board-local → world:   X_world = R_bv @ X_board + t_bv
    # Camera pose conversion: R_cw = R_cb @ R_bv.T
    #                         t_cw = t_cb - R_cw @ t_bv
    # Camera position in world: pos_world = R_bv @ (-R_cb.T @ t_cb) + t_bv
    board_normal_world = R_bv[:, 2]           # board Z-axis in Vicon world frame
    board_center_world = (R_bv @ board_arr.mean(axis=0)) + t_bv
    cam_pos            = extr_world.camera_position_world
    on_front_side      = float(np.dot(cam_pos - board_center_world, board_normal_world)) > 0

    logger.info(
        f"Reflection check: cam={np.array(cam_pos).round(0).tolist()} mm  "
        f"board_center={board_center_world.round(0).tolist()} mm  "
        f"board_normal={board_normal_world.round(3).tolist()}  "
        f"dot={float(np.dot(cam_pos - board_center_world, board_normal_world)):.1f}  "
        f"front_side={on_front_side}"
    )

    if not on_front_side:
        logger.warning(
            "Reflection ambiguity detected: camera resolved to the WRONG side of "
            f"the board (dist={np.linalg.norm(cam_pos) / 1000:.3f} m from origin). "
            "Running IPPE in board-local frame to recover the front-side solution…"
        )
        try:
            # IPPE MUST receive board-local (Z=0) points — not world-space points.
            # solvePnPGeneric with IPPE always returns exactly 2 solutions.
            _, rvecs_ippe, tvecs_ippe, _ = cv2.solvePnPGeneric(
                obj_pts_board_mm.reshape(-1, 1, 3).astype(np.float64),
                img_pts_all.reshape(-1, 1, 2).astype(np.float64),
                K, D,
                flags=cv2.SOLVEPNP_IPPE,
            )
            extr_corrected = None
            for rv_i, tv_i in zip(rvecs_ippe, tvecs_ippe):
                # rv_i, tv_i describe camera pose in board-local frame.
                # Camera position in board frame: -R_cb.T @ t_cb
                R_cb  = cv2.Rodrigues(rv_i)[0]
                t_cb  = tv_i.ravel()
                pos_board = -R_cb.T @ t_cb

                # Convert camera position to Vicon world frame
                pos_world_i = R_bv @ pos_board + t_bv

                side_dot = float(np.dot(pos_world_i - board_center_world, board_normal_world))
                logger.info(
                    f"IPPE candidate: pos_board={pos_board.round(1).tolist()} mm  "
                    f"pos_world={pos_world_i.round(1).tolist()} mm  "
                    f"side_dot={side_dot:.1f}"
                )

                if side_dot > 0:   # camera is on the correct (front) side
                    # Convert the board-local pose to world-frame rvec/tvec
                    R_cw    = R_cb @ R_bv.T
                    t_cw    = t_cb - R_cw @ t_bv
                    rv_world, _ = cv2.Rodrigues(R_cw)
                    tv_world    = t_cw.reshape(3, 1)

                    # Refine in world frame with LM
                    try:
                        rv_world, tv_world = cv2.solvePnPRefineLM(
                            obj_pts_world.reshape(-1, 1, 3).astype(np.float64),
                            img_pts_all.reshape(-1, 1, 2).astype(np.float64),
                            K, D, rv_world, tv_world,
                        )
                        R_cw, _ = cv2.Rodrigues(rv_world)
                        pos_world_i = (-R_cw.T @ tv_world).ravel()
                    except cv2.error:
                        pass

                    rms_i = _rms(
                        obj_pts_world, img_pts_all.reshape(-1, 2),
                        rv_world, tv_world, K, D,
                    )
                    extr_corrected = _build_extrinsic(
                        rv_world, tv_world, rms_i, extr_world.n_inliers,
                        len(obj_pts_world), "KABSCH",
                    )
                    dist_m = np.linalg.norm(pos_world_i) / 1000.0
                    logger.info(
                        f"Ambiguity resolved via IPPE: "
                        f"camera at {pos_world_i.round(1).tolist()} mm  "
                        f"dist={dist_m:.3f} m from Vicon origin  "
                        f"RMS={rms_i:.4f} px [{extr_corrected.quality_label()}]"
                    )
                    break

            if extr_corrected is not None:
                extr_world = extr_corrected
            else:
                logger.warning(
                    "IPPE found no front-side solution — both candidates are behind "
                    "the board.  This may mean R_bv[:,2] (board normal) is pointing "
                    "away from the camera.  Flipping board_normal and retrying…"
                )
                # Last resort: flip the normal and accept whichever candidate has
                # the camera on the flipped-normal front side.
                for rv_i, tv_i in zip(rvecs_ippe, tvecs_ippe):
                    R_cb      = cv2.Rodrigues(rv_i)[0]
                    t_cb      = tv_i.ravel()
                    pos_board = -R_cb.T @ t_cb
                    pos_world_i = R_bv @ pos_board + t_bv
                    side_dot  = float(np.dot(pos_world_i - board_center_world, -board_normal_world))
                    if side_dot > 0:
                        R_cw = R_cb @ R_bv.T
                        t_cw = t_cb - R_cw @ t_bv
                        rv_world, _ = cv2.Rodrigues(R_cw)
                        tv_world    = t_cw.reshape(3, 1)
                        try:
                            rv_world, tv_world = cv2.solvePnPRefineLM(
                                obj_pts_world.reshape(-1, 1, 3).astype(np.float64),
                                img_pts_all.reshape(-1, 1, 2).astype(np.float64),
                                K, D, rv_world, tv_world,
                            )
                        except cv2.error:
                            pass
                        rms_i = _rms(
                            obj_pts_world, img_pts_all.reshape(-1, 2),
                            rv_world, tv_world, K, D,
                        )
                        extr_world = _build_extrinsic(
                            rv_world, tv_world, rms_i, extr_world.n_inliers,
                            len(obj_pts_world), "KABSCH",
                        )
                        pos_world_i = extr_world.camera_position_world
                        logger.info(
                            f"Flipped-normal fallback: camera at "
                            f"{np.array(pos_world_i).round(1).tolist()} mm  "
                            f"dist={np.linalg.norm(pos_world_i)/1000:.3f} m  "
                            f"RMS={rms_i:.4f} px"
                        )
                        break

        except (cv2.error, AttributeError) as exc:
            logger.warning(
                f"IPPE resolution failed ({exc}). Keeping original (possibly reflected) result."
            )
    else:
        dist_m = np.linalg.norm(cam_pos) / 1000.0
        logger.info(
            f"No reflection: camera is correctly on the front side of the board "
            f"(dist={dist_m:.3f} m from Vicon origin)."
        )

    inner_labels = [f"inner_{i}" for i in range(len(obj_pts_world))]
    return extr_world, obj_pts_world, img_pts_all.reshape(-1, 2), inner_labels


def _rigid_body_calibration(
    rigid_body_cfg: dict,
    K:               np.ndarray,
    D:               np.ndarray,
    obj_pts_board_mm: np.ndarray,
    img_pts:         np.ndarray,
    solver_cfg:      dict,
) -> Tuple["ExtrinsicResult", np.ndarray, np.ndarray, List[str]]:
    """Calibrate camera pose using Vicon-tracked rigid body pose of the ChArUco board.

    This is the recommended calibration mode when the Vicon system is available:

    Hardware setup:
        • Attach ≥3 retroreflective Vicon markers to the board BORDER or BACK
          (NOT on the checkerboard face — they would occlude ChArUco corners).
        • Define the board as a Rigid Body in Vicon Nexus / Tracker.
        • Measure the offset from the Nexus Rigid Body origin to the board's
          top-left inner corner (board coordinate [0, 0, 0]) in board-local mm.

    Getting the pose from Vicon Nexus (static export):
        1. In Nexus 2.x: Open the trial, navigate to the calibration frame.
        2. Subject Properties → select the board Rigid Body segment.
        3. In the "Data" panel or via Pipeline: export Global Translation (mm)
           and Global Rotation Matrix for that segment at that frame.
        4. Paste the values into the config as ``translation_mm`` and
           ``rotation_matrix`` (see below).

    Getting the pose via live DataStream (Python):
        >>> client.GetSegmentGlobalTranslation('Board', 'Board')  # → (X, Y, Z) mm
        >>> client.GetSegmentGlobalRotationMatrix('Board', 'Board')  # → flat 9-float list

    Config keys (``rigid_body_pose`` sub-dict):
        translation_mm   : [X, Y, Z]          # board origin in Vicon world (mm)
        rotation_matrix  : [[r00,…],[…],[…]]  # 3×3 board orientation in Vicon world
        rvec             : [rx, ry, rz]        # alternative to rotation_matrix (Rodrigues)
        local_offset_mm  : [dx, dy, dz]        # optional offset: Nexus RB origin → board [0,0,0]

    Pipeline:
        1. Parse T_B→W from config.
        2. Apply local_offset to map Nexus RB origin → board corner (0,0,0).
        3. Transform all detected inner corners to Vicon world:
               pts_world = R_vb @ pts_board + t_board_origin_world
        4. solvePnP(RANSAC) on world-space 3D pts vs 2D image pts → T_W→C.
        5. Reflection disambiguation via IPPE if camera resolves to wrong side.

    Args:
        rigid_body_cfg:    See config keys above.
        K:                 Camera intrinsic matrix (adapted to current frame size).
        D:                 Distortion coefficients.
        obj_pts_board_mm:  (N, 3) detected inner corners in board-local mm (Z=0 plane).
        img_pts:           (N, 2) corresponding image pixel positions.
        solver_cfg:        solvePnP solver settings dict.

    Returns:
        (extr, obj_pts_world_mm, img_pts, labels)
    """
    # ── Parse board-to-world transform (T_B→W) ───────────────────────────────
    if "translation_mm" not in rigid_body_cfg:
        raise ValueError(
            "rigid_body_pose: 'translation_mm' is required. "
            "Export the board Rigid Body global translation from Vicon Nexus "
            "(e.g. Subject Properties → Global Translation at the calibration frame)."
        )
    t_vb = np.array(rigid_body_cfg["translation_mm"], dtype=np.float64).ravel()   # (3,) mm

    if "rotation_matrix" in rigid_body_cfg:
        R_vb = np.array(rigid_body_cfg["rotation_matrix"], dtype=np.float64).reshape(3, 3)
    elif "rvec" in rigid_body_cfg:
        R_vb, _ = cv2.Rodrigues(np.array(rigid_body_cfg["rvec"], dtype=np.float64))
    else:
        raise ValueError(
            "rigid_body_pose: 'rotation_matrix' (3×3 list) or 'rvec' (3-element Rodrigues) required."
        )

    # ── Apply local offset: Nexus RB origin → ChArUco board corner (0,0,0) ──
    # If the Nexus Rigid Body origin does not coincide with the board's top-left
    # inner corner, provide the offset in board-local mm.
    # Board corner (0,0,0) in world = t_vb + R_vb @ local_offset.
    local_offset = np.zeros(3, dtype=np.float64)
    if "local_offset_mm" in rigid_body_cfg:
        local_offset = np.array(rigid_body_cfg["local_offset_mm"], dtype=np.float64).ravel()

    t_board_origin = t_vb + R_vb @ local_offset   # board [0,0,0] in Vicon world mm

    logger.info(
        f"Rigid body pose: t_vb={t_vb.round(1).tolist()} mm, "
        f"local_offset={local_offset.round(1).tolist()} mm, "
        f"board_origin_world={t_board_origin.round(1).tolist()} mm, "
        f"board_normal_world={R_vb[:, 2].round(3).tolist()}"
    )

    # ── Transform inner corners to Vicon world ────────────────────────────────
    # X_world = R_vb @ X_board + t_board_origin
    obj_pts_world = (R_vb @ obj_pts_board_mm.T).T + t_board_origin   # (N, 3) mm

    board_normal_world = R_vb[:, 2]   # board Z-axis (front face normal) in world
    board_center_world = t_board_origin

    logger.info(
        f"Rigid body: {len(obj_pts_world)} inner corners transformed to Vicon world; "
        f"running solvePnP…"
    )

    # ── solvePnP in Vicon world space ─────────────────────────────────────────
    extr = _solve_pnp(obj_pts_world, img_pts.reshape(-1, 2), K, D, solver_cfg)
    extr.method = "RIGID_BODY"

    cam_pos  = extr.camera_position_world
    side_dot = float(np.dot(cam_pos - board_center_world, board_normal_world))
    on_front = side_dot > 0

    logger.info(
        f"RIGID_BODY: RMS={extr.rms_px:.4f} px [{extr.quality_label()}], "
        f"cam={np.array(cam_pos).round(1).tolist()} mm, "
        f"dist={np.linalg.norm(cam_pos) / 1000:.3f} m, "
        f"side_dot={side_dot:.1f}, front={on_front}"
    )

    # ── Reflection disambiguation ─────────────────────────────────────────────
    # A flat board (Z=0) has two solvePnP solutions with identical reprojection error.
    # Use IPPE in board-local frame to get both solutions, pick the front-side one,
    # then convert that board-local pose to Vicon world frame via T_W→C = T_B→C × T_W→B.
    if not on_front:
        logger.warning(
            f"Reflection: camera at {np.array(cam_pos).round(0).tolist()} mm is on "
            f"the WRONG side of the board (side_dot={side_dot:.1f}). "
            "Recovering front-side solution via IPPE…"
        )
        try:
            _, rvecs_ippe, tvecs_ippe, _ = cv2.solvePnPGeneric(
                obj_pts_board_mm.reshape(-1, 1, 3).astype(np.float64),
                img_pts.reshape(-1, 1, 2).astype(np.float64),
                K, D, flags=cv2.SOLVEPNP_IPPE,
            )
            extr_corrected = None
            for rv_i, tv_i in zip(rvecs_ippe, tvecs_ippe):
                R_cb      = cv2.Rodrigues(rv_i)[0]
                t_cb      = tv_i.ravel()              # board-local mm (board pts in mm)
                pos_board = -R_cb.T @ t_cb            # cam position in board frame (mm)
                pos_world_i = R_vb @ pos_board + t_board_origin

                dot_i = float(np.dot(pos_world_i - board_center_world, board_normal_world))
                logger.info(
                    f"IPPE candidate: pos_world={pos_world_i.round(1).tolist()} mm  "
                    f"dot={dot_i:.1f}"
                )
                if dot_i > 0:
                    # Convert board-local pose to Vicon world pose:
                    #   T_W→C = T_B→C × T_W→B
                    #   R_wc  = R_cb @ R_vb.T
                    #   t_wc  = t_cb − R_wc @ t_board_origin
                    R_wc = R_cb @ R_vb.T
                    t_wc = (t_cb - R_wc @ t_board_origin).reshape(3, 1)
                    rv_wc, _ = cv2.Rodrigues(R_wc)
                    # Refine with all inner corners in world frame
                    try:
                        rv_wc, t_wc = cv2.solvePnPRefineLM(
                            obj_pts_world.reshape(-1, 1, 3).astype(np.float64),
                            img_pts.reshape(-1, 1, 2).astype(np.float64),
                            K, D, rv_wc, t_wc,
                        )
                    except cv2.error:
                        pass
                    rms_i = _rms(obj_pts_world, img_pts.reshape(-1, 2), rv_wc, t_wc, K, D)
                    extr_corrected = _build_extrinsic(
                        rv_wc, t_wc, rms_i, extr.n_inliers,
                        len(obj_pts_world), "RIGID_BODY",
                    )
                    pos_final = extr_corrected.camera_position_world
                    logger.info(
                        f"IPPE front-side resolved: "
                        f"cam={np.array(pos_final).round(1).tolist()} mm  "
                        f"dist={np.linalg.norm(pos_final) / 1000:.3f} m  "
                        f"RMS={rms_i:.4f} px [{extr_corrected.quality_label()}]"
                    )
                    break
            if extr_corrected is not None:
                extr = extr_corrected
            else:
                logger.warning(
                    "IPPE: no front-side candidate found — keeping original solvePnP result. "
                    "If the camera is behind the board, flip the board or increase "
                    "local_offset_mm to ensure the board normal points toward the camera."
                )
        except (cv2.error, AttributeError) as exc:
            logger.warning(f"IPPE disambiguation failed ({exc}); keeping original result.")
    else:
        logger.info(
            f"No reflection ambiguity — camera is on the correct front side of the board "
            f"(dist={np.linalg.norm(cam_pos) / 1000:.3f} m from Vicon origin)."
        )

    labels = [f"inner_{i}" for i in range(len(obj_pts_world))]
    return extr, obj_pts_world, img_pts.reshape(-1, 2), labels


def _build_vicon_correspondences(
    marker_mapping: dict,
    markers: Dict[str, np.ndarray],
    vicon_frame_idx: int,
    charuco_corners: np.ndarray,
    charuco_ids: np.ndarray,
    board,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build 3D Vicon ↔ 2D image correspondences using a marker mapping dict.

    marker_mapping: {charuco_corner_id_str: vicon_marker_name}

    Each detected ChArUco inner-corner is associated with a Vicon marker
    placed at (or very near) that corner.  The Vicon 3D position is used
    directly as the world-space 3D point for solvePnP.

    Note on offset tolerance: if Vicon markers are placed at the outer board
    edge (physical corner) rather than exactly at an inner corner, their 3D
    positions will be offset by ~20-40 mm from the inner-corner position.
    solvePnP absorbs small consistent offsets; any resulting RMS above ~2 px
    suggests the mapping IDs need adjustment.
    """
    obj_pts, img_pts, labels = [], [], []
    ids_flat = charuco_ids.ravel()
    n_total  = len(ids_flat)
    n_matched = 0

    available_markers = list(markers.keys())

    for i, corner_id in enumerate(ids_flat):
        key = str(corner_id)
        vicon_name = marker_mapping.get(key)
        if vicon_name is None:
            continue  # this corner ID not in the mapping — skip
        n_matched += 1
        if vicon_name not in markers:
            logger.warning(
                f"Marker '{vicon_name}' (mapped to corner {key}) not found in Vicon data. "
                f"Available markers: {available_markers[:10]}{'…' if len(available_markers) > 10 else ''}"
            )
            continue
        xyz = markers[vicon_name][vicon_frame_idx]
        if np.any(np.isnan(xyz)):
            logger.warning(
                f"Marker '{vicon_name}' has NaN at Vicon frame {vicon_frame_idx}. "
                "Try a different Vicon frame # where this marker has valid data."
            )
            continue
        obj_pts.append(xyz)
        img_pts.append(charuco_corners[i].ravel())
        labels.append(f"{vicon_name} (id={key})")

    logger.info(
        f"Vicon correspondences: {len(obj_pts)} valid pairs "
        f"(mapping matched {n_matched}/{n_total} detected corners)"
    )

    if len(obj_pts) < 4:
        raise ValueError(
            f"Only {len(obj_pts)} valid Vicon correspondences found "
            f"(need ≥ 4). "
            f"Detected corner IDs in this frame: {sorted(ids_flat.tolist())}. "
            f"Mapped IDs: {sorted(int(k) for k in marker_mapping.keys())}. "
            "Check that the marker_mapping IDs match the detected corner IDs, "
            "and that vicon_frame points to a frame with valid marker data."
        )
    return np.array(obj_pts), np.array(img_pts), labels


def _solve_pnp(obj_pts, img_pts, K, D, solver_cfg) -> ExtrinsicResult:
    use_ransac = bool(solver_cfg.get("use_ransac", True))
    refine     = bool(solver_cfg.get("refine_lm", True))
    methods    = solver_cfg.get("methods", ["ITERATIVE", "EPNP", "SQPNP"])
    ransac_th  = float(solver_cfg.get("ransac_reprojection_threshold_px", 8.0))
    ransac_cf  = float(solver_cfg.get("ransac_confidence", 0.999))

    best: Optional[ExtrinsicResult] = None

    if use_ransac and len(obj_pts) >= 6:
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts.reshape(-1, 1, 3), img_pts.reshape(-1, 1, 2), K, D,
                iterationsCount=200,
                reprojectionError=ransac_th,
                confidence=ransac_cf,
            )
            if ok and rvec is not None:
                if refine and inliers is not None and len(inliers) >= 4:
                    mask = inliers.ravel()
                    rvec, tvec = cv2.solvePnPRefineLM(
                        obj_pts[mask].reshape(-1, 1, 3),
                        img_pts[mask].reshape(-1, 1, 2),
                        K, D, rvec, tvec,
                    )
                rms = _rms(obj_pts, img_pts, rvec, tvec, K, D)
                ni  = len(inliers) if inliers is not None else len(obj_pts)
                best = _build_extrinsic(rvec, tvec, rms, ni, len(obj_pts), "RANSAC")
        except cv2.error as e:
            logger.warning(f"RANSAC failed: {e}")

    for method_name in methods:
        flag = _PNP_METHODS.get(method_name, cv2.SOLVEPNP_ITERATIVE)
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts.reshape(-1, 1, 3), img_pts.reshape(-1, 1, 2), K, D, flags=flag,
            )
        except cv2.error as e:
            logger.warning(f"{method_name} failed: {e}")
            continue
        if not ok:
            continue
        if refine:
            try:
                rvec, tvec = cv2.solvePnPRefineLM(
                    obj_pts.reshape(-1, 1, 3), img_pts.reshape(-1, 1, 2), K, D, rvec, tvec,
                )
            except cv2.error:
                pass
        rms = _rms(obj_pts, img_pts, rvec, tvec, K, D)
        candidate = _build_extrinsic(rvec, tvec, rms, len(obj_pts), len(obj_pts), method_name)
        if best is None or candidate.rms_px < best.rms_px:
            best = candidate

    if best is None:
        raise RuntimeError(
            "All solvePnP attempts failed. "
            "Check that your 3D/2D correspondences are correct."
        )
    return best


def _rms(obj_pts, img_pts, rvec, tvec, K, D) -> float:
    proj, _ = cv2.projectPoints(obj_pts.reshape(-1, 1, 3), rvec, tvec, K, D)
    diff    = proj.reshape(-1, 2) - img_pts.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


def _draw_overlay(frame, obj_pts, img_pts, proj_pts, extr, K, D) -> np.ndarray:
    for i, (obs, prj) in enumerate(zip(img_pts, proj_pts)):
        cv2.circle(frame, tuple(obs.astype(int)), 8,  (0, 255, 0), 2)
        cv2.circle(frame, tuple(prj.astype(int)), 6,  (0, 0, 255), -1)
        cv2.line(frame, tuple(obs.astype(int)), tuple(prj.astype(int)), (255, 165, 0), 1)
    axis_len = float(np.mean(np.abs(obj_pts))) * 0.5 if len(obj_pts) > 0 else 100.0
    cv2.drawFrameAxes(frame, K, D, extr.rvec, extr.tvec, axis_len)
    cv2.putText(frame, f"RMS: {extr.rms_px:.3f}px  [{extr.quality_label()}]",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def _save_reprojection_plot(errors, labels, warn, bad, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(8, len(errors) * 0.4 + 2), 4), facecolor="#0F1115")
    ax.set_facecolor("#1A1D24")
    ax.tick_params(colors="#94A3B8", axis="both")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1E293B")

    colors = ["#22C55E" if e < warn else "#F59E0B" if e < bad else "#EF4444"
              for e in errors]
    ax.bar(range(len(errors)), errors, color=colors, alpha=0.85)
    ax.axhline(warn, color="#F59E0B", lw=1.2, ls="--", alpha=0.7)
    ax.axhline(bad,  color="#EF4444", lw=1.2, ls="--", alpha=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7, color="#94A3B8")
    ax.set_ylabel("Reprojection error (px)", color="#94A3B8", fontsize=10)
    ax.set_title("Per-Marker Reprojection Error", color="#E2E8F0", fontsize=12, fontweight="bold")
    fig.tight_layout(pad=1.5)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_error_heatmap(errors, img_pts, img_size, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0F1115")
    ax.set_facecolor("#1A1D24")
    ax.tick_params(colors="#94A3B8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1E293B")

    sc = ax.scatter(img_pts[:, 0], img_pts[:, 1], c=errors, cmap="hot",
                    s=60, alpha=0.9, vmin=0)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.ax.tick_params(colors="#94A3B8")
    cbar.set_label("Error (px)", color="#94A3B8")
    ax.set_xlim(0, img_size[0])
    ax.set_ylim(img_size[1], 0)
    ax.set_xlabel("X (px)", color="#94A3B8", fontsize=10)
    ax.set_ylabel("Y (px)", color="#94A3B8", fontsize=10)
    ax.set_title("Spatial Reprojection Error Heatmap", color="#E2E8F0", fontsize=12)
    ax.set_aspect("equal")
    fig.tight_layout(pad=1.5)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_extrinsic_report(extr, errors, labels, config,
                            video_path, vicon_path,
                            vid_f, vic_f, out_dir: Path):
    pos = extr.camera_position_world
    rx, ry, rz = extr.euler_angles_deg
    R, t = extr.R, extr.tvec.ravel()
    lines = [
        "=" * 65,
        "  EXTRINSIC ALIGNMENT REPORT — Vicon ↔ Smartphone Camera",
        "  ViconMoCap Sync Studio  |  Extrinsic calibration report",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 65, "",
        f"  Video         : {Path(video_path).name}  (frame {vid_f})",
        f"  Vicon file    : {Path(vicon_path).name}  (frame {vic_f})",
        f"  PnP method    : {extr.method}",
        f"  RMS error     : {extr.rms_px:.4f} px  [{extr.quality_label()}]",
        f"  Inliers       : {extr.n_inliers} / {extr.n_points}",
        "",
        "  Per-marker errors (px):",
    ]
    for lbl, e in zip(labels, errors):
        tag = "✓" if e < 3.0 else "⚠" if e < 8.0 else "✗"
        lines.append(f"    {tag}  {lbl:<22s}  {e:.4f} px")
    lines += [
        "",
        "  ROTATION MATRIX R",
        f"  [ {R[0,0]:+9.6f}  {R[0,1]:+9.6f}  {R[0,2]:+9.6f} ]",
        f"  [ {R[1,0]:+9.6f}  {R[1,1]:+9.6f}  {R[1,2]:+9.6f} ]",
        f"  [ {R[2,0]:+9.6f}  {R[2,1]:+9.6f}  {R[2,2]:+9.6f} ]",
        f"  Euler:  Rx={rx:.2f}°  Ry={ry:.2f}°  Rz={rz:.2f}°",
        "",
        "  TRANSLATION  (world → camera, mm)",
        f"  tx={t[0]:+12.3f}  ty={t[1]:+12.3f}  tz={t[2]:+12.3f}",
        "",
        "  CAMERA POSITION IN WORLD (mm)",
        f"  X={pos[0]:+12.3f}  Y={pos[1]:+12.3f}  Z={pos[2]:+12.3f}",
        f"  Distance from origin: {np.linalg.norm(pos)/1000:.3f} m",
        "=" * 65,
    ]
    with open(out_dir / "alignment_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
