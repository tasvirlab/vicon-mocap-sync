"""
intrinsic_calibration.py
========================
Intrinsic calibration pipeline.

Frame selection (same as calibrate_intrinsic.py):
  1. Sample every N frames (extraction.sample_interval_frames).
  2. Reject blurry frames (Laplacian variance < blur_threshold).
  3. Detect ChArUco corners; require >= min_corners detected corners.
  4. Bin frames by board centroid into a spatial grid (4 × 3 cells).
  5. Keep the sharpest per_cell frames per cell; stop at target count.

Output:
  intrinsic.json, distortion.json, intrinsic.yaml, calibration_report.txt,
  selected_frames/frame_XXXXXX.jpg (ChArUco overlay),
  reprojection_check/check_XXXXXX.jpg (green=detected, red=reprojected),
  reprojection_error_plot.png (bar chart), intrinsic_params.npz (for pipeline).
"""

from __future__ import annotations
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from .vicon_io import rotate_frame
from .calibration_output import (
    save_intrinsics, write_report,
    save_selected_frames, save_reprojection_check,
    save_reprojection_bar_chart,
)

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


# ── Board builder ─────────────────────────────────────────────────────────────

def _build_charuco(board_cfg: dict):
    """
    Build CharucoBoard and detector from config dict.

    Supports OpenCV >= 4.7 (CharucoDetector) and legacy versions.

    Returns:
        (board, detector_or_None, dictionary)
    """
    dict_id    = _DICT_MAP.get(board_cfg.get("dictionary", "DICT_6X6_250"),
                               cv2.aruco.DICT_6X6_250)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    cols = int(board_cfg.get("cols", 7))
    rows = int(board_cfg.get("rows", 10))
    sq   = float(board_cfg.get("square_length_mm", 40.0)) / 1000.0
    mk   = float(board_cfg.get("marker_length_mm", 26.0)) / 1000.0

    try:
        board = cv2.aruco.CharucoBoard((cols, rows), sq, mk, dictionary)
    except TypeError:
        board = cv2.aruco.CharucoBoard_create(cols, rows, sq, mk, dictionary)

    detector = None
    try:
        detector = cv2.aruco.CharucoDetector(board)
    except AttributeError:
        pass  # OpenCV < 4.7 — detect_charuco() uses legacy path

    return board, detector, dictionary


def _detect_charuco(gray: np.ndarray, board, detector, dictionary,
                    min_corners: int = 6):
    """
    Detect ChArUco corners — supports both modern and legacy OpenCV APIs.

    Returns:
        (charuco_corners, charuco_ids) or (None, None) if < min_corners detected.
    """
    if detector is not None:
        cc, cids, _, _ = detector.detectBoard(gray)
    else:
        params = (cv2.aruco.DetectorParameters_create()
                  if hasattr(cv2.aruco, "DetectorParameters_create")
                  else cv2.aruco.DetectorParameters())
        mc, mids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
        if mids is None or len(mids) < 4:
            return None, None
        _, cc, cids = cv2.aruco.interpolateCornersCharuco(mc, mids, gray, board)

    if cc is None or cids is None or len(cids) < min_corners:
        return None, None

    # detectBoard() can return corners shaped (N, 2) instead of the (N, 1, 2)
    # OpenCV's C++ side treats as a Point2f array. A plain (N, 2) float32 array
    # converts to a single-channel N×2 Mat whose .total() is 2N, not N, which
    # disagrees with charucoIds.total()=N even when the arrays are otherwise
    # fine — OpenCV 5.0.0 then raises inside drawDetectedCornersCharuco.
    # Normalising shape/dtype here (the single source both arrays come from)
    # fixes every downstream consumer at once.
    cc   = np.asarray(cc,   dtype=np.float32).reshape(-1, 1, 2)
    cids = np.asarray(cids, dtype=np.int32).reshape(-1, 1)
    return cc, cids


# ── Frame selection ───────────────────────────────────────────────────────────

def _select_frames(video_path: str, board, detector, dictionary,
                   cfg: dict, rotate: str,
                   progress_cb: Optional[ProgressCallback] = None) -> list:
    """
    Scan video and return a diverse, sharp subset for calibration.

    Algorithm:
      - Sort ALL candidates by sharpness DESC
      - Greedily pick per_cell frames per grid cell

    Returns:
        Sorted list of (frame_idx, charuco_corners, charuco_ids).
    """
    ex          = cfg.get("extraction", {})
    step        = int(ex.get("sample_interval_frames", 5))
    blur_thresh = float(ex.get("blur_threshold", 80.0))
    target      = int(ex.get("target_accepted_frames", 50))
    min_corners = int(cfg.get("board", {}).get("min_corners", 8))
    grid_c      = int(ex.get("diversity_grid_cols", 4))
    grid_r      = int(ex.get("diversity_grid_rows", 3))
    per_cell    = max(3, (target // (grid_c * grid_r)) + 2)

    cap   = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, probe = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Cannot read first frame of video.")
    H, W = rotate_frame(probe, rotate).shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    candidates = []
    fidx = 0
    logger.info(f"Scanning {total} frames (every {step}th)…")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx % step == 0:
            frame = rotate_frame(frame, rotate)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
            if sharp >= blur_thresh:
                cc, cids = _detect_charuco(gray, board, detector, dictionary,
                                           min_corners)
                if cc is not None:
                    pts = cc.reshape(-1, 2)
                    gx  = min(int(pts[:, 0].mean() / W * grid_c), grid_c - 1)
                    gy  = min(int(pts[:, 1].mean() / H * grid_r), grid_r - 1)
                    candidates.append((sharp, fidx, (gx, gy), cc, cids))
            if progress_cb and total > 0:
                pct = int(5 + 50 * fidx / total)
                progress_cb(min(pct, 55), f"Scanning… {len(candidates)} candidates")
        fidx += 1

    cap.release()
    logger.info(f"Candidates found: {len(candidates)}")

    if not candidates:
        raise RuntimeError(
            "No valid ChArUco detections. Try lowering blur_threshold, "
            "setting rotate=ccw90 for iPhone portrait video, or --step 1.")

    # Sort by sharpness DESC, then pick per_cell per grid cell
    candidates.sort(key=lambda t: -t[0])
    cell_count: dict = {}
    selected = []
    for sharp, fi, cell, cc, cids in candidates:
        if cell_count.get(cell, 0) < per_cell:
            cell_count[cell] = cell_count.get(cell, 0) + 1
            selected.append((fi, cc, cids))
        if len(selected) >= target:
            break

    selected.sort(key=lambda t: t[0])
    logger.info(
        f"Selected {len(selected)} frames across {len(cell_count)} grid cells.")
    return selected


# ── Calibration ───────────────────────────────────────────────────────────────

def _calibrate(selected: list, board, img_size: tuple,
               cal_cfg: dict) -> Tuple[float, np.ndarray, np.ndarray, list, list, int]:
    """
    Run cv2.calibrateCamera on selected ChArUco frames.

    Returns:
        (rms, K, dist, rvecs, tvecs, n_used)
    """
    flags      = int(cal_cfg.get("flags", 0))
    min_frames = int(cal_cfg.get("min_frames", 15))
    all_obj, all_img = [], []

    for _, cc, cids in selected:
        obj, img = board.matchImagePoints(cc, cids)
        if obj is not None and img is not None and len(obj) >= 4:
            all_obj.append(obj.astype(np.float32))
            all_img.append(img.astype(np.float32))

    if len(all_obj) < min_frames:
        raise RuntimeError(
            f"Only {len(all_obj)} usable frames (need >= {min_frames}). "
            "Lower blur_threshold or use a longer calibration video.")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_obj, all_img, img_size, None, None, flags=flags)
    return rms, K, dist, rvecs, tvecs, len(all_obj)


def _per_frame_rms(obj_pts, img_pts, rvecs, tvecs, K, dist) -> List[float]:
    """Compute per-frame RMS reprojection error.

    Both arrays passed to cv2.norm must be the SAME OpenCV type. `proj` is
    reshaped to (-1, 1, 2) but img_pts[i] was left as whatever shape
    board.matchImagePoints() returned; OpenCV 5.0.0 (unlike older versions)
    rejects the mismatch instead of tolerating it. Reshaping both sides the
    same way fixes it regardless of installed OpenCV version.
    """
    result = []
    for i in range(len(obj_pts)):
        proj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], K, dist)
        img_i = np.asarray(img_pts[i], dtype=np.float32).reshape(-1, 1, 2)
        err = cv2.norm(img_i, proj.reshape(-1, 1, 2), cv2.NORM_L2)
        result.append(float(err / np.sqrt(len(img_pts[i]))))
    return result


# ── Export + return dict ──────────────────────────────────────────────────────

def _export(selected, board, img_size, rms, K, dist, rvecs, tvecs,
            per_rms, config, output_dir, source_name, rotate, cb):
    """Save all output files and return the API result dict."""
    cal_cfg   = config.get("calibration", {})
    accept    = float(cal_cfg.get("rms_accept_threshold", 0.5))
    warn      = float(cal_cfg.get("rms_warn_threshold",   1.0))
    quality   = "EXCELLENT" if rms < accept else ("GOOD" if rms < warn else "POOR")

    ts      = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(output_dir) / f"intrinsic_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    W, H    = img_size

    cb(65, "Saving intrinsic.json, .yaml, distortion.json…")
    save_intrinsics(out_dir, rms, K, dist, img_size, source_name, len(selected))

    cb(70, "Writing calibration_report.txt…")
    write_report(out_dir, rms, K, dist, img_size, source_name,
                 len(selected), accept, warn)

    cb(75, "Saving selected_frames/…")
    save_selected_frames(out_dir, selected, source_name, board, rotate)

    cb(83, "Saving reprojection_check/…")
    save_reprojection_check(out_dir, selected, source_name, board, K, dist, rotate)

    cb(90, "Saving reprojection bar chart…")
    save_reprojection_bar_chart(per_rms, out_dir / "reprojection_error_plot.png")

    # Also save NPZ for downstream pipeline compatibility
    np.savez_compressed(
        str(out_dir / "intrinsic_params.npz"),
        camera_matrix=K, dist_coeffs=dist,
        rms=np.array([rms]), image_size=np.array(img_size),
    )

    D_vec = dist.ravel()
    files = {
        "intrinsic_json":      str(out_dir / "intrinsic.json"),
        "distortion_json":     str(out_dir / "distortion.json"),
        "intrinsic_yaml":      str(out_dir / "intrinsic.yaml"),
        "calibration_report":  str(out_dir / "calibration_report.txt"),
        "intrinsic_params_npz":str(out_dir / "intrinsic_params.npz"),
        "reprojection_plot":   str(out_dir / "reprojection_error_plot.png"),
        "selected_frames_dir": str(out_dir / "selected_frames"),
        "reprojection_check_dir": str(out_dir / "reprojection_check"),
    }

    return {
        "rms_px":     round(rms, 6),
        "quality":    quality,
        "num_frames": len(selected),
        "image_width":  W,
        "image_height": H,
        "camera_matrix": {
            "fx": round(float(K[0, 0]), 4), "fy": round(float(K[1, 1]), 4),
            "cx": round(float(K[0, 2]), 4), "cy": round(float(K[1, 2]), 4),
            "matrix_3x3": K.tolist(),
        },
        "distortion_coefficients": {
            "k1": round(float(D_vec[0]), 8),
            "k2": round(float(D_vec[1]), 8) if len(D_vec) > 1 else 0.0,
            "p1": round(float(D_vec[2]), 8) if len(D_vec) > 2 else 0.0,
            "p2": round(float(D_vec[3]), 8) if len(D_vec) > 3 else 0.0,
            "k3": round(float(D_vec[4]), 8) if len(D_vec) > 4 else 0.0,
            "vector": D_vec.tolist(),
        },
        "per_frame_rms": [round(v, 6) for v in per_rms],
        "output_dir":    str(out_dir),
        "files":         files,
    }


# ── Public entry points ───────────────────────────────────────────────────────

def run_intrinsic_calibration(
    video_path:  str,
    output_dir:  str,
    config:      dict,
    progress_cb: Optional[ProgressCallback] = None,
) -> dict:
    """
    Run intrinsic calibration from a video file.

    Args:
        video_path:  Path to ChArUco calibration video.
        output_dir:  Parent directory to write timestamped results into.
        config:      Dict with keys: board, extraction, calibration, rotate.
        progress_cb: Optional (percent, step_str) callback.
    """
    def _cb(pct, step):
        if progress_cb:
            progress_cb(pct, step)

    board_cfg = config.get("board", {})
    cal_cfg   = config.get("calibration", {})
    rotate    = config.get("rotate", "ccw90")

    _cb(2, "Building ChArUco board…")
    board, detector, dictionary = _build_charuco(board_cfg)

    _cb(5, "Selecting calibration frames…")
    selected = _select_frames(video_path, board, detector, dictionary,
                              config, rotate, progress_cb)

    min_fr = int(cal_cfg.get("min_frames", 15))
    if len(selected) < min_fr:
        raise RuntimeError(
            f"Only {len(selected)} frames selected (need >= {min_fr}). "
            "Try: lower blur_threshold, longer video, or step=1.")

    _cb(57, f"Calibrating with {len(selected)} frames…")
    rms, K, dist, rvecs, tvecs, n_used = _calibrate(selected, board,
                                                      _video_size(video_path, rotate),
                                                      cal_cfg)

    all_obj = [board.matchImagePoints(cc, cids)[0].astype(np.float32)
               for _, cc, cids in selected
               if board.matchImagePoints(cc, cids)[0] is not None]
    all_img = [board.matchImagePoints(cc, cids)[1].astype(np.float32)
               for _, cc, cids in selected
               if board.matchImagePoints(cc, cids)[1] is not None]
    per_rms = _per_frame_rms(all_obj, all_img, rvecs, tvecs, K, dist)

    logger.info(f"Calibration done: RMS={rms:.4f} px  frames={n_used}")
    return _export(selected, board, _video_size(video_path, rotate),
                   rms, K, dist, rvecs, tvecs, per_rms,
                   config, output_dir, video_path, rotate, _cb)


def _video_size(video_path: str, rotate: str) -> Tuple[int, int]:
    """Return (W, H) of the first video frame after rotation."""
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read video: {video_path}")
    h, w = rotate_frame(frame, rotate).shape[:2]
    return w, h


def run_intrinsic_calibration_from_images(
    image_paths: List[str],
    output_dir:  str,
    config:      dict,
    progress_cb: Optional[ProgressCallback] = None,
) -> dict:
    """
    Run intrinsic calibration from a list of image files.

    Images are treated as pre-selected frames; blur check applied but
    frames are not spatially subsampled (all accepted images are used).
    """
    def _cb(pct, step):
        if progress_cb:
            progress_cb(pct, step)

    board_cfg   = config.get("board", {})
    cal_cfg     = config.get("calibration", {})
    ext_cfg     = config.get("extraction", {})
    blur_thresh = float(ext_cfg.get("blur_threshold", 80.0))
    min_corners = int(board_cfg.get("min_corners", 8))
    rotate      = config.get("rotate", "ccw90")

    _cb(2, "Building ChArUco board…")
    board, detector, dictionary = _build_charuco(board_cfg)

    _cb(5, f"Processing {len(image_paths)} images…")
    selected = []
    image_size = (0, 0)

    for i, img_path in enumerate(sorted(image_paths)):
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning(f"Cannot read image: {img_path}")
            continue
        frame = rotate_frame(frame, rotate)
        if image_size == (0, 0):
            h, w = frame.shape[:2]
            image_size = (w, h)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        if sharp < blur_thresh / 2:   # halved for JPEG
            logger.info(f"Skipped (blurry): {Path(img_path).name}")
            continue
        cc, cids = _detect_charuco(gray, board, detector, dictionary, min_corners)
        if cc is not None:
            selected.append((i, cc, cids))
        _cb(int(5 + 50 * i / max(1, len(image_paths))),
            f"Detected {len(selected)}/{i+1} images")

    if not selected:
        raise RuntimeError("No ChArUco detections in any image.")

    min_fr = int(cal_cfg.get("min_frames", 15))
    if len(selected) < min_fr:
        raise RuntimeError(
            f"Only {len(selected)} detectable images (need >= {min_fr}).")

    _cb(57, f"Calibrating with {len(selected)} images…")
    rms, K, dist, rvecs, tvecs, _ = _calibrate(selected, board, image_size, cal_cfg)

    all_obj = [board.matchImagePoints(cc, cids)[0].astype(np.float32)
               for _, cc, cids in selected
               if board.matchImagePoints(cc, cids)[0] is not None]
    all_img = [board.matchImagePoints(cc, cids)[1].astype(np.float32)
               for _, cc, cids in selected
               if board.matchImagePoints(cc, cids)[1] is not None]
    per_rms = _per_frame_rms(all_obj, all_img, rvecs, tvecs, K, dist)

    return _export(selected, board, image_size, rms, K, dist, rvecs, tvecs,
                   per_rms, config, output_dir,
                   source_name=image_paths[0], rotate=rotate, cb=_cb)
