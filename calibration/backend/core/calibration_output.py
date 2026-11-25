"""
calibration_output.py
=====================
Output helpers for intrinsic_calibration.py.

Saves intrinsic
parameters, human-readable reports, and visual verification images.
Additional: save_reprojection_bar_chart (matplotlib bar chart).
"""

from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .vicon_io import rotate_frame

logger = logging.getLogger(__name__)


def save_intrinsics(out: Path, rms: float, K: np.ndarray, dist: np.ndarray,
                    img_size: tuple, source_name: str, n_frames: int):
    """
    Write intrinsic.json, intrinsic.yaml, and distortion.json.

    JSON format:
      camera_matrix (raw 3x3 list), dist_coefficients (flat list),
      image_width, image_height, rms_reprojection_px, n_calibration_frames.
    """
    out.mkdir(parents=True, exist_ok=True)
    K_list = K.tolist()
    d_list = dist.flatten().tolist()
    ts     = datetime.now().isoformat()

    payload = {
        "camera_matrix":        K_list,
        "dist_coefficients":    d_list,
        "image_width":          img_size[0],
        "image_height":         img_size[1],
        "rms_reprojection_px":  round(rms, 6),
        "n_calibration_frames": n_frames,
        "source_video":         str(source_name),
        "date":                 ts,
    }
    with (out / "intrinsic.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with (out / "distortion.json").open("w", encoding="utf-8") as f:
        json.dump({
            "dist_coefficients": d_list,
            "image_width":  img_size[0],
            "image_height": img_size[1],
        }, f, indent=2)

    with (out / "intrinsic.yaml").open("w", encoding="utf-8") as f:
        f.write(f"# calibrate_intrinsic  {ts}\n\n")
        f.write(f"image_width:  {img_size[0]}\n")
        f.write(f"image_height: {img_size[1]}\n")
        f.write(f"rms_reprojection_px: {rms:.6f}\n")
        f.write(f"n_calibration_frames: {n_frames}\n")
        f.write("camera_matrix:\n")
        for row in K_list:
            f.write("  - [" + ", ".join(f"{v:.10g}" for v in row) + "]\n")
        f.write("dist_coefficients:\n")
        f.write("  [" + ", ".join(f"{v:.10g}" for v in d_list) + "]\n")


def write_report(out: Path, rms: float, K: np.ndarray, dist: np.ndarray,
                 img_size: tuple, source_name: str, n_frames: int,
                 accept_thresh: float, warn_thresh: float):
    """Write calibration_report.txt with focal lengths, FOV, and tips."""
    out.mkdir(parents=True, exist_ok=True)
    W, H   = img_size
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    quality = ("EXCELLENT" if rms < accept_thresh
               else "GOOD" if rms < warn_thresh
               else "POOR — consider recalibration with better coverage")

    with (out / "calibration_report.txt").open("w", encoding="utf-8") as f:
        f.write("Camera Intrinsic Calibration Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date             : {datetime.now().isoformat()}\n")
        f.write(f"Source           : {Path(source_name).name}\n")
        f.write(f"Image resolution : {W} x {H} px\n")
        f.write(f"Frames used      : {n_frames}\n\n")
        f.write(f"RMS reprojection error: {rms:.4f} px  [{quality}]\n\n")
        f.write("Camera Matrix K:\n")
        f.write(f"  fx = {fx:.4f} px\n")
        f.write(f"  fy = {fy:.4f} px\n")
        f.write(f"  cx = {cx:.4f} px   (image centre x = {W/2:.1f})\n")
        f.write(f"  cy = {cy:.4f} px   (image centre y = {H/2:.1f})\n")
        f.write(f"  fx/fy = {fx/fy:.5f}  (should be ~1.000)\n\n")
        f.write("Distortion Coefficients:\n")
        labels = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
        for i, v in enumerate(dist.flatten()):
            lbl = labels[i] if i < len(labels) else f"d{i}"
            f.write(f"  {lbl} = {v:.8f}\n")
        f.write("\nField of View (from K):\n")
        f.write(f"  Horizontal: {2*np.degrees(np.arctan(W/(2*fx))):.2f} deg\n")
        f.write(f"  Vertical  : {2*np.degrees(np.arctan(H/(2*fy))):.2f} deg\n")
        f.write("\nTips for recalibration if RMS > 1.0 px:\n")
        f.write("  - Use more varied board orientations (tilts up to 45 deg).\n")
        f.write("  - Ensure board fills the frame to different extents.\n")
        f.write("  - Move board to all image corners, not just the centre.\n")
        f.write("  - Keep board stationary during each captured frame.\n")


def save_selected_frames(out: Path, selected: list, video_path: str,
                          board, rotate: str):
    """Save selected calibration frames with ChArUco overlay to selected_frames/."""
    frames_dir = out / "selected_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    for fidx, cc, cids in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = rotate_frame(frame, rotate)
        cv2.aruco.drawDetectedCornersCharuco(frame, cc, cids, (0, 255, 0))
        scale = min(1.0, 1280.0 / frame.shape[1])
        small = cv2.resize(frame, None, fx=scale, fy=scale)
        cv2.imwrite(str(frames_dir / f"frame_{fidx:06d}.jpg"), small,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    cap.release()
    logger.info(f"Saved {len(selected)} selected frames to {frames_dir}")


def save_reprojection_check(out: Path, selected: list, video_path: str,
                             board, K: np.ndarray, dist: np.ndarray,
                             rotate: str, max_render: int = 30):
    """
    Re-run PnP on each selected frame; draw detected (green) vs reprojected (red).
    Saves to reprojection_check/ for visual quality verification.
    """
    rp_dir = out / "reprojection_check"
    rp_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    saved = 0
    for fidx, cc, cids in selected[:max_render]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = rotate_frame(frame, rotate)
        obj, img = board.matchImagePoints(cc, cids)
        if obj is None or len(obj) < 4:
            continue
        ok_pnp, rvec_f, tvec_f = cv2.solvePnP(
            obj.astype(np.float32), img.astype(np.float32), K, dist)
        if not ok_pnp:
            continue
        for pt in cc.reshape(-1, 2):
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), 2)
        proj, _ = cv2.projectPoints(obj, rvec_f, tvec_f, K, dist)
        for pt in proj.reshape(-1, 2):
            cv2.drawMarker(frame, (int(pt[0]), int(pt[1])),
                           (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        err = np.sqrt(np.mean(np.sum(
            (proj.reshape(-1, 2) - cc.reshape(-1, 2)) ** 2, axis=1)))
        cv2.putText(frame,
                    f"F:{fidx}  RMS={err:.2f}px  green=detected  red=reproj",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 220), 2)
        scale = min(1.0, 1280.0 / frame.shape[1])
        cv2.imwrite(str(rp_dir / f"check_{fidx:06d}.jpg"),
                    cv2.resize(frame, None, fx=scale, fy=scale),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved += 1
    cap.release()
    logger.info(f"Saved {saved} reprojection check images to {rp_dir}")


def save_reprojection_bar_chart(per_frame_rms: list, out_path: Path):
    """Save a matplotlib bar chart of per-frame RMS errors (app bonus output)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(6, len(per_frame_rms) * 0.25 + 2), 4),
                               facecolor="#0F1115")
        ax.set_facecolor("#1A1D24")
        ax.tick_params(colors="#94A3B8")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1E293B")
        colors = ["#22C55E" if v < 0.5 else "#F59E0B" if v < 1.0 else "#EF4444"
                  for v in per_frame_rms]
        ax.bar(range(len(per_frame_rms)), per_frame_rms, color=colors, alpha=0.85)
        ax.axhline(0.5, color="#22C55E", lw=1.2, ls="--", alpha=0.6, label="0.5 px")
        ax.axhline(1.0, color="#F59E0B", lw=1.2, ls="--", alpha=0.6, label="1.0 px")
        ax.set_xlabel("Frame", color="#94A3B8")
        ax.set_ylabel("RMS error (px)", color="#94A3B8")
        ax.set_title("Per-Frame Reprojection Error", color="#E2E8F0",
                     fontweight="bold")
        ax.legend(facecolor="#1A1D24", labelcolor="#94A3B8", fontsize=8)
        fig.tight_layout(pad=1.5)
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Could not save reprojection bar chart: {e}")
