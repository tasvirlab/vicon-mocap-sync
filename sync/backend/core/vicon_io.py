"""
vicon_io.py
===========
Shared I/O utilities used by every pipeline script.

Provides:
  rotate_frame             -- rotate a BGR video frame (single source of truth)
  read_vicon_trajectories  -- load a cleaned Vicon CSV into memory
  resolve_labels           -- match requested marker names to CSV columns
  interp_marker_at_time    -- linearly interpolate a marker at any time
  load_intrinsics          -- load K, dist, image size from intrinsics JSON
  read_frame_ids           -- read a frame-identifier column from any stage's CSV
"""

from pathlib import Path
import json
import re

import cv2
import numpy as np
import pandas as pd


def read_frame_ids(csv_path: Path, candidates: tuple) -> "np.ndarray | None":
    """Return the frame-identifier column (first of `candidates` present), or
    None if none of them exist in the CSV.

    Shared by rtmpose_pipeline.py (checks rgb.mp4/synced_60hz.csv frame counts
    agree BEFORE running RTMPose) and dataset_export.py (checks pose2d.csv/
    synced_60hz.csv frame IDs agree before pairing them into a window) — same
    "stale/misaligned artifact must be rejected, not silently truncated"
    check, just at two different pipeline stages.
    """
    header = pd.read_csv(csv_path, nrows=0)
    header.columns = header.columns.str.strip()
    for c in candidates:
        if c in header.columns:
            col = pd.read_csv(csv_path, usecols=lambda x, _c=c: x.strip() == _c)
            return col.iloc[:, 0].to_numpy()
    return None


# ── Frame rotation ─────────────────────────────────────────────────────────────

def rotate_frame(frame, rotate: str):
    """
    Rotate a BGR video frame.

    Single definition used by all pipeline scripts.

    Args:
        frame:  BGR numpy array from cv2.VideoCapture.read().
        rotate: One of 'ccw90', 'cw90', '180', 'none'.

    Returns:
        Rotated (or unchanged) BGR array.
    """
    if rotate == "ccw90":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotate == "cw90":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotate == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def rotate_points_2d(pts, rotate: str, width: int, height: int):
    """Rotate pixel coordinates the same way rotate_frame() rotates the image.

    Keeps projected/2D points consistent with a rotated video frame (e.g. iPhone
    MOV needs 'ccw90'). Mirrors rotate_frame exactly.

    Args:
        pts:    (N, 2) array of (x, y) pixels in the ORIGINAL (unrotated) frame.
        rotate: One of 'ccw90', 'cw90', '180', 'none'.
        width, height: dimensions of the ORIGINAL (unrotated) frame.

    Returns:
        (N, 2) array of (x, y) pixels in the rotated frame.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    if rotate == "ccw90":
        out = np.column_stack([y, (width - 1) - x])
    elif rotate == "cw90":
        out = np.column_stack([(height - 1) - y, x])
    elif rotate == "180":
        out = np.column_stack([(width - 1) - x, (height - 1) - y])
    else:
        out = np.column_stack([x, y])
    return out


def read_vicon_trajectories(csv_path: Path, fps: float = 100.0,
                             mm_to_m: bool = True) -> dict:
    """
    Load a cleaned Vicon CSV into a trajectory dict.

    The CSV format is:
      Frame, [time_s,] <LABEL>_X, <LABEL>_Y, <LABEL>_Z, ...
    where Frame holds raw Vicon frame numbers and time_s (when present) holds
    seconds elapsed since Vicon capture start (from XCP START_TIME).

    The ``times`` array is used by the overlay for millisecond-accurate
    time-based interpolation and trimming instead of frame-number arithmetic.
    When time_s is absent (legacy CSVs), times is approximated from frames.

    Args:
        csv_path:  Path to cleaned Vicon CSV.
        fps:       Vicon capture rate in Hz.  Default 100.
        mm_to_m:   Convert Vicon millimetres to metres (divide by 1000).

    Returns:
        {
          "fps":                float,
          "frames":             np.ndarray (N,) int64  — raw Vicon frame numbers,
          "times":              np.ndarray (N,) float64 — seconds from Vicon start,
          "has_absolute_times": bool — True if time_s column was present in CSV,
          "labels":             list[str],
          "data":               {label: np.ndarray (N, 3)},
          "source":             {label: np.ndarray (N,) object} of "c3d" /
                                "interpolated" / "recovered_csv" / "invalid",
                                or {} if the CSV predates c3d_pipeline.py's
                                {joint}_Source columns (legacy file).
        }
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    x_cols = [c for c in df.columns if c.endswith("_X")]
    labels = [c[:-2] for c in x_cols]

    frames = df["Frame"].to_numpy(dtype=np.int64)

    # Use time_s column when available (written by c3d_to_clean_csv).
    # It encodes seconds from XCP START_TIME so the overlay can do exact
    # time-based trimming and interpolation without frame-number arithmetic.
    if "time_s" in df.columns:
        times = df["time_s"].to_numpy(dtype=np.float64)
        has_absolute_times = True
    else:
        # Backward compatibility: approximate from frame numbers.
        # Assumes first frame in the trimmed CSV is the start of valid data.
        # Error = (first_frame - ACTUAL_START_FIELD) / fps (typically < 1 s).
        times = (frames - frames[0]) / fps
        has_absolute_times = False

    scale = 0.001 if mm_to_m else 1.0

    data = {}
    source = {}
    for label in labels:
        xyz = df[[f"{label}_X", f"{label}_Y", f"{label}_Z"]].to_numpy(
            dtype=np.float64) * scale
        data[label] = xyz
        src_col = f"{label}_Source"
        if src_col in df.columns:
            source[label] = df[src_col].to_numpy(dtype=object)

    return {
        "fps":                float(fps),
        "frames":             frames,
        "times":              times,
        "has_absolute_times": has_absolute_times,
        "labels":             labels,
        "data":               data,
        "source":             source,
    }


def resolve_labels(all_labels: list, requested: list) -> list:
    """
    Match each requested marker name to an available label in the CSV.

    Matching order:
      1. Exact match (case-insensitive, whitespace-normalised).
      2. Substring match.

    Raises ValueError if any requested label is ambiguous or absent.
    """
    def normalise(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())

    norm_map = {normalise(l): l for l in all_labels}
    resolved = []
    for req in requested:
        n = normalise(req)
        if n in norm_map:
            resolved.append(norm_map[n])
            continue
        matches = [orig for key, orig in norm_map.items() if n in key]
        if len(matches) == 1:
            resolved.append(matches[0])
        elif len(matches) > 1:
            raise ValueError(f"Ambiguous label '{req}' matches: {matches}.")
        else:
            raise ValueError(f"Label '{req}' not found. Available: {all_labels}")
    return resolved


def interp_marker_at_time(data: np.ndarray, frames: np.ndarray,
                           fps: float, t_sec: float) -> np.ndarray:
    """
    Linearly interpolate a marker trajectory at time t_sec from recording start.

    Args:
        data:   shape (N, 3) — marker XYZ for each frame (metres).
        frames: shape (N,)   — raw Vicon frame indices.
        fps:    Vicon capture rate in Hz.
        t_sec:  Query time in seconds from start of Vicon recording.

    Returns:
        shape (3,) float — interpolated XYZ, or NaN array if out of range.
    """
    nan3 = np.full(3, np.nan)
    t_frame = t_sec * fps

    idx = int(np.searchsorted(frames, t_frame))

    if idx <= 0:
        return data[0].copy() if not np.any(np.isnan(data[0])) else nan3
    if idx >= len(frames):
        return data[-1].copy() if not np.any(np.isnan(data[-1])) else nan3

    f0 = float(frames[idx - 1])
    f1 = float(frames[idx])
    alpha = (t_frame - f0) / (f1 - f0) if f1 != f0 else 0.0

    p0 = data[idx - 1]
    p1 = data[idx]
    if np.any(np.isnan(p0)) or np.any(np.isnan(p1)):
        return nan3
    return (1.0 - alpha) * p0 + alpha * p1


def load_intrinsics(json_path: Path):
    """
    Load camera matrix K, distortion coefficients, and image dimensions.

    Supports JSON format written by the calibration backend.

    Args:
        json_path: Path to intrinsic.json.

    Returns:
        (K, dist, width, height)
          K    — numpy (3, 3) float64
          dist — numpy (1, N) float64
          width, height — int
    """
    with open(json_path, encoding="utf-8") as f:
        d = json.load(f)

    def _arr(v):
        return np.asarray(v, dtype=np.float64)

    cm = d.get("camera_matrix") or d.get("K")
    if cm is None:
        raise KeyError(f"No 'camera_matrix' or 'K' in {json_path}")
    K = _arr(cm)
    if K.ndim == 1:
        K = K.reshape(3, 3)

    dc = (d.get("dist_coefficients")
          or d.get("dist_coefs")
          or d.get("distortion_coefficients"))
    if dc is None:
        raise KeyError(f"No distortion key in {json_path}")
    dist = _arr(dc).reshape(1, -1)

    w = d.get("image_width") or d.get("width")
    h = d.get("image_height") or d.get("height")
    if w is None:
        raise KeyError(f"No 'image_width' or 'width' in {json_path}")
    if h is None:
        raise KeyError(f"No 'image_height' or 'height' in {json_path}")
    return K, dist, int(w), int(h)
