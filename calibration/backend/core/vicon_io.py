"""
vicon_io.py
===========
Vicon trajectory I/O — returns the dict format used by pnp_solver.

read_vicon_trajectories() delegates the actual file parsing to
core/vicon_loader.py (raw Nexus CSV export or C3D, auto-detected by
extension) and reshapes the result into the dict this module's callers
expect.

Provides:
  read_vicon_trajectories  -- load CSV/C3D, return solver-compatible dict
  resolve_labels           -- exact-then-substring label matching
  interp_marker_at_time    -- linear interpolation at arbitrary time
  rotate_frame             -- single source of truth for frame rotation
"""

from __future__ import annotations
import re
import logging
import numpy as np
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def rotate_frame(frame: np.ndarray, rotate: str) -> np.ndarray:
    """
    Rotate a BGR video frame.  Single source of truth — import from here.

    Args:
        frame:  BGR numpy array from cv2.VideoCapture.read().
        rotate: 'ccw90' | 'cw90' | '180' | 'none'

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


def read_vicon_trajectories(path: str | Path, fps: float = 100.0,
                             mm_to_m: bool = True) -> dict:
    """
    Load Vicon trajectories (raw Nexus CSV export or C3D, auto-detected by
    extension) into a trajectory dict.

    [BUG, found 2026-07-20] This used to assume every input was already a
    "cleaned" CSV (Frame, LSHO_X, LSHO_Y, LSHO_Z, …) and fed it straight to
    pd.read_csv(). Real files hitting this function are neither: a raw Nexus
    CSV export has a multi-section header (a Devices block, then a
    Trajectories block with its own frame-rate/marker-name/units rows before
    the data starts) that plain pd.read_csv() cannot parse — it raised
    "Error tokenizing data ... Expected 1 fields ... saw 21". A raw .c3d is a
    binary motion-capture format, not text, and raised a UnicodeDecodeError
    the same way. core/vicon_loader.py already has correct, tested readers
    for both (ViconCSVReader for the real Nexus export structure,
    ViconC3DReader via ezc3d for .c3d) — this now delegates to that module
    instead of reimplementing CSV parsing here, so there is one source of
    truth for "how to read a Vicon file" rather than two disagreeing ones.

    Args:
        path:    Path to a raw Vicon Nexus CSV export, or a C3D file.
        fps:     Capture rate in Hz — used only as a fallback if the file
                 itself doesn't carry a frame rate.
        mm_to_m: Convert mm → metres (default True).

    Returns:
        {
          "fps":    float,
          "frames": np.ndarray (N,) int64  — frame indices,
          "labels": list[str],
          "data":   {label: np.ndarray (N, 3) float64} in metres
        }
    """
    from .vicon_loader import load_vicon_markers

    path = Path(path)
    markers, frame_rate = load_vicon_markers(str(path))
    if not markers:
        raise ValueError(f"No marker trajectories found in {path.name}.")

    n_frames = next(iter(markers.values())).shape[0]
    scale = 0.001 if mm_to_m else 1.0
    data = {label: xyz * scale for label, xyz in markers.items()}

    return {
        "fps": float(frame_rate or fps),
        "frames": np.arange(n_frames, dtype=np.int64),
        "labels": list(markers.keys()),
        "data": data,
    }


def resolve_labels(all_labels: list, requested: list) -> list:
    """
    Match requested marker names to available CSV labels.

    Matching order:
      1. Exact match (case-insensitive, whitespace-normalised).
      2. Substring: label starts with requested string (safe partial match).

    Unlike the previous implementation, this does NOT match labels that
    merely contain the requested string as a substring — that caused
    'LSHO' to match 'LSHO_X', 'LSHO_Y', 'LSHO_Z' in pandas-format CSVs.

    Args:
        all_labels: All labels present in the Vicon file.
        requested:  Labels the caller wants, e.g. ["LSHO", "RSHO", …].

    Returns:
        Resolved label strings in the same order as requested.
    """
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())

    norm_map = {_norm(l): l for l in all_labels}
    resolved = []
    for req in requested:
        n = _norm(req)
        # 1. Exact
        if n in norm_map:
            resolved.append(norm_map[n])
            continue
        # 2. Partial: available label starts with requested (e.g. "lsho" → "lsho:left 11")
        matches = [orig for key, orig in norm_map.items() if key.startswith(n)]
        if len(matches) == 1:
            resolved.append(matches[0])
        elif len(matches) > 1:
            raise ValueError(
                f"Ambiguous label '{req}' matches: {matches}. Be more specific."
            )
        else:
            raise ValueError(
                f"Label '{req}' not found in Vicon CSV. "
                f"Available: {all_labels[:20]}"
            )
    return resolved


def interp_marker_at_time(data: np.ndarray, frames: np.ndarray,
                           fps: float, t_sec: float) -> np.ndarray:
    """
    Linearly interpolate a marker trajectory at time t_sec (seconds).

    Args:
        data:   (N, 3) float64 — XYZ per frame in metres.
        frames: (N,)   int64   — frame indices.
        fps:    Vicon frame rate in Hz.
        t_sec:  Query time in seconds from recording start.

    Returns:
        (3,) interpolated XYZ, or NaN-filled array if out of range or
        if either bounding frame contains NaN.
    """
    nan3    = np.full(3, np.nan)
    t_frame = t_sec * fps
    idx     = int(np.searchsorted(frames, t_frame))

    if idx <= 0:
        return data[0].copy() if not np.any(np.isnan(data[0])) else nan3
    if idx >= len(frames):
        return data[-1].copy() if not np.any(np.isnan(data[-1])) else nan3

    f0    = float(frames[idx - 1])
    f1    = float(frames[idx])
    alpha = (t_frame - f0) / (f1 - f0) if f1 != f0 else 0.0
    p0, p1 = data[idx - 1], data[idx]
    if np.any(np.isnan(p0)) or np.any(np.isnan(p1)):
        return nan3
    return (1.0 - alpha) * p0 + alpha * p1
