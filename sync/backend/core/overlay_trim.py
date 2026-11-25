"""
overlay_trim.py
===============
Trim-point detection and synced CSV output helpers for overlay_pipeline.py.

Provides:
  find_trim_start  -- first video frame where all joints are visible
  find_trim_end    -- last  video frame where all joints are visible
  write_synced_csv -- write video-rate aligned Vicon CSV for the trimmed window
"""

from pathlib import Path

import numpy as np
import pandas as pd


def find_trim_start(frame_stats: list, total_joints: int) -> int:
    """
    Return the first video frame where all joints project inside the image.

    Falls back to the first in-range frame if no frame has all joints visible,
    and to frame 0 if there is no in-range frame at all.

    Args:
        frame_stats:  list of (video_frame, vicon_time_s, in_range, n_visible).
        total_joints: number of joints that must be visible.

    Returns:
        0-indexed video frame number.
    """
    first_in_range = None
    for V, vf, in_range, n_vis in frame_stats:
        if in_range and first_in_range is None:
            first_in_range = V
        if in_range and n_vis >= total_joints:
            return V
    return first_in_range if first_in_range is not None else 0


def find_trim_end(frame_stats: list, total_joints: int) -> int:
    """
    Return the last video frame where all joints project inside the image.

    Scans from the end of the recording.  Falls back to the last in-range
    frame if no frame has all joints visible.

    Args:
        frame_stats:  list of (video_frame, vicon_time_s, in_range, n_visible).
        total_joints: number of joints that must be visible.

    Returns:
        0-indexed video frame number (inclusive end frame).
    """
    last_in_range = None
    for V, vf, in_range, n_vis in reversed(frame_stats):
        if in_range and last_in_range is None:
            last_in_range = V
        if in_range and n_vis >= total_joints:
            return V
    return last_in_range if last_in_range is not None else frame_stats[-1][0]


def write_synced_csv(label_list: list, trimmed_stats: list,
                     output_path: Path, interp_fn, source_fn=None):
    """
    Write a video-rate aligned Vicon CSV for the trimmed frame window.

    One row per trimmed video frame.  Coordinates are in mm (matching the
    input cleaned CSV format).  Out-of-range frames are written with NaN
    so the row count equals the trimmed video frame count.

    Columns: synced_video_frame, vicon_time_s, <LABEL>_X/Y/Z, [<LABEL>_Source] …

    Args:
        label_list:    ordered list of joint label strings.
        trimmed_stats: list of (orig_V, vicon_time_s, in_range, n_vis).
        output_path:   destination CSV path.
        interp_fn:     callable(vicon_time_s: float) → {label: (3,) metres}.
                       Provided as a closure by overlay_pipeline.run_overlay.
        source_fn:     optional callable(vicon_time_s: float) →
                       {label: "c3d"|"interpolated"|"recovered_csv"|"invalid"}
                       — c3d_pipeline.py's provenance, resampled to this row's
                       timestamp. None (default) omits the _Source columns
                       entirely, for callers with a legacy clean CSV that
                       never had them.
    """
    rows = []
    for synced_idx, (V, t_s, in_range, _) in enumerate(trimmed_stats):
        row = {"synced_video_frame": synced_idx, "vicon_time_s": round(t_s, 4)}
        if in_range:
            j3d = interp_fn(t_s)
            j_src = source_fn(t_s) if source_fn else {}
        else:
            j3d = {lbl: np.full(3, np.nan) for lbl in label_list}
            j_src = {lbl: "invalid" for lbl in label_list} if source_fn else {}
        for label in label_list:
            p    = j3d.get(label, np.full(3, np.nan))
            p_mm = np.round(p * 1000.0, 6)   # metres → mm, rounded once
            row[f"{label}_X"] = None if np.isnan(p_mm[0]) else float(p_mm[0])
            row[f"{label}_Y"] = None if np.isnan(p_mm[1]) else float(p_mm[1])
            row[f"{label}_Z"] = None if np.isnan(p_mm[2]) else float(p_mm[2])
            if source_fn:
                row[f"{label}_Source"] = j_src.get(label, "invalid")
        rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(output_path, index=False, float_format="%.6f")


def write_synced_csv_2d(label_list: list, trimmed_stats: list,
                        output_path: Path, interp_2d_fn):
    """
    Write the camera-projected 2D pixel coordinates for the trimmed frame window —
    the same joints/frames as write_synced_csv, but pixel-space (X, Y) instead of
    3D mm, using the same overlay projection already computed to draw the skeleton.

    One row per trimmed video frame, same row count/order as <task>_synced.csv,
    so the two files line up frame-for-frame. Out-of-range frames are NaN.

    Columns: synced_video_frame, vicon_time_s, <LABEL>_X/Y (pixels)

    Args:
        label_list:    ordered list of joint label strings.
        trimmed_stats: list of (orig_V, vicon_time_s, in_range, n_vis).
        output_path:   destination CSV path.
        interp_2d_fn:  callable(vicon_time_s: float) → {label: (2,) pixel coords}.
                       Provided as a closure by overlay_pipeline.run_overlay.
    """
    rows = []
    for synced_idx, (V, t_s, in_range, _) in enumerate(trimmed_stats):
        row = {"synced_video_frame": synced_idx, "vicon_time_s": round(t_s, 4)}
        if in_range:
            j2d = interp_2d_fn(t_s)
        else:
            j2d = {lbl: np.full(2, np.nan) for lbl in label_list}
        for label in label_list:
            p = j2d.get(label, np.full(2, np.nan))
            row[f"{label}_X"] = None if np.isnan(p[0]) else float(round(p[0], 3))
            row[f"{label}_Y"] = None if np.isnan(p[1]) else float(round(p[1], 3))
        rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(output_path, index=False, float_format="%.6f")
