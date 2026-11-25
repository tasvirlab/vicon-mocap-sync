"""
sync_pipeline.py
================
Compute Vicon-video synchronisation offset from wall-clock timestamps.
Importable module.

Algorithm:
  1. Parse the Vicon XCP file for Capture START_TIME (Vicon PC local time).
  2. Extract the video's creation_time via ffprobe (iPhone stores UTC).
  3. Apply video_time_correction_hours to fix misconfigured phone clocks.
  4. sync_offset_s = video_UTC - vicon_UTC  (positive = video started after Vicon)
"""

from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional
import cv2

from core.sync_io import (parse_xcp_capture_start, get_video_creation_time_utc,
                           get_video_fps_ffprobe, get_video_duration_s,
                           read_first_valid_frame, read_last_valid_frame)


def compute_sync(
    xcp_path: Path,
    video_path: Path,
    vicon_csv: Path,
    timezone_hours: float,
    manual_frame_offset: int,
    manual_video_start_local: Optional[str],
    vicon_fps: float,
    video_time_correction_hours: float = 0.0,
    video_rotation: str = "ccw90",
    progress_cb=None,
) -> dict:
    """
    Compute and return the full sync metadata dict.

    Args:
        xcp_path:                    Vicon XCP for this trial.
        video_path:                  RGB video file.
        vicon_csv:                   Cleaned Vicon joint CSV.
        timezone_hours:              UTC offset of Vicon PC clock (e.g. 5.0 for UTC+5).
        manual_frame_offset:         Fine-tune in video frames.
        manual_video_start_local:    Override ffprobe: 'YYYY-MM-DD HH:MM:SS.fff'
                                     in local time.
        vicon_fps:                   Vicon sample rate in Hz.
        video_time_correction_hours: Hours to add to the iPhone's stored UTC
                                     to correct a misconfigured phone clock.
                                     Example: stored 06:56 + 9h = 15:56 local (UTC+5).
        video_rotation:              How to rotate video frames for display ('ccw90',
                                     'cw90', '180', 'none'). Default 'ccw90' for iPhone.
        progress_cb:                 Optional callback(pct: int, step: str).

    Returns:
        dict with full sync metadata.
    """
    def _cb(pct, step):
        if progress_cb:
            progress_cb(pct, step)

    tz = timezone(timedelta(hours=timezone_hours))

    _cb(10, "Parsing XCP…")
    vicon_start_naive = parse_xcp_capture_start(xcp_path)
    vicon_start_utc   = vicon_start_naive.replace(tzinfo=tz).astimezone(timezone.utc)

    _cb(20, "Reading video timestamp…")
    if manual_video_start_local:
        try:
            dt = datetime.strptime(manual_video_start_local.strip(),
                                   "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            dt = datetime.strptime(manual_video_start_local.strip(),
                                   "%Y-%m-%d %H:%M:%S")
        video_start_utc   = dt.replace(tzinfo=tz).astimezone(timezone.utc)
        video_time_source = "manual_override"
    else:
        video_start_utc   = get_video_creation_time_utc(video_path)
        video_time_source = "ffprobe_creation_time"

    video_start_utc_raw = video_start_utc  # before clock correction — needed to auto-solve one

    # Apply phone clock correction
    if video_time_correction_hours != 0.0:
        naive_corrected = (video_start_utc.replace(tzinfo=None)
                           + timedelta(hours=video_time_correction_hours))
        video_start_utc = naive_corrected.replace(tzinfo=tz).astimezone(timezone.utc)
        video_time_source += f"+{video_time_correction_hours:.0f}h_clock_correction"

    _cb(35, "Getting video FPS…")
    fps_video = get_video_fps_ffprobe(video_path)
    if fps_video is None:
        cap = cv2.VideoCapture(str(video_path))
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
    fps_video = float(fps_video)

    _cb(50, "Reading Vicon frame range…")
    first_valid = read_first_valid_frame(vicon_csv)
    last_valid  = read_last_valid_frame(vicon_csv)

    # Auto-solve the clock correction that would land video frame 0 exactly at
    # the start of valid Vicon data — the timezone offset cancels out of this
    # (it's applied identically to both sides, see docs/engineering-notes.md,
    # Synchronization), so only video_time_correction_hours can actually move the
    # offset. Solving directly saves the "how to fix" panel's previous
    # trial-and-error loop when overlap comes back False.
    vicon_naive_as_utc = vicon_start_naive.replace(tzinfo=timezone.utc)
    raw_diff_s = (video_start_utc_raw - vicon_naive_as_utc).total_seconds()
    target_offset_s = first_valid / vicon_fps
    suggested_correction_h = (target_offset_s - raw_diff_s) / 3600.0

    _cb(65, "Computing offset…")
    sync_offset_s    = (video_start_utc - vicon_start_utc).total_seconds()
    manual_offset_s  = manual_frame_offset / fps_video
    vicon_clean_start_s = first_valid / vicon_fps
    vicon_clean_end_s   = last_valid  / vicon_fps
    video_at_vicon_s = sync_offset_s + manual_offset_s

    # Use actual video duration for overlap check.
    # The old code used `video_at_vicon_s + 1.0` which falsely reported
    # "no overlap" when the video started before the valid Vicon window
    # (e.g. video starts 1.6 s into Vicon, valid data starts at 11.7 s —
    # there is real overlap but the 1-second proxy missed it).
    _cb(67, "Reading video duration…")
    video_duration_s = get_video_duration_s(video_path)
    video_end_s      = video_at_vicon_s + max(video_duration_s, 1.0)
    overlap = (video_at_vicon_s <= vicon_clean_end_s
               and video_end_s >= vicon_clean_start_s)

    _cb(80, "Building result…")

    return {
        "version": 1,
        "sync_method": "wall_clock_timestamps",
        "accuracy_note": (
            "Wall-clock sync only — accuracy ~±1 s. The camera's creation_time "
            "is quantised to whole seconds (measured: '...T06:43:13.000000Z', "
            "no sub-second digits), so this offset carries up to ±1000 ms of "
            "quantisation error on its own. Cross-correlation refinement is "
            "REQUIRED, not optional: without it a trial can be mispaired by up "
            "to ~60 video frames."
        ),
        "vicon_fps":      vicon_fps,
        "video_fps":      fps_video,
        "video_path":     str(video_path),
        "vicon_csv_path": str(vicon_csv),
        "xcp_path":       str(xcp_path),
        "vicon_capture_start_local_iso":  vicon_start_naive.isoformat(),
        "video_creation_time_utc_iso":    video_start_utc.isoformat(),
        "video_creation_time_source":     video_time_source,
        # Two DISTINCT quantities, kept as separate fields on purpose: the
        # Vicon PC's actual recording timezone (timezone_offset_hours, e.g.
        # 5.0 for UTC+5) vs. a correction for a misconfigured camera clock
        # (video_time_correction_hours, e.g. the iPhone's clock being set to
        # the wrong hour). Previously the correction was only visible embedded
        # as a string suffix on video_creation_time_source — readable by eye,
        # not by anything parsing this JSON for the actual numeric value.
        "timezone_offset_hours":          timezone_hours,
        "video_time_correction_hours":    video_time_correction_hours,
        "sync_offset_s":                  round(sync_offset_s, 6),
        "manual_frame_offset":            manual_frame_offset,
        "first_valid_raw_vicon_frame":    first_valid,
        "last_valid_raw_vicon_frame":     last_valid,
        "first_valid_vicon_time_s":       round(first_valid / vicon_fps, 6),
        "video_at_vicon_start_s":         round(video_at_vicon_s, 6),
        "video_duration_s":               round(video_duration_s, 3),
        "overlap":                        overlap,
        "video_rotation":                 video_rotation,
        "suggested_clock_correction_hours": round(suggested_correction_h, 4),
    }
