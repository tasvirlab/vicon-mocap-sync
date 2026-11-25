"""
timestamp_sync.py
=================
Wall-clock timestamp-based synchronisation between a camera video and a Vicon
motion-capture recording.

Both the Blackmagic Camera app (on iPhone) and Vicon Nexus embed precise
creation timestamps in their output files:

  Video   → QuickTime tag  ``com.blackmagic-design.camera.dateRecorded``
              e.g.  "2026-05-18T14:53:01+0900"
            Fallback: ``com.apple.quicktime.creationdate`` (same format)
            Fallback: format-level ``creation_time`` (UTC, second-level)

  Vicon   → .xcp XML attribute  <Capture START_TIME="2026-05-18 15:49:20.398 ST"/>
            "ST" = System Time (local clock of the Vicon PC).
            Millisecond precision.

Algorithm
---------
1. Parse both start timestamps and convert to UTC-aware datetimes.
2. Compute delta_s = vicon_start_utc − video_start_utc  (signed seconds).
3. The overlap window is:
     video_start_frame = max(0, round(delta_s * video_fps))   if Vicon starts later
     vicon_start_frame = max(0, round(−delta_s * vicon_fps))  if Video starts later
4. Clip to available frames and return a sync result dict.

Usage
-----
    from core.timestamp_sync import (
        parse_xcp_start_time,
        probe_video_start_time,
        compute_timestamp_sync,
        trim_vicon_csv,
    )

Timezone assumption
-------------------
The Vicon .xcp "ST" tag carries no timezone.  The video ``dateRecorded`` tag
carries an explicit offset (e.g. +0900).  We assume the Vicon PC's system
clock is set to the **same local timezone** as the camera.  This is the common
lab setup.  The ``vicon_tz_offset_hours`` parameter (default derived from the
video's own timezone) lets callers override this if needed.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

FFPROBE_BIN = shutil.which("ffprobe") or "ffprobe"


# ── XCP parser ─────────────────────────────────────────────────────────────────

def parse_xcp_start_time(
    xcp_path: Path,
    tz_offset_hours: float,
) -> datetime:
    """Parse the Vicon .xcp file and return the Capture START_TIME as a UTC datetime.

    Vicon Nexus stores per-camera <Capture START_TIME="…"> attributes.  All
    cameras share the same system-clock timestamp, so we take the first one.

    Args:
        xcp_path:        Path to the .xcp file.
        tz_offset_hours: UTC offset of the Vicon PC's system clock in hours
                         (e.g. 9.0 for UTC+9, −5.0 for UTC−5).

    Returns:
        timezone-aware datetime (UTC).

    Raises:
        ValueError: if no Capture START_TIME element is found.
        ET.ParseError: if the file is not valid XML.
    """
    tree = ET.parse(str(xcp_path))
    root = tree.getroot()

    for capture in root.iter("Capture"):
        raw = capture.get("START_TIME", "").strip()
        if raw:
            # Strip trailing " ST" (or any trailing word after the timestamp)
            dt_str = re.sub(r"\s+\w+$", "", raw)
            try:
                dt_naive = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                dt_naive = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            tz = timezone(timedelta(hours=tz_offset_hours))
            dt_local = dt_naive.replace(tzinfo=tz)
            dt_utc   = dt_local.astimezone(timezone.utc)
            logger.info(
                f"XCP START_TIME: {raw!r}  →  {dt_utc.isoformat()} UTC "
                f"(tz={tz_offset_hours:+.1f}h)"
            )
            return dt_utc

    raise ValueError(f"No <Capture START_TIME> found in {xcp_path}")


def parse_xcp_end_time(
    xcp_path: Path,
    tz_offset_hours: float,
) -> Optional[datetime]:
    """Return the Capture END_TIME as UTC datetime, or None if absent."""
    tree = ET.parse(str(xcp_path))
    root = tree.getroot()
    for capture in root.iter("Capture"):
        raw = capture.get("END_TIME", "").strip()
        if raw:
            dt_str = re.sub(r"\s+\w+$", "", raw)
            try:
                dt_naive = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                dt_naive = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            tz = timezone(timedelta(hours=tz_offset_hours))
            return dt_naive.replace(tzinfo=tz).astimezone(timezone.utc)
    return None


# ── Video timestamp extractor ──────────────────────────────────────────────────

def probe_video_start_time(video_path: str) -> Tuple[datetime, float, float, int]:
    """Extract wall-clock creation time, FPS, and frame count from video metadata via ffprobe.

    Tag priority for the start timestamp:
      1. ``com.blackmagic-design.camera.dateRecorded``  (ISO-8601 + tz offset, e.g. +0900)
      2. ``com.apple.quicktime.creationdate``           (same format)
      3. Format-level ``creation_time``                 (UTC, second precision)

    Args:
        video_path: Path to the video file.

    Returns:
        (start_utc, fps, tz_offset_hours, total_frames):
            start_utc         – UTC-aware datetime of the first recorded frame.
            fps               – declared frame rate.
            tz_offset_hours   – UTC offset found in the tag (0.0 if only UTC available).
            total_frames      – total number of frames (from nb_frames or duration × fps).

    Raises:
        RuntimeError: ffprobe not found or file unreadable.
        ValueError:   no usable timestamp found in file metadata.
    """
    cmd = [
        FFPROBE_BIN, "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        video_path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=30)
    except FileNotFoundError:
        raise RuntimeError("ffprobe not found — install FFmpeg and add it to PATH")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed on {Path(video_path).name}: {e}")

    data   = json.loads(out)
    fmt    = data.get("format", {})
    fmt_tags = fmt.get("tags", {})
    streams  = data.get("streams", [])
    vstream  = next((s for s in streams if s.get("codec_type") == "video"), None)
    if vstream is None:
        raise RuntimeError(f"No video stream found in {Path(video_path).name}")

    # ── FPS ───────────────────────────────────────────────────────────────────
    fps_str = vstream.get("r_frame_rate") or vstream.get("avg_frame_rate", "30/1")
    num, den = fps_str.split("/") if "/" in fps_str else (fps_str, "1")
    fps = float(num) / float(den) if float(den) != 0 else 30.0

    # ── Total frame count ─────────────────────────────────────────────────────
    nb_frames_str = vstream.get("nb_frames", "")
    if nb_frames_str and str(nb_frames_str).isdigit() and int(nb_frames_str) > 0:
        total_frames = int(nb_frames_str)
    else:
        duration_s = float(vstream.get("duration") or fmt.get("duration") or 0)
        total_frames = max(1, int(round(duration_s * fps))) if duration_s > 0 else 1

    # Collect all tags from both stream and format level
    all_tags = {**fmt_tags}
    for k, v in vstream.get("tags", {}).items():
        all_tags.setdefault(k, v)

    # ── Timestamp candidates (priority order) ─────────────────────────────────
    # 1) Blackmagic dateRecorded — ISO-8601 with tz (most precise)
    bmd = all_tags.get("com.blackmagic-design.camera.dateRecorded", "")
    if bmd:
        dt_utc, tz_h = _parse_iso8601_with_tz(bmd)
        logger.info(
            f"Video timestamp from Blackmagic dateRecorded: {bmd!r}  "
            f"→  {dt_utc.isoformat()} UTC  ({total_frames} frames @ {fps:.3f} fps)"
        )
        return dt_utc, fps, tz_h, total_frames

    # 2) Apple QuickTime creationdate
    qtcd = all_tags.get("com.apple.quicktime.creationdate", "")
    if qtcd:
        dt_utc, tz_h = _parse_iso8601_with_tz(qtcd)
        logger.info(
            f"Video timestamp from quicktime.creationdate: {qtcd!r}  "
            f"→  {dt_utc.isoformat()} UTC  ({total_frames} frames @ {fps:.3f} fps)"
        )
        return dt_utc, fps, tz_h, total_frames

    # 3) Format-level creation_time (UTC, second precision)
    ct = all_tags.get("creation_time", "")
    if ct:
        dt_utc, tz_h = _parse_iso8601_with_tz(ct)
        logger.info(
            f"Video timestamp from format creation_time: {ct!r}  "
            f"→  {dt_utc.isoformat()} UTC  ({total_frames} frames @ {fps:.3f} fps)"
        )
        return dt_utc, fps, tz_h, total_frames

    raise ValueError(
        f"No usable creation timestamp found in {Path(video_path).name}. "
        "Expected com.blackmagic-design.camera.dateRecorded or "
        "com.apple.quicktime.creationdate metadata tags."
    )


def _parse_iso8601_with_tz(s: str) -> Tuple[datetime, float]:
    """Parse an ISO-8601 string (with or without tz) to (utc_datetime, tz_hours).

    Handles formats:
      "2026-05-18T14:53:01+0900"      → (+9.0h offset, converted to UTC)
      "2026-05-18T14:53:01.123+0900"
      "2026-05-18T05:53:01.000000Z"   → UTC directly
      "2026-05-18T05:53:01"           → assumed UTC (returns tz_hours=0.0)
    """
    s = s.strip().replace(" ", "T")
    tz_h = 0.0

    # Try standard fromisoformat first (Python 3.7+, handles +HH:MM)
    # But Python < 3.11 doesn't handle +HHMM (no colon), so we normalise first
    # e.g. "+0900" → "+09:00"
    s_norm = re.sub(r"([+-])(\d{2})(\d{2})$", r"\1\2:\3", s)

    try:
        dt = datetime.fromisoformat(s_norm)
        if dt.tzinfo is not None:
            tz_h = dt.utcoffset().total_seconds() / 3600
            return dt.astimezone(timezone.utc), tz_h
        else:
            # No timezone info — assume UTC
            return dt.replace(tzinfo=timezone.utc), 0.0
    except ValueError:
        pass

    # Fallback: strip timezone manually
    m = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)(Z|[+-]\d{2}:?\d{2})?", s)
    if m:
        dt_str = m.group(1)
        tz_str = m.group(2) or "Z"
        try:
            dt_naive = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%f")
        except ValueError:
            dt_naive = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S")

        if tz_str in ("Z", "UTC", ""):
            return dt_naive.replace(tzinfo=timezone.utc), 0.0
        sign = 1 if tz_str[0] == "+" else -1
        digits = tz_str[1:].replace(":", "")
        hh = int(digits[:2]); mm = int(digits[2:4]) if len(digits) >= 4 else 0
        tz_h = sign * (hh + mm / 60.0)
        tz = timezone(timedelta(hours=tz_h))
        return dt_naive.replace(tzinfo=tz).astimezone(timezone.utc), tz_h

    raise ValueError(f"Cannot parse timestamp: {s!r}")


# ── Core sync computation ──────────────────────────────────────────────────────

def compute_timestamp_sync(
    video_start_utc:   datetime,
    vicon_start_utc:   datetime,
    video_fps:         float,
    vicon_fps:         float,
    video_total_frames: int,
    vicon_total_frames: int,
    vicon_end_utc:     Optional[datetime] = None,
) -> dict:
    """Compute the overlapping sync window from wall-clock timestamps.

    Sign convention mirrors the NCC path:
        delta_s > 0 → Vicon starts AFTER video   (video leads)
        delta_s < 0 → Video starts AFTER Vicon   (Vicon leads)

    Args:
        video_start_utc:    UTC datetime of first video frame.
        vicon_start_utc:    UTC datetime of first Vicon frame.
        video_fps:          Video frame rate (Hz).
        vicon_fps:          Vicon capture rate (Hz).
        video_total_frames: Total frames in video.
        vicon_total_frames: Total frames in Vicon recording.
        vicon_end_utc:      Optional UTC datetime of last Vicon frame (from .xcp END_TIME).

    Returns:
        dict with keys:
            offset_ms                  – signed ms (positive = Vicon starts later)
            offset_seconds             – same in seconds
            quality                    – always "TIMESTAMP"
            confidence                 – 1.0 (deterministic)
            sync_start_video_frame
            sync_end_video_frame
            sync_start_vicon_frame
            sync_end_vicon_frame
            sync_duration_s
            overlap_duration_s
            video_start_iso            – ISO-8601 string for display
            vicon_start_iso            – ISO-8601 string for display
    """
    delta_s = (vicon_start_utc - video_start_utc).total_seconds()
    offset_ms = delta_s * 1000.0

    # Which stream starts first?
    # If delta_s > 0: Vicon starts later → video must skip delta_s seconds at the front
    # If delta_s < 0: Video starts later → Vicon must skip |delta_s| seconds at the front
    video_start_frame = max(0, int(round(delta_s * video_fps)))   # frames to skip in video
    vicon_start_frame = max(0, int(round(-delta_s * vicon_fps)))  # frames to skip in Vicon

    # Available frames from each stream after trimming the start
    video_avail_frames = video_total_frames - video_start_frame
    vicon_avail_frames = vicon_total_frames - vicon_start_frame

    if video_avail_frames <= 0 or vicon_avail_frames <= 0:
        raise ValueError(
            f"No overlap between video (start offset {video_start_frame} fr) "
            f"and Vicon (start offset {vicon_start_frame} fr). "
            "Check that both recordings were made at the same session."
        )

    # Convert vicon_avail_frames to video-frame count for comparison
    vicon_avail_in_video_frames = int(round(vicon_avail_frames * video_fps / vicon_fps))
    sync_len_video = min(video_avail_frames, vicon_avail_in_video_frames)
    sync_len_vicon = int(round(sync_len_video * vicon_fps / video_fps))

    video_end_frame = video_start_frame + sync_len_video
    vicon_end_frame = vicon_start_frame + sync_len_vicon

    overlap_duration_s = sync_len_video / video_fps

    logger.info(
        f"Timestamp sync: delta={delta_s:+.3f}s ({offset_ms:+.1f}ms), "
        f"video[{video_start_frame}:{video_end_frame}], "
        f"vicon[{vicon_start_frame}:{vicon_end_frame}], "
        f"overlap={overlap_duration_s:.2f}s"
    )

    return {
        "offset_ms":              offset_ms,
        "offset_seconds":         delta_s,
        "offset_frames_video":    video_start_frame,   # mirrors NCC field name
        "quality":                "TIMESTAMP",
        "confidence":             1.0,
        "drift_ms_per_min":       0.0,
        "drift_quality":          "N/A (timestamp)",
        "correlation_curve":      [],
        "peak_lag_frames":        video_start_frame,
        "sync_start_video_frame": video_start_frame,
        "sync_end_video_frame":   video_end_frame,
        "sync_start_vicon_frame": vicon_start_frame,
        "sync_end_vicon_frame":   vicon_end_frame,
        "sync_duration_s":        overlap_duration_s,
        "overlap_duration_s":     overlap_duration_s,
        "video_start_iso":        video_start_utc.isoformat(),
        "vicon_start_iso":        vicon_start_utc.isoformat(),
    }


# ── Vicon CSV trimmer ──────────────────────────────────────────────────────────

def trim_vicon_csv(
    input_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
) -> Path:
    """Trim a Vicon Nexus CSV export to [start_frame, end_frame).

    Preserves all header rows (everything before the data section) and writes
    only the data rows in the specified frame range.  The output frame numbers
    are re-indexed from 1.

    Args:
        input_path:  Path to the original Vicon CSV.
        output_path: Path for the trimmed output CSV.
        start_frame: First Vicon frame to keep (0-indexed row in data section).
        end_frame:   Last Vicon frame (exclusive).

    Returns:
        output_path on success.
    """
    with open(input_path, newline="", encoding="utf-8-sig") as f:
        raw_lines = f.readlines()

    # Detect data section start (same heuristic as ViconCSVReader)
    data_start_idx = _find_data_start(raw_lines)
    header_lines   = raw_lines[:data_start_idx]
    data_lines     = raw_lines[data_start_idx:]

    selected = data_lines[start_frame:end_frame]
    if not selected:
        raise ValueError(
            f"No data rows in range [{start_frame}:{end_frame}] — "
            f"CSV has {len(data_lines)} data rows."
        )

    # Re-index frame numbers in column 0
    output_data = []
    for new_idx, line in enumerate(selected, start=1):
        parts = line.split(",", 2)
        if parts and parts[0].strip().isdigit():
            parts[0] = str(new_idx)
            line = ",".join(parts)
        output_data.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        f.writelines(header_lines)
        f.writelines(output_data)

    logger.info(
        f"Trimmed Vicon CSV: {input_path.name} "
        f"frames [{start_frame}:{end_frame}] → {output_path.name} "
        f"({len(output_data)} rows)"
    )
    return output_path


def _find_data_start(lines: list) -> int:
    """Return line index of first numeric data row in a Nexus CSV."""
    for i, line in enumerate(lines):
        s = line.strip().upper()
        if s.startswith("TRAJECTORIES"):
            return i + 5          # 5 header lines below "Trajectories"
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if (
            len(parts) >= 2
            and parts[0].isdigit()
            and _is_numeric(parts[1])
        ):
            return i
    return 0


def _is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
