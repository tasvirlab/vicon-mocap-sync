"""
sync_io.py
==========
Low-level I/O helpers for the synchronisation pipeline.

Provides:
  parse_xcp_capture_start     -- extract Vicon trial start time from XCP
  get_video_creation_time_utc -- extract iPhone creation_time via ffprobe
  get_video_fps_ffprobe       -- extract video FPS via ffprobe
  find_xcp                    -- auto-discover XCP from vicon_csv filename
  read_first_valid_frame      -- first Frame with every joint valid
  read_last_valid_frame       -- last  Frame with every joint valid
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import re
import subprocess

import numpy as np
import pandas as pd


def parse_xcp_capture_start(xcp_path: Path) -> datetime:
    """
    Extract Capture START_TIME from a Vicon XCP file.

    XCP contains: <Capture START_TIME="2026-05-18 15:43:11.118 ST" .../>
    "ST" = Vicon PC system (local) time; no UTC offset is stored.

    Returns:
        Naive datetime representing Vicon PC local time.
    """
    text = xcp_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'<Capture[^>]+START_TIME="([^"]+)"', text)
    if not m:
        raise ValueError(f"No Capture START_TIME found in {xcp_path}")
    raw = re.sub(r"\s+\w+$", "", m.group(1).strip()).strip()  # strip " ST"
    try:
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")


def get_video_creation_time_utc(video_path: Path) -> datetime:
    """
    Extract the video creation timestamp (UTC) via ffprobe.

    iPhone MOV/MP4 files store creation_time in the QuickTime container as UTC.

    Returns:
        Timezone-aware datetime in UTC.

    Raises:
        RuntimeError if ffprobe is missing or the tag is absent.
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", str(video_path)],
            capture_output=True, text=True, check=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ffprobe not found.  Install FFmpeg and add it to PATH.\n"
            "  Download: https://ffmpeg.org/download.html\n"
            "  Or use manual_video_start_local to bypass ffprobe."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed: {e.stderr.strip()}")

    tags = {k.lower(): v
            for k, v in json.loads(result.stdout)
                               .get("format", {}).get("tags", {}).items()}
    ct = tags.get("creation_time")
    if not ct:
        raise RuntimeError(
            f"No 'creation_time' tag in {video_path.name}.\n"
            "Use manual_video_start_local 'YYYY-MM-DD HH:MM:SS.fff' "
            "in local time."
        )
    ct_clean = ct.rstrip("Z").replace("T", " ")
    try:
        dt = datetime.strptime(ct_clean, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        dt = datetime.strptime(ct_clean, "%Y-%m-%d %H:%M:%S")
    return dt.replace(tzinfo=timezone.utc)


def get_video_fps_ffprobe(video_path: Path):
    """Return video FPS from ffprobe, or None if unavailable.

    Prefers `avg_frame_rate` (total frames / duration) over `r_frame_rate`.
    They are equal for true constant-frame-rate video and differ otherwise:
    r_frame_rate is the container's NOMINAL base rate, while avg_frame_rate is
    what the file actually contains. iPhone footage is frequently
    variable-frame-rate, which is exactly the case where r_frame_rate misleads.

    Measured on SUB01/lifting/video.MOV:
        r_frame_rate   = 60/1          = 60.0000   <- what this used to return
        avg_frame_rate = 127200/2119   = 60.0283
        nb_frames / duration           = 60.0283   <- confirms avg is correct
    Every video timestamp is frame_index / fps, so a wrong fps is a drift, not a
    constant offset: assuming 60.000 instead of 60.0283 accumulates +6.4 ms over
    an 817-frame trial. That is sub-frame here, but it grows with clip length and
    it silently biases the Vicon<->video pairing more at the end of a trial than
    at the start.
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", str(video_path)],
            capture_output=True, text=True, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    for stream in json.loads(result.stdout).get("streams", []):
        if stream.get("codec_type") == "video":
            for key in ("avg_frame_rate", "r_frame_rate"):
                frac = stream.get(key, "")
                if "/" in frac:
                    num, den = frac.split("/")
                    # avg_frame_rate is "0/0" on some streams — fall through.
                    if int(den) > 0 and int(num) > 0:
                        return int(num) / int(den)
    return None


def get_video_duration_s(video_path: Path) -> float:
    """
    Return video duration in seconds via ffprobe format duration tag.

    Falls back to OpenCV frame count / FPS if ffprobe is unavailable.

    Args:
        video_path: Path to the video file.

    Returns:
        Duration in seconds, or 0.0 if it cannot be determined.
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", str(video_path)],
            capture_output=True, text=True, check=True,
        )
        fmt = json.loads(result.stdout).get("format", {})
        dur = fmt.get("duration")
        if dur:
            return float(dur)
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        pass

    # OpenCV fallback
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps   = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps > 0 and total > 0:
            return total / fps
    except Exception:
        pass

    return 0.0


def find_xcp(xcp_dir: Path, vicon_csv: Path) -> Path:
    """
    Auto-discover the XCP file for a trial from the raw Vicon directory.

    Extracts the task tag (e.g. "Task1") and trial number (e.g. "01") from
    the cleaned CSV filename, then finds the XCP whose name matches both.

    Raises:
        FileNotFoundError if no match or multiple ambiguous matches found.
    """
    stem     = re.sub(r"_clean$", "", vicon_csv.stem)
    task_m   = re.match(r"(Task[\w\-]+?)_", stem, re.IGNORECASE)
    trial_m  = re.search(r"_(\d{2})$", stem)
    if not task_m or not trial_m:
        raise FileNotFoundError(
            f"Cannot parse task/trial from '{vicon_csv.name}'. "
            "Provide --xcp explicitly."
        )
    task_tag  = task_m.group(1).lower()
    trial_num = trial_m.group(1)
    candidates = [p for p in Path(xcp_dir).glob("*.xcp")
                  if task_tag in p.stem.lower() and p.stem.endswith(trial_num)]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No XCP in '{xcp_dir}' matching task='{task_tag}' "
            f"trial='{trial_num}'.  Provide --xcp explicitly."
        )
    exact = [p for p in candidates
             if re.search(r"[\s_]" + trial_num + r"$", p.stem)]
    if len(exact) == 1:
        return exact[0]
    raise FileNotFoundError(
        f"Ambiguous XCP matches: {[p.name for p in candidates]}. "
        "Provide --xcp explicitly."
    )


def _fully_valid_mask(vicon_csv: Path) -> "tuple[pd.DataFrame, list[str]] | None":
    """Load Frame + every {joint}_X column and return (df, x_cols), or None if
    the CSV has no joint columns at all (legacy/malformed file)."""
    header = pd.read_csv(vicon_csv, nrows=0)
    x_cols = [c for c in header.columns if c.endswith("_X")]
    if not x_cols:
        return None
    return pd.read_csv(vicon_csv, usecols=["Frame"] + x_cols), x_cols


def read_first_valid_frame(vicon_csv: Path) -> int:
    """Return the first Frame where every joint's X coordinate is present.

    NOT just the first row in the file: c3d_pipeline.py no longer truncates
    its output (a deliberate fix so one badly-occluded joint can't discard
    otherwise-good data for every other joint — see its module comments), so
    the file's first row is now typically the trial's absolute recording
    start, before Vicon has necessarily locked onto every marker. Treating
    that as "the first valid frame" silently fed a partially-occluded frame
    into the sync overlap check and the Overlay trim window as if it were a
    clean starting point.
    """
    loaded = _fully_valid_mask(vicon_csv)
    if loaded is None:
        return int(pd.read_csv(vicon_csv, usecols=["Frame"], nrows=1).iloc[0]["Frame"])
    df, x_cols = loaded
    fully_valid = df[x_cols].notna().all(axis=1)
    if not fully_valid.any():
        raise ValueError(
            f"{vicon_csv.name}: no frame has all {len(x_cols)} joints valid — "
            "cannot determine a usable start frame."
        )
    return int(df.loc[fully_valid.idxmax(), "Frame"])


def read_last_valid_frame(vicon_csv: Path) -> int:
    """Return the last Frame where every joint's X coordinate is present.
    See read_first_valid_frame() for why this is not just the file's last row."""
    loaded = _fully_valid_mask(vicon_csv)
    if loaded is None:
        return int(pd.read_csv(vicon_csv, usecols=["Frame"]).iloc[-1]["Frame"])
    df, x_cols = loaded
    fully_valid = df[x_cols].notna().all(axis=1)
    if not fully_valid.any():
        raise ValueError(
            f"{vicon_csv.name}: no frame has all {len(x_cols)} joints valid — "
            "cannot determine a usable end frame."
        )
    last_idx = fully_valid[::-1].idxmax()
    return int(df.loc[last_idx, "Frame"])


# ── Self-check ───────────────────────────────────────────────────────────────

def _demo() -> None:
    """Assert-based self-check: a leading/trailing NaN run must be skipped,
    not read as the first/last valid frame."""
    import tempfile

    df = pd.DataFrame({
        "Frame": [10, 11, 12, 13, 14, 15],
        "HD_X":     [np.nan, np.nan, 1.0, 2.0, 3.0, np.nan],
        "HD_Y":     [np.nan, np.nan, 1.0, 2.0, 3.0, np.nan],
        "HD_Z":     [np.nan, np.nan, 1.0, 2.0, 3.0, np.nan],
        "PELCEN_X": [np.nan, 0.5,   1.0, 2.0, 3.0, 4.0],
        "PELCEN_Y": [np.nan, 0.5,   1.0, 2.0, 3.0, 4.0],
        "PELCEN_Z": [np.nan, 0.5,   1.0, 2.0, 3.0, 4.0],
    })
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "test_clean.csv"
        df.to_csv(p, index=False)
        # HD is NaN for frames 10-11 and 15; PELCEN is NaN for frame 10 only.
        # "Fully valid" (both joints present) only holds for frames 12-14.
        assert read_first_valid_frame(p) == 12, read_first_valid_frame(p)
        assert read_last_valid_frame(p) == 14, read_last_valid_frame(p)

        # All-invalid case must raise, not silently return frame 0.
        df_bad = df.copy()
        df_bad["HD_X"] = np.nan
        p_bad = Path(d) / "test_bad.csv"
        df_bad.to_csv(p_bad, index=False)
        try:
            read_first_valid_frame(p_bad)
            raise AssertionError("expected ValueError for an all-invalid trial")
        except ValueError:
            pass

    print("sync_io self-check OK")


if __name__ == "__main__":
    _demo()
