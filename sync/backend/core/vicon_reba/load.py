"""Load Vicon CSV exports into a joint position dictionary.

Supports two formats produced by the sync backend:
  - video_synced.csv  : columns [synced_video_frame, vicon_time_s, JOINT_X, ...]
  - *_clean.csv       : columns [Frame, time_s, JOINT_X, ...]

All 21 Vicon markers are expected (Z-up coordinate system, units = mm).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Ordered joint list matching the actual Vicon export column names
JOINT_NAMES = [
    "HD", "C7", "RSHO", "LSHO", "RELB", "LELB",
    "RWR", "LWR", "RHAND", "LHAND",
    "T10", "L45", "PELCEN",
    "RHIP", "LHIP", "RKNE", "LKNE",
    "RANK", "LANK", "RTOE", "LTOE",
]


def _detect_format(df: pd.DataFrame) -> str:
    """Return 'synced' or 'clean' based on column names."""
    if "synced_video_frame" in df.columns:
        return "synced"
    if "Frame" in df.columns or "frame" in df.columns:
        return "clean"
    raise ValueError(
        f"Unrecognized CSV format. First columns: {list(df.columns[:5])}"
    )


def load_csv(path: Path) -> pd.DataFrame:
    """Load a Vicon CSV (synced or clean format) and return raw DataFrame.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with columns [frame, time_s, HD_X, HD_Y, HD_Z, ..., LTOE_Z].
        Frame column is always named 'frame', time column 'time_s'.
    """
    return normalize_frame(pd.read_csv(Path(path)))


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Column-normalise an already-parsed Vicon CSV (the non-I/O half of
    load_csv). Split out so a caller holding a DataFrame from an upload gets
    byte-identical treatment to one reading from disk, rather than a second
    hand-rolled copy of the rename/drop rules."""
    fmt = _detect_format(df)

    if fmt == "synced":
        df = df.rename(columns={
            "synced_video_frame": "frame",
            "vicon_time_s": "time_s",
        })
    else:
        frame_col = "Frame" if "Frame" in df.columns else "frame"
        time_col = "time_s" if "time_s" in df.columns else "Time"
        df = df.rename(columns={frame_col: "frame", time_col: "time_s"})

    # Drop rows where all joint columns are NaN (incomplete trailing rows)
    joint_cols = [f"{j}_{ax}" for j in JOINT_NAMES for ax in ("X", "Y", "Z")]
    present = [c for c in joint_cols if c in df.columns]
    df = df.dropna(subset=present, how="all").reset_index(drop=True)

    return df


def csv_to_joints(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Convert a loaded DataFrame to a dict of (F, 3) position arrays (mm).

    Values pass through EXACTLY as the CSV wrote them — no re-interpolation
    here. Gap-filling happens exactly once, in c3d_pipeline.py, at CSV-write
    time; a second, silent interpolation pass at load time (removed) meant
    every consumer of this function saw data the clean CSV's own gap-filled
    mask no longer described, with no record that a second fill had
    happened. Any remaining NaN reaching here is either a long dropout
    c3d_pipeline.py correctly left unfilled, or came from a CSV that was
    never gap-filled at all — either way, downstream angle/REBA computation
    is what should decide how to handle it, not this loader.

    Args:
        df: DataFrame from load_csv().

    Returns:
        dict mapping joint name → np.ndarray of shape (F, 3) in mm.
        Missing joints are filled with NaN.
    """
    F = len(df)
    joints: dict[str, np.ndarray] = {}
    for name in JOINT_NAMES:
        cols = [f"{name}_{ax}" for ax in ("X", "Y", "Z")]
        if all(c in df.columns for c in cols):
            joints[name] = df[cols].to_numpy(dtype=float)
        else:
            joints[name] = np.full((F, 3), np.nan)
    return joints


def load(path: Path) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Load a Vicon CSV and return (joints_dict, raw_df).

    Args:
        path: Path to the CSV file.

    Returns:
        joints: dict {joint_name: (F, 3) ndarray in mm}
        df:     raw DataFrame with frame and time_s columns
    """
    df = load_csv(path)
    joints = csv_to_joints(df)
    return joints, df


# ── Self-check ───────────────────────────────────────────────────────────────

def _demo() -> None:
    """Assert-based self-check: a gap present in the CSV must survive into
    csv_to_joints() unchanged — no silent second interpolation pass."""
    df = pd.DataFrame({
        "frame":  [0, 1, 2, 3, 4],
        "time_s": [0.0, 0.01, 0.02, 0.03, 0.04],
        "HD_X": [1.0, np.nan, np.nan, np.nan, 5.0],
        "HD_Y": [1.0, np.nan, np.nan, np.nan, 5.0],
        "HD_Z": [1.0, np.nan, np.nan, np.nan, 5.0],
    })
    joints = csv_to_joints(df)
    hd = joints["HD"]
    assert np.isnan(hd[1:4]).all(), \
        "csv_to_joints must not re-interpolate a gap the CSV already left as NaN"
    assert hd[0, 0] == 1.0 and hd[4, 0] == 5.0, "real values must pass through unchanged"

    # A joint entirely absent from the CSV still comes back as all-NaN, not KeyError.
    missing = joints.get("PELCEN")
    assert missing is not None and np.isnan(missing).all()

    print("vicon_reba.load self-check OK")


if __name__ == "__main__":
    _demo()
