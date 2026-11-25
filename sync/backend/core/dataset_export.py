"""
dataset_export.py
====================
Convert the sync backend's `<task>_synced.csv` (3D, mm) + `<task>_pose2d.csv` (2D, pixels,
from rtmpose_pipeline.py) into windowed, lifting-model-ready train/test `.npz` files.

Adapted from the standalone vicon_dataset_converter.py CLI script for use by
routers/dataset.py's job-based endpoint. See that script's original docstring
for full design rationale — repeated briefly here:

  - Joint order is the CANONICAL 21-joint order shared by the training loss
    (loss_construction.py) and the live-app RTMPose
    extractor — Pelvis at index 3. Both 2D and 3D are emitted in this order so
    the lifting model's loss/eval root-centering (index 3) lands on the pelvis.
    Do NOT revert to the old vicon_reba order (PELCEN at 12): the loss indexes
    Pelvis=3, so that order silently mis-rooted every REBA-weighted/bone term.
  - T10/L45 2D approximation ratios are recalibrated from real 3D Vicon data,
    TRAINING SUBJECTS ONLY, then frozen and applied to all splits.
  - 2D normalization is camera-invariant: root-center on PELCEN, scale by half
    the video width. No learned scaler — generalizes to any camera setup.
  - Windows are taken per-trial only, never spanning two trials/subjects.
  - Subject-wise train/val/test split (3-way) — a subject's frames never appear
    in more than one split. Val exists so model selection never touches test.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
from utils.task_paths import discover_task_dirs, Artifacts, read_metadata
from core.vicon_io import read_frame_ids

# CANONICAL 21-joint order — index-for-index with loss_construction.py:
# Head0 Neck_C7 1 Chest_T8 2 Pelvis 3 L_Shoulder4 R_Shoulder5 L_Elbow6 R_Elbow7
# L_Wrist8 R_Wrist9 L_Hand10 R_Hand11 L_Hip12 R_Hip13 L_Knee14 R_Knee15
# L_Ankle16 R_Ankle17 L_Foot18 R_Foot19 Spine_L5 20
# (Sync backend marker names map: Chest_T8->T10, Pelvis->PELCEN, Foot->TOE, Spine_L5->L45.)
JOINT_NAMES: list[str] = [
    "HD", "C7", "T10", "PELCEN", "LSHO", "RSHO", "LELB", "RELB", "LWR", "RWR",
    "LHAND", "RHAND", "LHIP", "RHIP", "LKNE", "RKNE", "LANK", "RANK",
    "LTOE", "RTOE", "L45",
]
N_JOINTS = len(JOINT_NAMES)
assert N_JOINTS == 21
assert (JOINT_NAMES[1] == "C7" and JOINT_NAMES[2] == "T10"
        and JOINT_NAMES[3] == "PELCEN" and JOINT_NAMES[20] == "L45"), (
    "Joint order drifted from the canonical loss_construction.py contract "
    "(Pelvis must be index 3) — do not silently re-index."
)

ROOT_JOINT = "PELCEN"
ROOT_IDX = JOINT_NAMES.index(ROOT_JOINT)
T10_IDX, L45_IDX, C7_IDX = JOINT_NAMES.index("T10"), JOINT_NAMES.index("L45"), JOINT_NAMES.index("C7")
MM_TO_M = 1.0 / 1000.0

SUBID_RE = re.compile(r"^(SUB\d+)_")

MIN_TRIAL_VALID_FRACTION = 0.5
MIN_WINDOW_VALID_FRACTION_DEFAULT = 0.9
WINDOW_SIZE_DEFAULT = 243
STEP_SIZE_DEFAULT = 81
# Max consecutive interpolated frames allowed in the 3D TARGET of a window.
# Longer occlusions fabricate ground truth, so those windows are dropped.
# 2D input keeps a looser limit (fabricated inputs are far less harmful).
MAX_TARGET_GAP_DEFAULT = 5
MAX_INPUT_GAP_DEFAULT = 30

# Canonical temporal rate every trial's window must represent, regardless of
# the camera's native fps. 8/10 subjects are natively ~60.03 fps CFR (left
# untouched — 60.03 already reads as "60fps" to any reported precision, same
# reasoning as overlay_pipeline.py's frame-rate note). SUB02/SUB03 are
# natively ~30 fps, so without resampling a 243-frame window would span
# ~8.1s for them vs ~4.05s for everyone else — different real-world duration
# AND a wrong implicit Δt for any future velocity/acceleration computation.
CANONICAL_FPS = 60.0
FPS_MISMATCH_TOLERANCE_HZ = 5.0  # only trips for genuine 30fps, not 60.03 jitter

# overlay_pipeline.py's _ffmpeg_trim()/_verify_normalized() already hard-fails
# unless every rgb.mp4 leaving that stage is exactly this size, so RTMPose
# (which runs on rgb.mp4) can never have produced coordinates in any other
# pixel space. get_video_dims_from_metadata() below only ASSERTS this
# pre-existing guarantee still holds — it does not re-derive or rescale
# coordinates, since they are already correct.
NORMALIZED_WIDTH = 1920
NORMALIZED_HEIGHT = 1080

ProgressCb = Optional[Callable[[int, str], None]]


# ── Loading ──────────────────────────────────────────────────────────────────────────

def parse_subject_id(task_id: str) -> str:
    """Extract SUBID from a composite task_id string, e.g. 'SUB01_lifting_T01' -> 'SUB01'.

    Operates on the task_id string itself, NOT trial_dir.name — trial_dir may
    now be nested (outputs/SUB01/lifting/T01/), where .name is just "T01".
    """
    m = SUBID_RE.match(task_id)
    if not m:
        raise ValueError(f"Could not parse SUBID from '{task_id}' "
                          f"(expected '<SUBID>_<TASK>_<TRIAL>' naming).")
    return m.group(1)


def _load_joint_csv(csv_path: Path, suffixes: tuple[str, ...]) -> np.ndarray:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    missing = [n for n in JOINT_NAMES if not all(f"{n}_{s}" in df.columns for s in suffixes)]
    if missing:
        raise ValueError(f"{csv_path.name}: missing columns for joints {missing}")
    cols = [f"{n}_{s}" for n in JOINT_NAMES for s in suffixes]
    return df[cols].to_numpy(dtype=np.float64).reshape(len(df), N_JOINTS, len(suffixes))


def load_trial_3d_mm(csv_path: Path) -> np.ndarray:
    return _load_joint_csv(csv_path, ("X", "Y", "Z"))


def load_trial_2d_px(csv_path: Path) -> np.ndarray:
    return _load_joint_csv(csv_path, ("X", "Y"))


def _load_joint_scalar_csv(csv_path: Path, suffix: str) -> np.ndarray:
    """Like _load_joint_csv but for one value per joint, not a coordinate
    pair — used for pose2d_rtmpose.csv's {joint}_conf/{joint}_valid columns.

    Raises ValueError rather than defaulting silently: a pose2d_rtmpose.csv
    written before rtmpose_pipeline.py started recording confidence/
    visibility must be re-run through 2D Pose, not treated as if every joint
    were fully confident and originally visible.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    missing = [n for n in JOINT_NAMES if f"{n}_{suffix}" not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path.name}: missing '_{suffix}' column for joints {missing} "
            f"— this trial's pose2d.csv predates confidence/visibility "
            f"tracking; re-run 2D Pose for this trial."
        )
    cols = [f"{n}_{suffix}" for n in JOINT_NAMES]
    return df[cols].to_numpy(dtype=np.float64)


def load_trial_confidence(csv_path: Path) -> np.ndarray:
    """(F, 21) raw RTMPose confidence per joint, from pose2d_rtmpose.csv."""
    return _load_joint_scalar_csv(csv_path, "conf")


def load_trial_visibility(csv_path: Path) -> np.ndarray:
    """(F, 21) bool — True where the ORIGINAL RTMPose detection passed
    threshold. Never True for anything filled in later by interpolation."""
    return _load_joint_scalar_csv(csv_path, "valid").astype(bool)


def load_trial_3d_source(csv_path: Path) -> np.ndarray:
    """(F, 21) object array of c3d_pipeline.py's per-joint provenance
    ("c3d"/"interpolated"/"recovered_csv"/"invalid"), resampled to 60Hz and
    carried through by overlay_pipeline.py into synced_60hz.csv's
    {joint}_Source columns.

    Raises ValueError rather than defaulting silently: a synced_60hz.csv
    written before this propagation existed must be re-generated (re-run
    Overlay), not treated as if every frame were a raw C3D measurement.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    missing = [n for n in JOINT_NAMES if f"{n}_Source" not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path.name}: missing '_Source' column for joints {missing} "
            f"— this trial's synced_60hz.csv predates provenance tracking; "
            f"re-run Overlay for this trial."
        )
    cols = [f"{n}_Source" for n in JOINT_NAMES]
    return df[cols].to_numpy(dtype=object)


def get_video_dims(task_id: str, trial_dir: Path) -> tuple[int, int]:
    video_path = trial_dir / Artifacts.RGB_VIDEO
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {video_path} to read frame dimensions.")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w <= 0 or h <= 0:
        raise ValueError(f"{video_path}: invalid dimensions ({w}x{h})")
    return w, h


def get_video_dims_from_metadata(meta: dict) -> tuple[int, int]:
    """Read + verify resolution from metadata.json's recorded video_resolution,
    instead of opening rgb.mp4 via get_video_dims() above.

    Two reasons to prefer this: (1) a copied/staged dataset (e.g. 02_PROCESSED)
    may legitimately not carry the multi-GB rgb.mp4 alongside the CSVs dataset
    export actually needs; (2) resolution is already hard-normalised and
    verified to NORMALIZED_WIDTH x NORMALIZED_HEIGHT by overlay_pipeline.py's
    _verify_normalized() before rgb.mp4 is ever written, so RTMPose's 2D output
    is already guaranteed to be in that pixel space — re-deriving it here would
    only risk rescaling coordinates that are already correct. This function
    ASSERTS that guarantee rather than re-implementing it.
    """
    res = meta.get("video_resolution")
    if not res:
        raise ValueError(
            "metadata.json missing 'video_resolution' — cannot verify the "
            f"{NORMALIZED_WIDTH}x{NORMALIZED_HEIGHT} normalisation guarantee "
            "for this trial."
        )
    m = re.match(r"^(\d+)x(\d+)$", str(res).strip())
    if not m:
        raise ValueError(f"Unparseable video_resolution {res!r} in metadata.json.")
    w, h = int(m.group(1)), int(m.group(2))
    if (w, h) != (NORMALIZED_WIDTH, NORMALIZED_HEIGHT):
        raise ValueError(
            f"video_resolution={w}x{h}, expected "
            f"{NORMALIZED_WIDTH}x{NORMALIZED_HEIGHT}. overlay_pipeline.py's "
            f"normalisation guarantee did not hold for this trial — do not "
            f"feed its 2D coordinates into training without investigating."
        )
    return w, h


# ── T10/L45 calibration (train subjects only) ───────────────────────────────────────

def fit_spine_ratio(positions_3d_mm: np.ndarray, target_idx: int) -> float:
    c7 = positions_3d_mm[:, C7_IDX, :]
    pelcen = positions_3d_mm[:, ROOT_IDX, :]
    target = positions_3d_mm[:, target_idx, :]

    valid = np.isfinite(c7).all(axis=1) & np.isfinite(pelcen).all(axis=1) & np.isfinite(target).all(axis=1)
    if not valid.any():
        raise ValueError("No valid frames to calibrate spine ratio from training data.")

    v = pelcen[valid] - c7[valid]
    u = target[valid] - c7[valid]
    denom = np.einsum("ij,ij->i", v, v)
    denom = np.where(denom < 1e-9, np.nan, denom)
    t = np.einsum("ij,ij->i", u, v) / denom

    t = t[np.isfinite(t)]
    if t.size == 0:
        raise ValueError("All frames degenerate (C7 == PELCEN) — cannot calibrate spine ratio.")
    return float(np.mean(t))


def apply_calibrated_spine_2d(positions_2d_px: np.ndarray, t_t10: float, t_l45: float) -> np.ndarray:
    positions_2d_px = positions_2d_px.copy()
    c7 = positions_2d_px[:, C7_IDX, :]
    pelcen = positions_2d_px[:, ROOT_IDX, :]
    positions_2d_px[:, T10_IDX, :] = c7 + t_t10 * (pelcen - c7)
    positions_2d_px[:, L45_IDX, :] = c7 + t_l45 * (pelcen - c7)
    return positions_2d_px


# ── Normalization ────────────────────────────────────────────────────────────────────

def normalize_2d(positions_2d_px: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    pelvis = positions_2d_px[:, ROOT_IDX, :]
    pelvis_valid = np.isfinite(pelvis).all(axis=1)
    centered = positions_2d_px - pelvis[:, None, :]
    normalized = centered / (width / 2.0)
    joints_valid = np.isfinite(positions_2d_px).all(axis=(1, 2))
    return normalized, joints_valid & pelvis_valid


def normalize_3d(positions_3d_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pelvis = positions_3d_mm[:, ROOT_IDX, :]
    pelvis_valid = np.isfinite(pelvis).all(axis=1)
    relative_mm = positions_3d_mm - pelvis[:, None, :]
    positions_3d_m = relative_mm * MM_TO_M
    joints_valid = np.isfinite(positions_3d_mm).all(axis=(1, 2))
    return positions_3d_m, joints_valid & pelvis_valid


# ── Frame-rate normalisation (before windowing — see CANONICAL_FPS above) ────────────

def resample_to_canonical_fps(
    positions_2d_px: np.ndarray, positions_3d_mm: np.ndarray, confidence: np.ndarray,
    visibility_mask: np.ndarray, target_3d_source: np.ndarray, native_fps: float,
    target_fps: float = CANONICAL_FPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample one trial's raw per-frame arrays onto a target_fps timeline.

    Continuous arrays (positions, confidence) use time-based LINEAR
    interpolation. Categorical/boolean arrays (visibility_mask,
    target_3d_source) use NEAREST-neighbour hold — you cannot linearly
    interpolate a category or a boolean.

    NaN propagates correctly through np.interp: a resampled instant that falls
    between two native samples where either is NaN comes out NaN too, so a
    genuinely-occluded native frame still marks its neighbourhood invalid
    after resampling instead of silently smoothing through it. The window-
    level gap-fill (`_interpolate_window_gaps`, called later by make_windows)
    still applies on top of this, unchanged.

    This does NOT add real motion information for an upsampled (e.g. 30fps)
    trial — it only gives every trial a consistent timestep and per-window
    real-world duration for windowing. Callers must keep the trial's
    ORIGINAL native_fps as separate provenance (see ProcessedTrial.native_fps)
    so nothing later claims interpolated 30fps footage carries true 60fps
    motion detail.
    """
    n = positions_2d_px.shape[0]
    if n < 2:
        raise ValueError(f"Cannot resample a {n}-frame trial (need >= 2 frames).")

    t_native = np.arange(n, dtype=np.float64) / native_fps
    duration = t_native[-1]
    n_target = int(round(duration * target_fps)) + 1
    t_target = np.arange(n_target, dtype=np.float64) / target_fps

    def _interp_continuous(arr: np.ndarray) -> np.ndarray:
        orig_shape = arr.shape
        flat = arr.reshape(n, -1).astype(np.float64)
        out = np.empty((n_target, flat.shape[1]), dtype=np.float64)
        for c in range(flat.shape[1]):
            out[:, c] = np.interp(t_target, t_native, flat[:, c])
        return out.reshape((n_target,) + orig_shape[1:])

    # Nearest-hold index: native_fps is CFR (constant-frame-rate — verified
    # upstream by overlay_pipeline.assert_constant_frame_rate), so the
    # closest native sample to t_target[i] is simply round(t*native_fps).
    nearest_idx = np.clip(np.round(t_target * native_fps).astype(int), 0, n - 1)

    positions_2d_r = _interp_continuous(positions_2d_px)
    positions_3d_r = _interp_continuous(positions_3d_mm)
    confidence_r = _interp_continuous(confidence)
    visibility_mask_r = visibility_mask[nearest_idx]
    target_3d_source_r = target_3d_source[nearest_idx]

    return positions_2d_r, positions_3d_r, confidence_r, visibility_mask_r, target_3d_source_r


# ── Windowing ────────────────────────────────────────────────────────────────────────

def _max_interior_gap(valid: np.ndarray) -> int:
    """Longest run of False strictly between the first and last True (interior)."""
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return len(valid)
    interior = valid[idx[0]:idx[-1] + 1]
    longest = run = 0
    for v in interior:
        run = 0 if v else run + 1
        longest = max(longest, run)
    return longest


def _interpolate_window_gaps(window: np.ndarray, valid: np.ndarray,
                             max_gap: int | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """Linearly fill interior gaps. Returns (filled_window, interpolation_mask),
    or None (reject window) if any interior gap is longer than max_gap.

    interpolation_mask is (T, J) bool, True on every joint of a frame that was
    invalid before this call and finite after. Validity here is per-FRAME —
    normalize_2d/normalize_3d require ALL joints finite for a frame to count
    as valid — so every joint in an interpolated frame gets the same flag;
    this reflects the actual per-frame granularity this gap logic already
    operates at, not a genuinely per-joint interpolation decision.

    max_gap guards the 3D TARGET: interpolating a long occlusion fabricates the
    ground truth the model is scored against. Pass a small max_gap for 3D; the
    2D input can use a looser limit (None = fill any interior gap).
    """
    T, J = window.shape[0], window.shape[1]
    if valid.all():
        return window, np.zeros((T, J), dtype=bool)
    if max_gap is not None and _max_interior_gap(valid) > max_gap:
        return None
    out = window.copy()
    flat = out.reshape(T, -1)
    for col in range(flat.shape[1]):
        s = pd.Series(flat[:, col])
        flat[:, col] = s.interpolate(method="linear", limit_direction="both", limit_area="inside").to_numpy()
    out = flat.reshape(window.shape)
    if not np.isfinite(out).all():
        return None
    interpolation_mask = np.broadcast_to((~valid)[:, None], (T, J)).copy()
    return out, interpolation_mask


def make_windows(
    positions_2d: np.ndarray, positions_3d: np.ndarray, valid_mask: np.ndarray,
    confidence: np.ndarray, visibility_mask: np.ndarray, target_3d_source: np.ndarray,
    window_size: int, step_size: int, min_valid_fraction: float,
    max_target_gap: int = MAX_TARGET_GAP_DEFAULT,
    max_input_gap: int = MAX_INPUT_GAP_DEFAULT,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Returns a list of (input_2d, target_3d, confidence, visibility_mask,
    interpolation_mask, target_3d_source, target_3d_interpolated_by_export)
    windows.

    confidence/visibility_mask carry RTMPose's per-joint values straight
    through. interpolation_mask is the 2D-input gap-fill mask from this
    function's own windowing. target_3d_source is c3d_pipeline.py's
    provenance (c3d/interpolated/recovered_csv/invalid), sliced unchanged —
    it's descriptive metadata about already-computed values, not something
    to re-interpolate. target_3d_interpolated_by_export is the 3D target's
    OWN window-level gap-fill mask (previously computed and discarded here)."""
    T = positions_2d.shape[0]
    windows = []
    for start in range(0, T - window_size + 1, step_size):
        end = start + window_size
        w_valid = valid_mask[start:end]
        if w_valid.mean() < min_valid_fraction:
            continue
        # 3D target: strict gap cap (fabricated ground truth is harmful).
        w3d_result = _interpolate_window_gaps(positions_3d[start:end], w_valid, max_target_gap)
        if w3d_result is None:
            continue
        w3d, w3d_interp_mask = w3d_result
        # 2D input: looser cap.
        w2d_result = _interpolate_window_gaps(positions_2d[start:end], w_valid, max_input_gap)
        if w2d_result is None:
            continue
        w2d, w2d_interp_mask = w2d_result
        windows.append((
            w2d, w3d, confidence[start:end], visibility_mask[start:end], w2d_interp_mask,
            target_3d_source[start:end], w3d_interp_mask,
        ))
    return windows


# ── Per-trial pipeline ───────────────────────────────────────────────────────────────

@dataclass
class ProcessedTrial:
    subject_id: str
    task_name: str
    windows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    native_fps: float  # ORIGINAL camera fps, always — even when the trial was
                       # resampled to CANONICAL_FPS for windowing. Kept as
                       # provenance so a resampled 30fps trial is never later
                       # mistaken for genuine 60fps motion detail.


def process_trial(
    task_id: str, trial_dir: Path, meta: dict, t_t10: float, t_l45: float,
    window_size: int, step_size: int, min_window_valid_fraction: float,
) -> tuple[ProcessedTrial | None, str | None]:
    """Returns (ProcessedTrial, None) on success, or (None, reason) if skipped.

    `meta` is this trial's metadata.json (already loaded by the caller for the
    sync gate check) — reused here for video_resolution and video_fps rather
    than re-deriving either from rgb.mp4, which a staged/copied dataset may
    not even carry alongside the CSVs this function actually needs.
    """
    synced_csv = trial_dir / Artifacts.SYNCED_60HZ
    pose2d_csv = trial_dir / Artifacts.POSE2D_RTMPOSE
    if not synced_csv.exists() or not pose2d_csv.exists():
        return None, "missing synced.csv or pose2d.csv"

    native_fps = meta.get("video_fps")
    if not native_fps:
        return None, "metadata.json missing video_fps — cannot verify frame timing"

    try:
        subject_id = parse_subject_id(task_id)
        positions_3d_mm = load_trial_3d_mm(synced_csv)
        positions_2d_px = load_trial_2d_px(pose2d_csv)
        confidence = load_trial_confidence(pose2d_csv)
        visibility_mask = load_trial_visibility(pose2d_csv)
        target_3d_source = load_trial_3d_source(synced_csv)
        width, _height = get_video_dims_from_metadata(meta)
    except (ValueError, FileNotFoundError) as e:
        return None, str(e)

    # Require the 2D and 3D artifacts to describe the SAME frames. Both stages
    # index frames 0..N-1 over the same rgb.mp4 / synced window, so any length or
    # frame-id difference means a stale/inconsistent artifact (e.g. Overlay
    # re-run without re-running 2D Pose). Silent truncation to the shorter length
    # only aligns correctly if the drop is at the TAIL; an internal or leading
    # drop would mis-pair every subsequent 2D-3D frame. Reject, do not guess.
    # This check runs on the RAW native-fps arrays, before any fps resampling
    # below — resampling only ever runs on two arrays already proven aligned.
    frames_3d = read_frame_ids(synced_csv, ("synced_video_frame", "frame"))
    frames_2d = read_frame_ids(pose2d_csv, ("frame", "synced_video_frame"))
    if frames_3d is not None and frames_2d is not None:
        if len(frames_3d) != len(frames_2d) or not np.array_equal(frames_3d, frames_2d):
            return None, (f"2D/3D frame mismatch (3D={len(frames_3d)} frames, "
                          f"2D={len(frames_2d)} frames) — stale/inconsistent "
                          f"artifacts; re-run Overlay and 2D Pose together")
    elif positions_3d_mm.shape[0] != positions_2d_px.shape[0]:
        return None, (f"2D/3D length mismatch (3D={positions_3d_mm.shape[0]}, "
                      f"2D={positions_2d_px.shape[0]}) with no frame-id column to "
                      f"align on — rejecting rather than truncating blindly")

    # FPS normalisation — BEFORE windowing (see CANONICAL_FPS note above), so
    # every window represents the same real-world duration and Δt regardless
    # of the source camera's native rate. Left untouched for the ~60.03fps
    # majority (within tolerance of CANONICAL_FPS already); only trips for a
    # genuine mismatch like SUB02/SUB03's native ~30fps.
    if abs(native_fps - CANONICAL_FPS) > FPS_MISMATCH_TOLERANCE_HZ:
        try:
            (positions_2d_px, positions_3d_mm, confidence,
             visibility_mask, target_3d_source) = resample_to_canonical_fps(
                positions_2d_px, positions_3d_mm, confidence,
                visibility_mask, target_3d_source, native_fps,
            )
        except ValueError as e:
            return None, f"fps resample failed: {e}"

    positions_2d_px = apply_calibrated_spine_2d(positions_2d_px, t_t10, t_l45)
    positions_2d_norm, valid_2d = normalize_2d(positions_2d_px, width)
    positions_3d_norm, valid_3d = normalize_3d(positions_3d_mm)
    valid_mask = valid_2d & valid_3d

    if valid_mask.mean() < MIN_TRIAL_VALID_FRACTION:
        return None, f"only {valid_mask.mean():.0%} of frames usable"

    windows = make_windows(positions_2d_norm, positions_3d_norm, valid_mask,
                            confidence, visibility_mask, target_3d_source,
                            window_size, step_size, min_window_valid_fraction)
    if not windows:
        return None, "no valid windows produced (too short/corrupted)"

    return ProcessedTrial(subject_id=subject_id, task_name=task_id, windows=windows,
                          native_fps=float(native_fps)), None


def discover_trials(outputs_dir: Path) -> list[tuple[str, Path]]:
    """Find every trial under outputs/ (any nesting depth) as (task_id, trial_dir) pairs."""
    return sorted(discover_task_dirs(outputs_dir))


# ── Synchronization export gate ────────────────────────────────────────────────────────

def sync_gate_reason(meta: dict) -> Optional[str]:
    """Return a block reason string if this trial's metadata fails the sync
    review gate, or None if it's allowed through.

    Blocked whenever sync_needs_review is True OR missing (fail-safe — matches
    the sync stage's own default, refine_result.get("needs_review", True)),
    UNLESS manual_review_approved is explicitly True in the same metadata.json.
    There is no other way past this: no request-level flag can bypass it, only
    a deliberate per-trial edit to that trial's own metadata.json.
    """
    needs_review = meta.get("sync_needs_review")
    approved = meta.get("manual_review_approved") is True
    if needs_review is not False and not approved:
        return (
            f"blocked: sync needs manual review (needs_review={needs_review!r}). "
            f"Check qa/diagnostic_sync.png, then set "
            f'"manual_review_approved": true in this trial\'s metadata.json '
            f"to allow export."
        )
    return None


# ── Subject-wise split ───────────────────────────────────────────────────────────────

def split_subjects(
    trials: list[tuple[str, Path]],
    test_subjects: list[str] | None,
    val_subjects: list[str] | None,
    test_fraction: float,
    val_fraction: float,
) -> tuple[set[str], set[str], set[str]]:
    """Subject-wise 3-way split. A subject never appears in more than one split.

    Explicit test_subjects / val_subjects override the fraction-based auto-split.
    Auto-split holds out the last N_test subjects (sorted) as test, the next
    N_val as val, the rest as train. Degrades gracefully on tiny cohorts:
    with <3 subjects val (and possibly test) may be empty so train is never
    starved — T10/L45 calibration always has training data.
    """
    all_set = {parse_subject_id(task_id) for task_id, _ in trials}
    all_subjects = sorted(all_set)
    n = len(all_subjects)

    # Explicit lists must reference subjects that actually exist — a typo'd or
    # stale SUBID would otherwise silently produce an empty/wrong split.
    if test_subjects is not None:
        unknown = set(test_subjects) - all_set
        if unknown:
            raise ValueError(f"test_subjects not present in the data: {sorted(unknown)}")
        test_set = set(test_subjects)
    elif n < 2:
        test_set = set()
    else:
        n_test = max(1, round(n * test_fraction))
        test_set = set(all_subjects[n - n_test:])

    remaining = [s for s in all_subjects if s not in test_set]
    if val_subjects is not None:
        unknown = set(val_subjects) - all_set
        if unknown:
            raise ValueError(f"val_subjects not present in the data: {sorted(unknown)}")
        val_set = set(val_subjects)
    elif len(remaining) < 2:
        # Only one subject left after test — keep it for training, no val split.
        val_set = set()
    else:
        n_val = max(1, round(n * val_fraction))
        n_val = min(n_val, len(remaining) - 1)  # never empty train
        val_set = set(remaining[len(remaining) - n_val:]) if n_val > 0 else set()

    # No subject may appear in two splits. train = all - test - val makes train
    # disjoint by construction, but explicit val/test lists can overlap EACH
    # OTHER — the same subject in val AND test contaminates model selection
    # (val picks the model, test reports it). Reject rather than silently allow.
    vt_overlap = val_set & test_set
    if vt_overlap:
        raise ValueError(
            f"val and test splits overlap on {sorted(vt_overlap)} — a subject "
            f"cannot be in both (val is used for model selection, test for the "
            f"final report). Fix the explicit subject lists.")

    train_set = all_set - test_set - val_set
    if not train_set:
        raise ValueError(
            f"empty training split: test={sorted(test_set)} + val={sorted(val_set)} "
            f"cover all {n} subjects. Leave at least one subject for training.")

    return train_set, val_set, test_set


# ── Stacking + export ────────────────────────────────────────────────────────────────

def stack_windows(processed: list[ProcessedTrial]) -> dict[str, np.ndarray]:
    all_2d, all_3d, all_conf, all_vis, all_interp = [], [], [], [], []
    all_3d_source, all_3d_interp = [], []
    subject_ids, task_names, native_fps_list = [], [], []
    for trial in processed:
        for w2d, w3d, w_conf, w_vis, w_interp, w_3d_source, w_3d_interp in trial.windows:
            all_2d.append(w2d)
            all_3d.append(w3d)
            all_conf.append(w_conf)
            all_vis.append(w_vis)
            all_interp.append(w_interp)
            all_3d_source.append(w_3d_source)
            all_3d_interp.append(w_3d_interp)
            subject_ids.append(trial.subject_id)
            task_names.append(trial.task_name)
            native_fps_list.append(trial.native_fps)

    if not all_2d:
        return {
            "positions_2d": np.zeros((0, 0, N_JOINTS, 2), dtype=np.float32),
            "positions_3d": np.zeros((0, 0, N_JOINTS, 3), dtype=np.float32),
            "confidence": np.zeros((0, 0, N_JOINTS), dtype=np.float32),
            "visibility_mask": np.zeros((0, 0, N_JOINTS), dtype=bool),
            "interpolation_mask": np.zeros((0, 0, N_JOINTS), dtype=bool),
            "target_3d_source": np.zeros((0, 0, N_JOINTS), dtype=object),
            "target_3d_interpolated_by_export": np.zeros((0, 0, N_JOINTS), dtype=bool),
            "subject_ids": np.array([], dtype=object),
            "task_names": np.array([], dtype=object),
            "native_fps": np.array([], dtype=np.float32),
            "joint_names": np.array(JOINT_NAMES, dtype=object),
        }

    return {
        "positions_2d": np.stack(all_2d, axis=0).astype(np.float32),
        "positions_3d": np.stack(all_3d, axis=0).astype(np.float32),
        "confidence": np.stack(all_conf, axis=0).astype(np.float32),
        "visibility_mask": np.stack(all_vis, axis=0).astype(bool),
        "interpolation_mask": np.stack(all_interp, axis=0).astype(bool),
        # c3d_pipeline.py's per-joint provenance (c3d/interpolated/
        # recovered_csv/invalid), resampled through Overlay — describes the
        # 3D TARGET's origin, independent of target_3d_interpolated_by_export
        # below (this export's OWN window-level gap-fill, a separate step).
        "target_3d_source": np.stack(all_3d_source, axis=0).astype(object),
        "target_3d_interpolated_by_export": np.stack(all_3d_interp, axis=0).astype(bool),
        "subject_ids": np.array(subject_ids, dtype=object),
        "task_names": np.array(task_names, dtype=object),
        # ORIGINAL camera fps this window's trial was recorded at — 60.0x for
        # most, ~30.0x for SUB02/SUB03 — kept per-window (not just per-trial)
        # so a reader filtering/reporting on windows never has to rejoin back
        # to metadata.json to know whether a window's 2D input was upsampled.
        "native_fps": np.array(native_fps_list, dtype=np.float32),
        "joint_names": np.array(JOINT_NAMES, dtype=object),
    }


def visualize_sample_window(npz_data: dict[str, np.ndarray], out_path: Path) -> bool:
    """Quick-look QA plot of one sample window. Returns False if there's nothing to plot."""
    if npz_data["positions_2d"].shape[0] == 0:
        return False

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SKELETON = [
        ("PELCEN", "L45"), ("L45", "T10"), ("T10", "C7"), ("C7", "HD"),
        ("PELCEN", "LHIP"), ("PELCEN", "RHIP"),
        ("LHIP", "LKNE"), ("LKNE", "LANK"), ("LANK", "LTOE"),
        ("RHIP", "RKNE"), ("RKNE", "RANK"), ("RANK", "RTOE"),
        ("C7", "LSHO"), ("LSHO", "LELB"), ("LELB", "LWR"), ("LWR", "LHAND"),
        ("C7", "RSHO"), ("RSHO", "RELB"), ("RELB", "RWR"), ("RWR", "RHAND"),
    ]
    idx = {n: i for i, n in enumerate(JOINT_NAMES)}

    p2d = npz_data["positions_2d"][0, 0]
    p3d = npz_data["positions_3d"][0, 0]

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    for a, b in SKELETON:
        ax1.plot([p2d[idx[a], 0], p2d[idx[b], 0]], [-p2d[idx[a], 1], -p2d[idx[b], 1]], "b-")
    ax1.scatter(p2d[:, 0], -p2d[:, 1], c="red")
    ax1.scatter([0], [0], c="black", marker="x", s=100, label="PELCEN (root)")
    ax1.set_title("2D (root-centered, scaled by image width/2)")
    ax1.legend()
    ax1.set_aspect("equal")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for a, b in SKELETON:
        ax2.plot([p3d[idx[a], 0], p3d[idx[b], 0]],
                  [p3d[idx[a], 1], p3d[idx[b], 1]],
                  [p3d[idx[a], 2], p3d[idx[b], 2]], "b-")
    ax2.scatter(p3d[:, 0], p3d[:, 1], p3d[:, 2], c="red")
    ax2.scatter([0], [0], [0], c="black", marker="x", s=100, label="PELCEN (root)")
    ax2.set_title("3D (root-centered, meters)")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


# ── Orchestration ─────────────────────────────────────────────────────────────────────

def run_export(
    outputs_dir: Path,
    task_names: list[str] | None = None,
    window_size: int = WINDOW_SIZE_DEFAULT,
    step_size: int = STEP_SIZE_DEFAULT,
    min_window_valid_fraction: float = MIN_WINDOW_VALID_FRACTION_DEFAULT,
    test_subjects: list[str] | None = None,
    val_subjects: list[str] | None = None,
    test_fraction: float = 0.2,
    val_fraction: float = 0.2,
    progress_cb: ProgressCb = None,
) -> dict:
    """Full pipeline: split subjects -> calibrate T10/L45 (train only) -> process
    every trial -> stack -> save train/test .npz to a timestamped export folder.

    Args:
        outputs_dir: the sync backend's outputs/ directory.
        task_names: subset of trial folder names to include, or None for all
            discovered trials.
        test_subjects: explicit SUBIDs for the test split, or None to auto-split
            by test_fraction (sorted order, last N subjects).

    Returns:
        Result dict: train/test window+trial counts, calibrated ratios, skipped
        trials with reasons, output paths.

    Raises:
        ValueError: if no trials, no usable training 3D data, or nothing to save.
    """
    def _cb(pct: int, step: str):
        if progress_cb:
            progress_cb(pct, step)

    all_trials = discover_trials(outputs_dir)
    if task_names is not None:
        wanted = set(task_names)
        trials = [(tid, d) for tid, d in all_trials if tid in wanted]
    else:
        trials = all_trials
    if not trials:
        raise ValueError("No trial folders found/selected.")

    # ── Synchronization export gate ───────────────────────────────────────────
    # A trial whose sync was never confirmed (needs_review missing/True) must
    # not reach training data — matching frame counts alone proves nothing
    # about whether the offset is actually correct (see docs/engineering-notes.md,
    # Synchronization). The ONLY way past this is a manual_review_approved:
    # true field already sitting in that specific trial's metadata.json,
    # set deliberately beforehand — there is no per-request bypass flag here,
    # so a bad-sync trial can't be waved through by accident on a whole-cohort
    # export run.
    _cb(1, "Checking sync review status…")
    gated_trials: list[tuple[str, Path]] = []
    sync_gate_skipped: list[dict] = []
    for task_id, trial_dir in trials:
        meta = read_metadata(outputs_dir, task_id) or {}
        blocked_reason = sync_gate_reason(meta)
        if blocked_reason is not None:
            sync_gate_skipped.append({"task_name": task_id, "reason": blocked_reason})
            continue
        gated_trials.append((task_id, trial_dir))
    trials = gated_trials
    if not trials:
        raise ValueError(
            f"All {len(sync_gate_skipped)} selected trial(s) are blocked by "
            f"the sync review gate — nothing left to export. See skipped_trials "
            f"in a smaller test run, or approve trials individually."
        )

    _cb(2, "Splitting subjects…")
    train_subjects, val_subjects_set, test_subjects_set = split_subjects(
        trials, test_subjects, val_subjects, test_fraction, val_fraction)

    _cb(5, "Calibrating T10/L45 spine ratio (train subjects only)…")
    train_3d_chunks = []
    for task_id, trial_dir in trials:
        if parse_subject_id(task_id) not in train_subjects:
            continue
        csv_path = trial_dir / Artifacts.SYNCED_60HZ
        if not csv_path.exists():
            continue
        try:
            train_3d_chunks.append(load_trial_3d_mm(csv_path))
        except ValueError:
            continue

    if not train_3d_chunks:
        raise ValueError("No usable training 3D data found — cannot calibrate T10/L45 ratio.")

    train_3d_all = np.concatenate(train_3d_chunks, axis=0)
    t_t10 = fit_spine_ratio(train_3d_all, T10_IDX)
    t_l45 = fit_spine_ratio(train_3d_all, L45_IDX)

    processed: list[ProcessedTrial] = []
    skipped: list[dict] = list(sync_gate_skipped)
    n = len(trials)
    for i, (task_id, trial_dir) in enumerate(trials):
        _cb(10 + int(80 * i / n), f"Processing {task_id} ({i + 1}/{n})…")
        # Re-read metadata here rather than threading it from the earlier sync-
        # gate loop: keeps `trials`/`gated_trials` as plain (task_id, Path)
        # tuples throughout (no signature changes to split_subjects() or the
        # T10/L45 calibration loop above), at the cost of one cheap re-read of
        # a small JSON file per trial.
        trial_meta = read_metadata(outputs_dir, task_id) or {}
        result, reason = process_trial(
            task_id, trial_dir, trial_meta, t_t10, t_l45,
            window_size, step_size, min_window_valid_fraction,
        )
        if result is not None:
            processed.append(result)
        else:
            skipped.append({"task_name": task_id, "reason": reason})
            warnings.warn(f"Skipping {task_id}: {reason}")

    if not processed:
        raise ValueError("No usable trials after windowing — nothing to save.")

    train_trials = [p for p in processed if p.subject_id in train_subjects]
    val_trials = [p for p in processed if p.subject_id in val_subjects_set]
    test_trials = [p for p in processed if p.subject_id in test_subjects_set]

    _cb(92, "Stacking + saving .npz…")
    train_data = stack_windows(train_trials)
    val_data = stack_windows(val_trials)
    test_data = stack_windows(test_trials)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = outputs_dir / "_dataset_exports" / timestamp
    export_dir.mkdir(parents=True, exist_ok=True)

    train_path = export_dir / "construction_vicon_train.npz"
    val_path = export_dir / "construction_vicon_val.npz"
    test_path = export_dir / "construction_vicon_test.npz"
    np.savez_compressed(train_path, **train_data)
    np.savez_compressed(val_path, **val_data)
    np.savez_compressed(test_path, **test_data)

    _cb(98, "Generating QA plot…")
    viz_path = export_dir / "visualize_check.png"
    has_viz = visualize_sample_window(train_data, viz_path)

    _cb(100, "Done")
    return {
        "export_dir": str(export_dir),
        "timestamp": timestamp,
        "train_npz": str(train_path),
        "val_npz": str(val_path),
        "test_npz": str(test_path),
        "visualize_png": str(viz_path) if has_viz else None,
        "calibrated_t_t10": round(t_t10, 4),
        "calibrated_t_l45": round(t_l45, 4),
        "train_subjects": sorted(train_subjects),
        "val_subjects": sorted(val_subjects_set),
        "test_subjects": sorted(test_subjects_set),
        "n_train_trials": len(train_trials),
        "n_val_trials": len(val_trials),
        "n_test_trials": len(test_trials),
        "n_train_windows": int(train_data["positions_3d"].shape[0]),
        "n_val_windows": int(val_data["positions_3d"].shape[0]),
        "n_test_windows": int(test_data["positions_3d"].shape[0]),
        "window_size": window_size,
        "step_size": step_size,
        "skipped_trials": skipped,
        "joint_names": JOINT_NAMES,
    }
