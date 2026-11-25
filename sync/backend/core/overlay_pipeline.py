"""
overlay_pipeline.py
===================
Overlay the Vicon 3D skeleton onto an RGB video by projecting joint positions
through calibrated camera intrinsics and extrinsics.

Importable module with progress callback support.

Per-frame algorithm:
  1. Compute the Vicon time for this video frame using sync_metadata.json.
  2. Linearly interpolate all joint 3D positions at that Vicon time.
  3. Project to 2D pixels via cv2.projectPoints (K, dist, rvec, tvec).
  4. Draw colour-coded skeleton bones and joint markers.
  5. Render HUD: video frame, Vicon frame, timestamp, per-joint visibility.
"""

from pathlib import Path
import json
import csv
import logging
import subprocess
from typing import Optional, Callable

import cv2
import numpy as np

from core.vicon_io import load_intrinsics, rotate_frame, read_vicon_trajectories
from core.skeleton import draw_skeleton, draw_hud
from core.extrinsic_correction import compose_delta, is_zero_delta
from core.overlay_trim import find_trim_start, find_trim_end, write_synced_csv, write_synced_csv_2d
from core.face_blur import build_face_tracks, apply as apply_face_blur
from utils.task_paths import Artifacts

logger = logging.getLogger(__name__)


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_extrinsics(json_path: Path, delta: Optional[dict] = None):
    """
    Load rvec and tvec from extrinsics.json.

    Args:
        delta: optional not-yet-saved manual spatial correction (see
            core/extrinsic_correction.py) — used for live preview before the
            user commits it via the "save spatial correction" endpoint, which
            instead writes the composed values back into the file itself.

    Returns:
        (rvec, tvec) as (3,1) numpy float64 arrays.
    """
    with open(json_path, encoding="utf-8") as f:
        d = json.load(f)
    rvec = np.asarray(d["rvec"],   dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(d["tvec_m"], dtype=np.float64).reshape(3, 1)
    if not is_zero_delta(delta):
        rvec, tvec = compose_delta(rvec, tvec, delta)
    return rvec, tvec


# ── Sync mapping ───────────────────────────────────────────────────────────────

def _video_to_vicon_time_s(video_frame: int, sync: dict) -> float:
    """
    Return the Vicon timestamp (seconds from XCP START_TIME) for a video frame.

    Formula:
      t_vicon_s = (V + manual_frame_offset) / video_fps + sync_offset_s

    Using seconds (not frame numbers) eliminates Vicon's 1-indexed frame
    numbering ambiguity and gives millisecond-accurate interpolation.
    """
    V = video_frame + sync["manual_frame_offset"]
    return V / sync["video_fps"] + sync["sync_offset_s"]


def _interp_joints_by_time(joints: dict, times: np.ndarray,
                            t_s: float) -> dict:
    """
    Linearly interpolate all joints at a given Vicon timestamp (seconds).

    Args:
        joints: {label: (N, 3) array in metres}
        times:  (N,) float64 — Vicon timestamps in seconds from capture start.
        t_s:    Query time in seconds.

    Returns:
        {label: (3,) array in metres} — NaN for missing/out-of-range.
    """
    nan3 = np.full(3, np.nan)
    idx  = int(np.searchsorted(times, t_s))
    result = {}
    for label, data in joints.items():
        if idx <= 0 or idx >= len(times):
            # t_s is outside [times[0], times[-1]] — there is no bracketing
            # pair to interpolate between. The previous behaviour held the
            # nearest boundary sample's value constant here, which is
            # zero-order-hold EXTRAPOLATION, not interpolation: a video frame
            # a few ms before Vicon locked on (or after it dropped) would get
            # a fabricated "static" position instead of being correctly
            # marked as having no Vicon data at that instant. Always NaN.
            p = nan3
        else:
            t0    = float(times[idx - 1])
            t1    = float(times[idx])
            alpha = (t_s - t0) / (t1 - t0) if t1 != t0 else 0.0
            p0, p1 = data[idx - 1], data[idx]
            p = ((1.0 - alpha) * p0 + alpha * p1
                 if not (np.any(np.isnan(p0)) or np.any(np.isnan(p1)))
                 else nan3)
        result[label] = p
    return result


_SOURCE_TRUST = {"c3d": 3, "recovered_csv": 2, "interpolated": 1, "invalid": 0}


def _worse_source(a: str, b: str) -> str:
    """Return whichever of two 4-state provenance labels is LESS trustworthy."""
    return a if _SOURCE_TRUST.get(a, 0) <= _SOURCE_TRUST.get(b, 0) else b


def _resample_source_by_time(source: dict, times: np.ndarray, t_s: float) -> dict:
    """Resample c3d_pipeline.py's {joint}_Source provenance ("c3d" /
    "interpolated" / "recovered_csv" / "invalid") to a query timestamp,
    mirroring _interp_joints_by_time()'s bracketing logic.

    The 3D POSITION at t_s is a linear blend of the two bracketing native-rate
    samples, so its true provenance is whichever of the two is the LESS
    trustworthy of the pair — reporting the better one would overstate
    confidence in a value that partially depends on the worse sample too.
    Out-of-range queries are "invalid", matching _interp_joints_by_time()'s
    NaN-for-out-of-range rule (no bracketing pair exists to blend at all).
    """
    idx = int(np.searchsorted(times, t_s))
    result = {}
    for label, src in source.items():
        if idx <= 0 or idx >= len(times):
            result[label] = "invalid"
        else:
            result[label] = _worse_source(str(src[idx - 1]), str(src[idx]))
    return result


def _first_last_fully_valid_time(joints: dict, times: np.ndarray) -> tuple[float, float]:
    """First/last timestamp where EVERY loaded joint has a finite position.

    `times[0]`/`times[-1]` alone are NOT a safe trim boundary: c3d_pipeline.py
    no longer truncates its output CSV (a deliberate change so one
    badly-occluded joint can't discard otherwise-good data for every other
    joint), so the CSV's absolute first/last row can be a frame recorded
    before Vicon locked onto the subject — or after it lost them — with some
    or all joints still NaN. Trimming Overlay's output to the raw recording
    bounds instead of the fully-tracked window lets those frames leak
    straight into rgb.mp4/synced_60hz.csv/frame_mapping.csv.

    Args:
        joints: {label: (N, 3) array}, NaN where occluded.
        times:  (N,) Vicon timestamps aligned with `joints`.

    Returns:
        (t_first, t_last) — the fully-valid window, same units as `times`.

    Raises:
        ValueError: if no frame has every joint valid.
    """
    if not joints:
        return float(times[0]), float(times[-1])
    fully_valid = np.ones(len(times), dtype=bool)
    for arr in joints.values():
        fully_valid &= np.isfinite(arr).all(axis=1)
    if not fully_valid.any():
        raise ValueError(
            "No frame has every one of the 21 joints valid — cannot "
            "determine a usable start/end for the Overlay trim window. "
            "Check the trial's Vicon data (missing markers, occlusion) "
            "before re-running Overlay."
        )
    idx = np.flatnonzero(fully_valid)
    return float(times[idx[0]]), float(times[idx[-1]])


def _project_joints(joints_3d: dict, rvec, tvec, K, dist, *,
                    undistorted: bool = False) -> dict:
    """
    Project 3D joint positions (metres, Vicon world) to 2D image coordinates.

    Args:
        undistorted: if True, project with ZERO distortion (ideal pinhole with
            K), giving coordinates in the lens-UNDISTORTED image space. Use this
            for anything that must line up with the undistorted rgb.mp4 (e.g.
            synced2d, which drives Vicon-guided subject selection on rgb.mp4).
            The default (False, full `dist`) is for drawing on the DISTORTED
            overlay.mp4 frame. Mixing the two spaces mis-registers points by
            several pixels.

    Returns:
        {label: (2,) float pixel coords} — NaN if the 3D point was NaN.
    """
    proj_dist = np.zeros((1, 5), dtype=np.float64) if undistorted else dist
    result = {}
    for label, p3 in joints_3d.items():
        if np.any(np.isnan(p3)):
            result[label] = np.full(2, np.nan)
            continue
        pts, _ = cv2.projectPoints(p3.reshape(1, 1, 3), rvec, tvec, K, proj_dist)
        result[label] = pts.reshape(2)
    return result


# ── FFmpeg trim helper ────────────────────────────────────────────────────────

# ── Output video normalisation ────────────────────────────────────────────────
# Every video leaving this pipeline is 1920x1080. Source footage is mixed: SUB01
# was shot 2160x3840 portrait (-> 3840x2160 after ccw90), i.e. 4K, while others
# are 1080p. Without this, the dataset mixes resolutions.
NORMALIZED_WIDTH  = 1920
NORMALIZED_HEIGHT = 1080

# Max tolerated jitter in frame intervals before a source is called
# variable-frame-rate. A CFR file's intervals are near-identical; SUB01 measures
# std = 0.083 ms over 400 frames.
CFR_JITTER_TOL_MS = 1.0

# NOTE ON FRAME RATE — deliberately NOT re-timed to exactly 60.000.
# The cameras run at ~60.028 fps CFR (measured: mean interval 16.6583 ms,
# std 0.083 ms, confirmed by nb_frames/duration). Forcing exactly 60.000 would
# mean either relabelling the frames — which stretches the timeline and
# re-introduces the +6.4 ms/trial drift that preferring avg_frame_rate over
# r_frame_rate just removed (see sync_io.get_video_fps_ffprobe) — or temporally
# resampling, which drops/duplicates frames and corrupts pose. Neither buys
# anything: RTMPose and the 3D lifter are frame-indexed and fps-agnostic, and
# 60.028 is "60 fps" to any reported precision. The real hazard is a genuinely
# VARIABLE frame rate, which no constant fps can represent — that is what
# assert_constant_frame_rate() rejects. Resolution is normalised; true timing is
# preserved and carried through as the measured fps.


def assert_constant_frame_rate(video_path: Path,
                               tol_ms: float = CFR_JITTER_TOL_MS) -> float:
    """Verify a video is constant-frame-rate and return its measured fps.

    A variable-frame-rate source cannot be represented by any single fps, so
    every `frame_index / fps` timestamp downstream would be wrong by a drifting,
    unknowable amount — silently mispairing Vicon with video. iPhone footage is
    frequently VFR, so this is checked rather than assumed.

    Reads PACKET timestamps and SORTS them before differencing. All three
    details are load-bearing:
      - `packet=pts_time` reads container headers only; `frame=...` decodes every
        frame. Identical numbers, but 0.3 s vs 71 s on a 954-frame 4K clip —
        71 s x 240 trials would be ~5 h of pure overhead.
      - ffprobe emits packets in DECODE order, and with H.264 B-frames decode
        order != presentation order, so raw diffs go negative/huge and a CFR file
        looks wildly variable. Sorting restores presentation order.
      - Read ALL packets. Truncating to the first N (`-read_intervals`) cuts in
        decode order, dropping frames whose PTS lies inside the sampled range and
        fabricating gaps: on SUB01/lifting/video.MOV that reported std 2.043 ms
        with a bogus 66.67 ms interval, where the full sorted read gives
        std 0.081 ms, max 16.67 ms, and zero intervals over 20 ms. The truncated
        version would have rejected every valid video in the dataset.

    Args:
        video_path: video to inspect.
        tol_ms:     max std-dev of frame intervals still considered constant.

    Returns:
        Measured fps (1000 / mean interval).

    Raises:
        ValueError: if the intervals jitter beyond tol_ms (VFR), or if PTS
            cannot be read.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "packet=pts_time",
             "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"Cannot read frame timestamps from {video_path}: {exc}")

    ts = []
    for line in out.splitlines():
        tok = line.strip().rstrip(",")
        if tok and tok != "N/A":
            try:
                ts.append(float(tok))
            except ValueError:
                pass
    if len(ts) < 10:
        raise ValueError(f"Too few frame timestamps read from {video_path} "
                         f"({len(ts)}) — cannot verify constant frame rate.")

    d = np.diff(np.sort(np.asarray(ts))) * 1000.0   # sort: decode -> display order
    if d.std() > tol_ms:
        raise ValueError(
            f"{video_path.name} is VARIABLE frame rate (interval std "
            f"{d.std():.3f} ms > {tol_ms} ms, range {d.min():.2f}-{d.max():.2f} "
            f"ms). No single fps describes it, so every frame_index/fps "
            f"timestamp downstream would drift by an unknown amount. Re-encode "
            f"to CFR before processing."
        )
    return float(1000.0 / d.mean())


def _verify_normalized(path: Path) -> None:
    """Fail loudly if an output video is not exactly NORMALIZED_WIDTH x HEIGHT."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {path} to verify normalisation.")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if (w, h) != (NORMALIZED_WIDTH, NORMALIZED_HEIGHT):
        raise ValueError(
            f"{path.name} is {w}x{h}, expected "
            f"{NORMALIZED_WIDTH}x{NORMALIZED_HEIGHT}. The normalisation step "
            f"did not take effect — do NOT feed this to the 2D detector."
        )


def _ffmpeg_trim(src: Path, dst: Path, t_start: float, t_end: float,
                 normalize: bool = True,
                 expected_frames: Optional[int] = None) -> Path:
    """Trim + re-encode src to dst with libx264, normalised to 1920x1080.

    `scale=1920:-2` preserves aspect (the -2 keeps height even for libx264);
    _verify_normalized() then hard-fails if the result is not exactly 1080p,
    which catches a non-16:9 source rather than silently distorting it. A source
    already at 1920x1080 scales to itself, so this is a no-op there.

    Frame timing is deliberately untouched — no `-r` — so the source's true CFR
    rate carries through. See the frame-rate note above.

    `-ss`/`-to` alone is a TIME window, not a frame count: ffmpeg re-derives
    how many frames fall inside it from the container's own PTS timing, which
    can differ from `t_start`/`t_end` by up to one frame whenever the caller's
    `video_fps` (from `cv2.CAP_PROP_FPS`) doesn't exactly match that internal
    timing. That silent drift is exactly what let `rgb.mp4`'s encoded frame
    count disagree with `synced_60hz.csv`'s row count — both computed from the
    same `trim_start`/`trim_end`, but one re-derived by ffmpeg from time,
    the other counted as an exact integer. When `expected_frames` is given
    (the caller already knows the exact intended count), `-frames:v` pins the
    output to that many frames outright, and the count is verified after
    encoding rather than assumed — see rtmpose_pipeline.py's frame-count guard
    for why row-for-row alignment with the CSV must hold exactly.

    Returns dst on success, a renamed mp4v file if ffmpeg is absent, or src if
    encoding fails.
    """
    cmd = ["ffmpeg", "-y", "-i", str(src),
           "-ss", f"{t_start:.6f}", "-to", f"{t_end:.6f}"]
    if normalize:
        cmd += ["-vf", f"scale={NORMALIZED_WIDTH}:-2"]
    if expected_frames is not None:
        cmd += ["-frames:v", str(expected_frames)]
    cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "fast", str(dst)]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        src.unlink(missing_ok=True)
        if normalize:
            _verify_normalized(dst)
        if expected_frames is not None:
            cap = cv2.VideoCapture(str(dst))
            actual = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if actual != expected_frames:
                raise RuntimeError(
                    f"{dst.name}: ffmpeg encoded {actual} frames, expected "
                    f"exactly {expected_frames} to match the synced CSV row "
                    f"count. Trim window: {t_start:.6f}s-{t_end:.6f}s.")
        return dst
    except FileNotFoundError:
        # No ffmpeg: the mp4v fallback is NOT normalised. Say so loudly rather
        # than letting a 4K file reach the detector labelled as if it were 1080p.
        fallback = dst.with_name(dst.stem + "_mp4v.mp4")
        src.rename(fallback)
        logger.error(
            "ffmpeg not found — %s is NOT normalised to %dx%d. Install ffmpeg "
            "and re-run before any pose-model processing.",
            fallback.name, NORMALIZED_WIDTH, NORMALIZED_HEIGHT)
        return fallback
    except subprocess.CalledProcessError:
        return src


# ── Single-frame preview ──────────────────────────────────────────────────────

def preview_single_frame(
    video_path: Path,
    vicon_csv: Path,
    intrinsics_path: Path,
    extrinsics_path: Path,
    sync: dict,
    video_frame: int,
    manual_frame_offset_override: Optional[int] = None,
    rotate: Optional[str] = None,
    extrinsic_delta: Optional[dict] = None,
) -> dict:
    """
    Render one video frame with Vicon skeleton overlay for live preview.

    Much faster than run_overlay — reads a single frame, does not write any
    files.  Used by the fine-tune UI so the user can adjust manual_frame_offset
    before committing to a full render.

    Args:
        video_path:    Source video (MOV or MP4).
        vicon_csv:     Cleaned Vicon joint CSV.
        intrinsics_path: Path to intrinsic.json.
        extrinsics_path: Path to extrinsics.json.
        sync:          Sync metadata dict from <task>_sync.json.
        video_frame:   0-based frame index to render.
        manual_frame_offset_override:
                       Overrides sync["manual_frame_offset"] without writing
                       to disk.  Pass None to use the saved value.
        rotate:        Override video rotation; None = use sync["video_rotation"].
        extrinsic_delta:
                       Optional not-yet-saved manual spatial correction (see
                       core/extrinsic_correction.py), applied for this preview
                       only — "save spatial correction" persists it instead.

    Returns:
        {
          "jpeg":         bytes  — JPEG-encoded image,
          "in_range":     bool   — frame maps to valid Vicon data,
          "vicon_time_s": float  — Vicon timestamp computed for this frame,
          "total_frames": int    — total frames in the video,
          "video_fps":    float,
        }
    """
    # Override offset without mutating the caller's dict
    if manual_frame_offset_override is not None:
        sync = {**sync, "manual_frame_offset": int(manual_frame_offset_override)}

    rotate_val = rotate if rotate else sync.get("video_rotation", "none")

    K, dist, _, _ = load_intrinsics(intrinsics_path)
    rvec, tvec    = _load_extrinsics(extrinsics_path, extrinsic_delta)

    vicon_d      = read_vicon_trajectories(vicon_csv)
    vicon_times  = vicon_d["times"]
    joints       = vicon_d["data"]
    label_list   = vicon_d["labels"]
    try:
        t_s_min, t_s_max = _first_last_fully_valid_time(joints, vicon_times)
    except ValueError:
        # Preview is a UI aid, not the final export — degrade to the raw
        # range rather than breaking offset fine-tuning entirely for a
        # trial with no fully-tracked frame at all.
        t_s_min, t_s_max = float(vicon_times[0]), float(vicon_times[-1])
    total_joints = len(label_list)

    cap          = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = float(cap.get(cv2.CAP_PROP_FPS)) or float(sync.get("video_fps", 60.0))
    vicon_fps    = float(sync.get("vicon_fps", 100.0))

    # Clamp to valid range
    video_frame = max(0, min(int(video_frame), max(total_frames - 1, 0)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {video_frame} from video.")

    frame    = rotate_frame(frame, rotate_val)
    t_s      = _video_to_vicon_time_s(video_frame, sync)
    in_range = t_s_min <= t_s <= t_s_max
    n_vis    = 0

    if in_range:
        joints_3d = _interp_joints_by_time(joints, vicon_times, t_s)
        joints_2d = _project_joints(joints_3d, rvec, tvec, K, dist)
        n_vis     = draw_skeleton(frame, joints_2d)

    draw_hud(frame, video_frame, t_s, n_vis, total_joints, video_fps, vicon_fps, in_range)

    ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok2:
        raise RuntimeError("JPEG encoding failed.")

    return {
        "jpeg":         buf.tobytes(),
        "in_range":     in_range,
        "vicon_time_s": float(t_s),
        "total_frames": total_frames,
        "video_fps":    float(video_fps),
    }


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_overlay(
    task_name: str,
    video_path: Path,
    vicon_csv: Path,
    intrinsics_path: Path,
    extrinsics_path: Path,
    sync_path: Path,
    output_dir: Path,
    rotate: Optional[str] = None,
    export_plain: bool = True,
    blur_faces_rgb: bool = False,
    export_2d_csv: bool = True,
    progress_cb: Optional[Callable] = None,
) -> dict:
    """
    Run the full Vicon skeleton overlay pipeline.

    Produces two output videos when export_plain=True:
      - <task>_overlay.mp4  — skeleton projected onto each frame. Never blurred.
      - <task>_rgb.mp4      — same trim window, undistorted, no skeleton
                              (clean RGB for model training). Face-blurred
                              when blur_faces_rgb=True (default False).

    When export_2d_csv=True (default), also writes synced_vicon2d_60hz.csv (Vicon-PROJECTED, not RTMPose) — the
    same per-frame joints as <task>_synced.csv, but as the camera-projected 2D
    pixel coordinates already computed each frame to draw the skeleton (no
    extra projection work, just persisting what run_overlay already calculates).

    Face blurring uses a two-pass, jitter-stable pipeline (core/face_blur.py):
    pass 1 detects+tracks faces across the WHOLE video first, interpolates
    short detection gaps, and smooths box coordinates; pass 2 is the
    per-frame render loop below, which just blurs using the precomputed
    boxes — no detection happens in that loop.

    Args:
        task_name:       composite task id used for all output filename
                         prefixes, e.g. "SUB01_lifting_T01". MUST be passed
                         explicitly — it used to be (wrongly) derived from
                         video_path.stem, which is always literally "video"
                         since sync.py saves every upload as video.<ext>
                         regardless of task. That bug produced video_*.mp4/csv
                         output files no task-aware code could find again.
        video_path:      Path to the RGB task video (MOV or MP4).
        vicon_csv:       Path to the cleaned Vicon joint coordinates CSV.
        intrinsics_path: Path to intrinsic.json.
        extrinsics_path: Path to extrinsics.json.
        sync_path:       Path to sync metadata JSON.
        output_dir:      Output directory for overlay video and CSVs.
        rotate:          Override video rotation ('ccw90', 'cw90', '180', 'none').
                         Default: read from sync metadata.
        export_plain:    If True, also write a lens-undistorted RGB video (no
                         skeleton) in the same trim window.  Default True.
        blur_faces_rgb:  If True, blur every detected face (every person, not
                         just the tracked subject) in the _rgb.mp4 output.
                         Has no effect if export_plain=False. Default False.
        export_2d_csv:   If True, also write synced_vicon2d_60hz.csv (Vicon-PROJECTED, not RTMPose) (camera-projected
                         2D pixel joints, same frames/order as synced_csv). Default True.
        progress_cb:     Optional callback(pct: int, step: str).

    Returns:
        dict with keys: overlay_video, plain_video (or None), frame_mapping_csv,
                        synced_csv, synced2d_csv (or None), total_video_frames,
                        trimmed_frames, trimmed_duration_s, full_skeleton_pct,
                        output_dir, task_name.
    """
    def _cb(pct, step):
        if progress_cb:
            progress_cb(pct, step)

    _cb(5, "Loading calibration data…")
    K, dist, iw, ih = load_intrinsics(intrinsics_path)
    rvec, tvec = _load_extrinsics(extrinsics_path)

    with open(sync_path, encoding="utf-8") as f:
        sync = json.load(f)

    rotate    = rotate if rotate else sync.get("video_rotation", "none")
    vicon_fps = float(sync["vicon_fps"])

    _cb(10, "Loading Vicon data…")
    vicon_d             = read_vicon_trajectories(vicon_csv)
    vicon_times         = vicon_d["times"]       # seconds from XCP START_TIME
    has_absolute_times  = vicon_d["has_absolute_times"]
    joints              = vicon_d["data"]
    label_list          = vicon_d["labels"]
    # c3d_pipeline.py's 4-state provenance (c3d/interpolated/recovered_csv/
    # invalid) — {} for a legacy clean CSV written before that column existed.
    joint_source        = vicon_d["source"]
    # First/last frame with ALL joints valid — NOT vicon_times[0]/[-1], which
    # since c3d_pipeline.py stopped truncating its output can be a frame
    # before Vicon locked on (or after it dropped) with joints still NaN.
    # This is the trim window ultimately baked into rgb.mp4/synced_60hz.csv.
    t_s_min, t_s_max    = _first_last_fully_valid_time(joints, vicon_times)
    total_joints        = len(label_list)

    # Reject variable-frame-rate footage BEFORE any processing. Every downstream
    # timestamp is frame_index / fps, which only means anything if the source is
    # constant-rate. iPhone video is often VFR, so this is verified, not assumed.
    _cb(12, "Verifying constant frame rate…")
    assert_constant_frame_rate(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, probe = cap.read()
    if not ok:
        raise RuntimeError("Cannot read first frame from video.")
    probe = rotate_frame(probe, rotate)
    H, W  = probe.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    face_tracks: dict = {}
    if blur_faces_rgb:
        _cb(11, "Detecting faces for blurring (pass 1/2)…")
        face_tracks = build_face_tracks(
            video_path, rotate_fn=lambda f: rotate_frame(f, rotate),
            progress_cb=lambda pct, step: _cb(11 + int(pct / 100), step),
        )

    _cb(12, "Rendering frames…")
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_mp4   = output_dir / "_temp_overlay.mp4"
    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    writer     = cv2.VideoWriter(str(temp_mp4), fourcc, video_fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create temp video writer in {output_dir}")

    # Plain RGB writer (undistorted, no skeleton) — same dimensions post-rotation
    temp_plain_mp4 = output_dir / "_temp_plain.mp4" if export_plain else None
    writer_plain   = None
    if export_plain:
        writer_plain = cv2.VideoWriter(str(temp_plain_mp4), fourcc, video_fps, (W, H))
        if not writer_plain.isOpened():
            writer_plain = None   # fail gracefully — overlay still proceeds

    # frame_stats: (video_frame, vicon_time_s, in_range, n_visible)
    frame_stats = []
    n_in_range  = 0

    for V in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frame = rotate_frame(frame, rotate)

        # ── Plain RGB frame: undistort BEFORE any annotation ─────────────────
        # Captured here so it shares the same rotation and trim logic as the
        # overlay video.  cv2.undistort keeps the same (W, H) dimensions.
        if writer_plain is not None:
            plain_frame = cv2.undistort(frame, K, dist)
            if blur_faces_rgb:
                plain_frame = apply_face_blur(plain_frame, V, face_tracks)

        # Time-based mapping: avoids Vicon frame-number indexing ambiguity
        t_s      = _video_to_vicon_time_s(V, sync)
        in_range = t_s_min <= t_s <= t_s_max
        n_vis    = 0

        if in_range:
            n_in_range += 1
            joints_3d = _interp_joints_by_time(joints, vicon_times, t_s)
            joints_2d = _project_joints(joints_3d, rvec, tvec, K, dist)
            n_vis     = draw_skeleton(frame, joints_2d)
        else:
            joints_2d = {}

        draw_hud(frame, V, t_s, n_vis, total_joints, video_fps, vicon_fps, in_range)
        writer.write(frame)

        if writer_plain is not None:
            writer_plain.write(plain_frame)

        frame_stats.append((V, t_s, in_range, n_vis))

        if V % 30 == 0 and total_frames > 0:
            pct = 12 + int(73 * V / total_frames)
            _cb(pct, f"Rendering frame {V}/{total_frames}…")

    cap.release()
    writer.release()
    if writer_plain is not None:
        writer_plain.release()

    _cb(85, "Computing trim points…")

    # ── Time-based trimming ────────────────────────────────────────────────────
    # When time_s column is present the trim window is derived directly from the
    # Vicon data boundaries so both streams start and end at the same moment.
    #
    #   video_frame = (vicon_time_s - sync_offset_s) * video_fps - manual_offset
    #
    # When time_s is absent (legacy CSVs), fall back to visibility-based trim.
    sync_offset_s    = sync["sync_offset_s"]
    manual_offset    = sync.get("manual_frame_offset", 0)

    if has_absolute_times:
        # Exact frame corresponding to first/last valid Vicon time
        trim_start = max(0, round(
            (t_s_min - sync_offset_s) * video_fps) - manual_offset)
        trim_end   = min(total_frames - 1, round(
            (t_s_max - sync_offset_s) * video_fps) - manual_offset)
        # Clamp to frames that are actually in-range (safety net)
        while trim_start < total_frames and not frame_stats[trim_start][2]:
            trim_start += 1
        while trim_end > 0 and not frame_stats[trim_end][2]:
            trim_end -= 1
    else:
        # Legacy fallback: trim to first/last frame where all joints visible
        trim_start = find_trim_start(frame_stats, total_joints)
        trim_end   = find_trim_end(frame_stats, total_joints)

    trim_start_sec = trim_start / video_fps
    trim_end_sec   = (trim_end + 1) / video_fps
    trimmed_stats  = frame_stats[trim_start : trim_end + 1]
    n_trimmed      = len(trimmed_stats)

    _cb(85, "Encoding overlay H.264…")

    qa_out = output_dir / Artifacts.QA_DIR
    qa_out.mkdir(parents=True, exist_ok=True)
    out_h264      = qa_out / Artifacts.QA_OVERLAY
    overlay_video = _ffmpeg_trim(temp_mp4, out_h264, trim_start_sec, trim_end_sec,
                                 expected_frames=n_trimmed)

    plain_video: Optional[Path] = None
    if temp_plain_mp4 and temp_plain_mp4.exists():
        _cb(91, "Encoding plain RGB H.264…")
        out_plain = output_dir / Artifacts.RGB_VIDEO
        # expected_frames pins rgb.mp4 to EXACTLY n_trimmed frames so it can
        # never disagree with synced_60hz.csv's row count — see _ffmpeg_trim's
        # docstring for why time-based -ss/-to trimming alone isn't enough.
        plain_video = _ffmpeg_trim(temp_plain_mp4, out_plain, trim_start_sec, trim_end_sec,
                                   expected_frames=n_trimmed)

    _cb(94, "Writing frame mapping…")
    mapping_path = output_dir / Artifacts.FRAME_MAPPING
    with mapping_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "synced_video_frame", "original_video_frame", "vicon_time_s",
            "in_range", "n_visible_joints",
            # Extended columns:
            "video_timestamp_s", "sync_offset_s",
            "rgb_valid", "vicon_valid", "pair_valid", "exclusion_reason",
        ])
        for synced_idx, (V, t_s, ir, nv) in enumerate(trimmed_stats):
            # video_timestamp_s: seconds from THIS synced/trimmed video's own
            # frame 0 — not the raw pre-trim video, and not Vicon time.
            video_timestamp_s = synced_idx / video_fps
            # rgb_valid: True by construction — synced_idx only ranges over
            # frames that ffmpeg actually encoded into rgb.mp4 above. This
            # stage runs before 2D Pose, so RTMPose confidence/detection
            # validity isn't known yet; that lives in pose2d_rtmpose.csv.
            rgb_valid = True
            # vicon_valid: the existing in-range flag — was a Vicon timestamp
            # within the trial's valid capture window available to interpolate
            # at this instant (see sync_io.read_first_valid_frame/
            # read_last_valid_frame, which now require ALL joints valid, not
            # just some).
            vicon_valid = bool(ir)
            pair_valid = rgb_valid and vicon_valid
            exclusion_reason = "" if pair_valid else "vicon_out_of_range"
            w.writerow([
                synced_idx, V, f"{t_s:.4f}", int(ir), nv,
                f"{video_timestamp_s:.6f}", f"{sync_offset_s:.6f}",
                int(rgb_valid), int(vicon_valid), int(pair_valid), exclusion_reason,
            ])

    _cb(97, "Writing synced CSV…")
    synced_csv = output_dir / Artifacts.SYNCED_60HZ
    # Pass closures so write_synced_csv doesn't need to know about vicon_times
    def _interp_at(t: float) -> dict:
        return _interp_joints_by_time(joints, vicon_times, t)
    source_fn = (
        (lambda t: _resample_source_by_time(joint_source, vicon_times, t))
        if joint_source else None
    )
    write_synced_csv(label_list, trimmed_stats, synced_csv, _interp_at, source_fn=source_fn)

    synced2d_csv: Optional[Path] = None
    if export_2d_csv:
        _cb(98, "Writing 2D projected CSV…")
        synced2d_csv = output_dir / Artifacts.SYNCED2D_60HZ
        # Joints are projected with the intrinsics calibrated at the video's
        # NATIVE resolution, but the encoded outputs are normalised to
        # NORMALIZED_WIDTH x NORMALIZED_HEIGHT. Scale the projected pixels by the
        # same factor so synced_vicon2d_60hz.csv lands in the coordinate space of the
        # <task>_rgb.mp4 it is meant to describe. Without this the CSV would be
        # in 4K pixels while the video it annotates is 1080p — a silent 2x
        # mismatch for anything comparing them.
        _sx = NORMALIZED_WIDTH / float(W)
        _sy = NORMALIZED_HEIGHT / float(H)

        def _interp_2d_at(t: float) -> dict:
            # undistorted=True: synced2d must match the lens-undistorted rgb.mp4
            # (it drives Vicon-guided subject selection there), NOT the distorted
            # overlay.mp4. See _project_joints docstring.
            pts = _project_joints(_interp_at(t), rvec, tvec, K, dist, undistorted=True)
            return {k: (v[0] * _sx, v[1] * _sy) for k, v in pts.items()}
        write_synced_csv_2d(label_list, trimmed_stats, synced2d_csv, _interp_2d_at)

    n_full_vis = sum(1 for _, _, ir, nv in trimmed_stats
                     if ir and nv >= total_joints)
    full_skeleton_pct = 100.0 * n_full_vis / max(1, n_trimmed)

    _cb(100, "Complete")

    return {
        "task_name":          task_name,
        "overlay_video":      str(overlay_video),
        "plain_video":        str(plain_video) if plain_video else None,
        "frame_mapping_csv":  str(mapping_path),
        "synced_csv":         str(synced_csv),
        "synced2d_csv":       str(synced2d_csv) if synced2d_csv else None,
        "total_video_frames": total_frames,
        "trimmed_frames":     n_trimmed,
        "trimmed_duration_s": round(n_trimmed / video_fps, 3),
        "full_skeleton_pct":  round(full_skeleton_pct, 1),
        "output_dir":         str(output_dir),
        "video_fps":          video_fps,
        "vicon_fps":          vicon_fps,
    }


def generate_plain_rgb(
    task_name: str,
    video_path: Path,
    vicon_csv: Path,
    intrinsics_path: Path,
    sync_path: Path,
    output_dir: Path,
    rotate: Optional[str] = None,
    progress_cb: Optional[Callable] = None,
) -> Path:
    """Generate ONLY the undistorted plain RGB video (no skeleton).

    For a trial whose Overlay already ran successfully with the "export
    plain RGB" option off — re-running the full Overlay would needlessly
    redo Vicon interpolation, skeleton projection, and HUD rendering (all
    already correct in the existing <task>_overlay.mp4/_synced.csv). This
    skips all of that: no extrinsics, no joint projection — just rotate,
    undistort, trim to the SAME window run_overlay() would compute (same
    formula, same inputs), and encode.

    Args:
        task_name:       e.g. "SUB01_lifting_T01" — used for the output filename.
        video_path:      the ORIGINAL raw video (still must exist, e.g. in temp/).
        vicon_csv:       same cleaned Vicon CSV used for the original Overlay run
                         — only used to recompute the trim window, not projected.
        intrinsics_path: Path to intrinsic.json (K, dist only — no extrinsics needed).
        sync_path:       Path to the existing <task>_sync.json.
        output_dir:      Where to write <task>_rgb.mp4 (the task's output folder).
        rotate:          Override rotation; default reads from sync.json.

    Returns:
        Path to the written <task_name>_rgb.mp4.
    """
    def _cb(pct, step):
        if progress_cb:
            progress_cb(pct, step)

    _cb(5, "Loading calibration + sync data…")
    K, dist, _iw, _ih = load_intrinsics(intrinsics_path)

    with open(sync_path, encoding="utf-8") as f:
        sync = json.load(f)
    rotate    = rotate if rotate else sync.get("video_rotation", "none")
    sync_offset_s = sync["sync_offset_s"]
    manual_offset = sync.get("manual_frame_offset", 0)

    vicon_d            = read_vicon_trajectories(vicon_csv)
    vicon_times        = vicon_d["times"]
    joints             = vicon_d["data"]
    has_absolute_times = vicon_d["has_absolute_times"]
    # Same fully-valid-window fix as run_overlay() — this must stay
    # frame-aligned with the existing overlay/synced outputs (see docstring:
    # "same formula, same inputs, same result").
    t_s_min, t_s_max   = _first_last_fully_valid_time(joints, vicon_times)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, probe = cap.read()
    if not ok:
        raise RuntimeError("Cannot read first frame from video.")
    probe = rotate_frame(probe, rotate)
    H, W  = probe.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_plain_mp4 = output_dir / "_temp_plain_only.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_plain_mp4), fourcc, video_fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create temp video writer in {output_dir}")

    _cb(15, "Undistorting frames…")
    in_range_flags = []
    for V in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frame = rotate_frame(frame, rotate)
        writer.write(cv2.undistort(frame, K, dist))

        # Same time-based in-range check as run_overlay(), needed only for
        # the legacy (no time_s) trim fallback below.
        if has_absolute_times:
            in_range_flags.append(True)   # unused in this branch
        else:
            t_s = _video_to_vicon_time_s(V, sync)
            in_range_flags.append(t_s_min <= t_s <= t_s_max)

        if V % 60 == 0 and total_frames > 0:
            _cb(15 + int(70 * V / total_frames), f"Undistorting frame {V}/{total_frames}…")

    cap.release()
    writer.release()

    _cb(88, "Computing trim points…")
    if has_absolute_times:
        # Identical formula to run_overlay() — same inputs, same result,
        # so this video stays frame-aligned with the existing overlay/synced outputs.
        trim_start = max(0, round((t_s_min - sync_offset_s) * video_fps) - manual_offset)
        trim_end   = min(total_frames - 1, round((t_s_max - sync_offset_s) * video_fps) - manual_offset)
        while trim_start < total_frames and not in_range_flags[trim_start]:
            trim_start += 1
        while trim_end > 0 and not in_range_flags[trim_end]:
            trim_end -= 1
    else:
        trim_start = next((i for i, v in enumerate(in_range_flags) if v), 0)
        trim_end   = next((i for i in range(len(in_range_flags) - 1, -1, -1) if in_range_flags[i]),
                           len(in_range_flags) - 1)

    trim_start_sec = trim_start / video_fps
    trim_end_sec   = (trim_end + 1) / video_fps

    _cb(92, "Encoding plain RGB H.264…")
    out_path = output_dir / Artifacts.RGB_VIDEO
    out_path = _ffmpeg_trim(temp_plain_mp4, out_path, trim_start_sec, trim_end_sec)

    _cb(100, "Complete")
    return out_path


# ── Self-check ───────────────────────────────────────────────────────────────

def _demo() -> None:
    """Assert-based self-check: querying outside the Vicon time range must
    return NaN, never the boundary sample held constant (extrapolation)."""
    times = np.array([1.0, 2.0, 3.0, 4.0])
    joints = {"PELCEN": np.array([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.], [3., 3., 3.]])}

    before = _interp_joints_by_time(joints, times, 0.5)
    assert np.all(np.isnan(before["PELCEN"])), \
        "query before the first sample must be NaN, not the first sample's value"

    after = _interp_joints_by_time(joints, times, 4.5)
    assert np.all(np.isnan(after["PELCEN"])), \
        "query after the last sample must be NaN, not the last sample's value"

    inside = _interp_joints_by_time(joints, times, 2.5)
    assert np.allclose(inside["PELCEN"], [1.5, 1.5, 1.5]), inside

    # _first_last_fully_valid_time: two joints, one (RHIP) missing at the
    # start and end — the fully-valid window must be the INTERSECTION, not
    # PELCEN's own full range.
    t2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    j2 = {
        "PELCEN": np.ones((5, 3)),
        "RHIP":   np.array([[np.nan]*3, [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [np.nan]*3]),
    }
    t_min, t_max = _first_last_fully_valid_time(j2, t2)
    assert (t_min, t_max) == (2.0, 4.0), (t_min, t_max)

    # No frame with every joint valid -> must raise, not silently return the
    # raw range (which would let untracked frames leak into the trim window).
    j_never_all_valid = {
        "PELCEN": np.array([[1., 1., 1.], [np.nan]*3]),
        "RHIP":   np.array([[np.nan]*3, [1., 1., 1.]]),
    }
    try:
        _first_last_fully_valid_time(j_never_all_valid, np.array([1.0, 2.0]))
        raise AssertionError("expected ValueError when no frame has every joint valid")
    except ValueError:
        pass

    # _resample_source_by_time: the resampled value must be the WORSE of the
    # two bracketing samples, and "invalid" out of range.
    src_times = np.array([1.0, 2.0, 3.0])
    src = {"PELCEN": np.array(["c3d", "interpolated", "c3d"], dtype=object)}
    assert _resample_source_by_time(src, src_times, 1.5)["PELCEN"] == "interpolated", \
        "must report the LESS trustworthy of the two bracketing samples"
    assert _resample_source_by_time(src, src_times, 2.5)["PELCEN"] == "interpolated"
    assert _resample_source_by_time(src, src_times, 0.5)["PELCEN"] == "invalid", \
        "out-of-range query must be invalid, matching _interp_joints_by_time's NaN rule"
    assert _worse_source("c3d", "recovered_csv") == "recovered_csv"
    assert _worse_source("invalid", "c3d") == "invalid"

    print("overlay_pipeline self-check OK")


if __name__ == "__main__":
    _demo()
