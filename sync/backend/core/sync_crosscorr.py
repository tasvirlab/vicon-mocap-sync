"""
sync_crosscorr.py
=================
Multi-signal sync refinement via cross-correlation.

Signals cross-correlated against the video optical-flow signal:
  1. 3D marker velocity          — sum of all joint velocities from the cleaned CSV.
                                   Always available. Does NOT require calibration.
  2. Projected Vicon motion      — per-video-frame pixel displacement of projected joints.
                                   Most precise but requires intrinsics + extrinsics JSON.
  3. GRF magnitude derivative    — DIAGNOSTIC ONLY, excluded from selection. See below.

The selectable signal with the highest Pearson r is chosen. If the selectable
signals disagree, or if r is below MEDIUM, the rough wall-clock offset is kept.

Confidence = Pearson r at the winning lag, in [-1, 1]:
  HIGH   (r ≥ 0.55)  — correction applied
  MEDIUM (r ≥ 0.30)  — correction applied, review diagnostic_sync.png
  LOW    (r <  0.30)  — correction NOT applied, wall-clock offset kept

  These thresholds are only meaningful now that confidence is a real r. It used
  to be an unbounded peak-prominence z-score compared against the same numbers:
  pure white noise scores mean 2.89 on that metric, so EVERY trial passed "HIGH"
  and no correction was ever rejected. See _run_ncc().

Why GRF is diagnostic-only
──────────────────────────
GRF is the only signal independent of the Vicon markers, which makes it valuable
as a cross-check — but it cannot be correlated against optical flow directly.
Optical flow measures image SPEED; ground reaction force is proportional to
ACCELERATION (F = ma). Speed and acceleration are ~90 deg out of phase, so the
correlation peak is displaced by up to a quarter of the movement period —
roughly 550 ms for a ~2.2 s lifting cycle. On SUB01/lifting, GRF voted +50 ms
while the two marker-derived signals voted -217 and -233 ms: a 283 ms (17-frame)
disagreement that is structural, not noise. It is still computed and plotted so
the disagreement stays visible, but it no longer votes.

Signal agreement
────────────────
MarkerVelocity and ProjectedVicon are NOT independent — both derive from the
same Vicon markers correlated against the same optical flow — so their agreement
is weak corroboration. Their DISagreement, however, is a genuine red flag, and
is treated as one (see _AGREEMENT_TOL_FRAMES).

Sign convention (same as original):
  peak_lag > 0  → Vicon leads video → decrease sync_offset_s
  peak_lag < 0  → Vicon lags  video → increase sync_offset_s
  refined = rough_sync_offset_s - peak_lag / video_fps
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from core.vicon_io import (rotate_frame, rotate_points_2d, load_intrinsics,
                            read_vicon_trajectories, interp_marker_at_time)
from utils.task_paths import Artifacts

logger = logging.getLogger(__name__)

# ── Confidence thresholds ──────────────────────────────────────────────────────
# Compared against the Pearson r returned by _run_ncc(), in [-1, 1]. The numbers
# are unchanged from before, but they only became meaningful when the metric was
# fixed: they were always written for a correlation coefficient, while the code
# supplied an unbounded z-score that pure noise scores ~2.9 on.
_CONF_HIGH   = 0.55
_CONF_MEDIUM = 0.30   # below this → don't apply correction

# Signals computed and plotted for diagnostics but NOT allowed to vote.
# GRF is out because it is physically incomparable to optical flow (force ~
# acceleration vs. image speed, ~90 deg out of phase). See module docstring.
_DIAGNOSTIC_ONLY_SIGNALS = ("GRF",)

# Max spread (video frames) between selectable signals' chosen lags before the
# result is treated as untrustworthy. Two signals landing >2 frames apart on the
# same trial means at least one is locking onto the wrong feature, and there is
# no basis for preferring the higher-r one.
_AGREEMENT_TOL_FRAMES = 2

# Fraction of the search window at which a winning lag is treated as clipped
# rather than a genuine interior optimum. A lag landing here means the true
# best-aligning offset may lie outside the searched range entirely — the
# correlation curve was very likely still rising toward the edge rather than
# having turned over. A narrow window can pick an offset one frame from its edge while a
# wider window finds the real optimum outside it. Neither `signals_agree`
# nor the confidence label catches this on its own -- both signals agreed and
# scored HIGH while clipped at the same wrong boundary.
_BOUNDARY_TOL_FRAC = 0.9


# ── Video signal: optical flow ─────────────────────────────────────────────────

def _optical_flow_signal(video_path: Path, rotate: str,
                          downsample: int = 4,
                          bboxes: Optional[np.ndarray] = None,
                          progress_cb=None) -> Tuple[np.ndarray, float]:
    """
    Per-frame mean optical-flow magnitude (Farneback).

    Args:
        bboxes: optional (N, 4) array of (x0, y0, x1, y1) subject bounding boxes
            in ROTATED full-res pixels, one per video frame (NaN row = unknown).
            When a finite box is available the flow mean is restricted to that
            region so bystanders outside the box don't pollute the sync signal.
            Falls back to the whole frame when the box is missing/degenerate.

    Returns:
        (signal, video_fps)  signal shape (N,) float32
    """
    cap     = cv2.VideoCapture(str(video_path))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps     = cap.get(cv2.CAP_PROP_FPS)
    signal  = np.zeros(n_total, dtype=np.float32)
    prev    = None
    fidx    = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = rotate_frame(frame, rotate)
        h, w  = frame.shape[:2]
        small = cv2.resize(frame, (max(1, w // downsample),
                                   max(1, h // downsample)))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            sh, sw = mag.shape
            roi = None
            if bboxes is not None and fidx < len(bboxes) and np.all(np.isfinite(bboxes[fidx])):
                x0, y0, x1, y1 = bboxes[fidx] / downsample
                xi0, yi0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
                xi1, yi1 = min(sw, int(np.ceil(x1))), min(sh, int(np.ceil(y1)))
                if xi1 - xi0 >= 2 and yi1 - yi0 >= 2:
                    roi = mag[yi0:yi1, xi0:xi1]
            signal[fidx] = float((roi if roi is not None else mag).mean())
        prev  = gray
        fidx += 1
        if fidx % 30 == 0 and progress_cb and n_total > 0:
            progress_cb(int(100 * fidx / n_total))

    cap.release()
    return signal[:fidx], float(fps)


def _projected_bbox_per_frame(
        vicon_csv: Path, intrinsics_json: Path, extrinsics_json: Path,
        rough_sync_offset_s: float, manual_frame_offset: int,
        vicon_fps: float, video_fps: float, n_frames: int,
        rotate: str, margin_frac: float = 0.25) -> np.ndarray:
    """Per-video-frame subject bounding box (ROTATED full-res pixels) from the
    projected Vicon skeleton. Rows are NaN where the subject is out of range.

    Used to focus the optical-flow sync signal on the subject and ignore
    bystanders (Finding 8). Boxes are padded by margin_frac of their size.
    """
    K, dist, W, H = load_intrinsics(intrinsics_json)
    with open(extrinsics_json, encoding="utf-8") as f:
        ext = json.load(f)
    rvec = np.array(ext["rvec"],   dtype=np.float64).reshape(3, 1)
    tvec = np.array(ext["tvec_m"], dtype=np.float64).reshape(3, 1)

    vicon  = read_vicon_trajectories(vicon_csv)
    fr_arr = vicon["frames"]
    data   = vicon["data"]
    labels = vicon["labels"]
    vf_min, vf_max = float(fr_arr[0]), float(fr_arr[-1])

    boxes = np.full((n_frames, 4), np.nan, dtype=np.float64)
    for V in range(n_frames):
        t_s = (V + manual_frame_offset) / video_fps + rough_sync_offset_s
        vf  = t_s * vicon_fps
        if not (vf_min <= vf <= vf_max):
            continue
        pts = []
        for label in labels:
            p = interp_marker_at_time(data[label], fr_arr, vicon_fps, t_s)
            if not np.any(np.isnan(p)):
                pt2, _ = cv2.projectPoints(p.reshape(1, 1, 3), rvec, tvec, K, dist)
                pts.append(pt2.reshape(2))
        if len(pts) < 3:
            continue
        rp = rotate_points_2d(np.array(pts), rotate, W, H)
        x0, y0 = rp[:, 0].min(), rp[:, 1].min()
        x1, y1 = rp[:, 0].max(), rp[:, 1].max()
        mx, my = (x1 - x0) * margin_frac, (y1 - y0) * margin_frac
        boxes[V] = [x0 - mx, y0 - my, x1 + mx, y1 + my]
    return boxes


# ── Vicon signals ──────────────────────────────────────────────────────────────

def _grf_signal_from_c3d(c3d_path: Path, vicon_fps: float) -> Optional[np.ndarray]:
    """
    Extract total GRF magnitude derivative from C3D POINT data.

    Looks for LGroundReactionForce and RGroundReactionForce labels (standard
    Vicon Plug-in Gait output stored as POINT vectors).  Falls back to analog
    channel search if those labels are absent.

    Returns:
        np.ndarray shape (N,) at vicon_fps, or None if no GRF data found.
    """
    try:
        import ezc3d
    except ImportError:
        return None

    try:
        c      = ezc3d.c3d(str(c3d_path))
        labels = [l.strip() for l in c['parameters']['POINT']['LABELS']['value']]
        pts    = c['data']['points']   # (4, n_markers, n_frames)

        grf_total = None
        for label in ['LGroundReactionForce', 'RGroundReactionForce',
                       'LNormalisedGRF',       'RNormalisedGRF']:
            if label not in labels:
                continue
            i   = labels.index(label)
            xyz = pts[:3, i, :].T.copy()        # (N, 3)
            res = pts[3,  i, :]
            bad = (res < 0) | np.any(~np.isfinite(xyz), axis=1)
            xyz[bad] = 0.0
            mag = np.sqrt(np.sum(xyz**2, axis=1))
            grf_total = mag if grf_total is None else grf_total + mag

        if grf_total is None:
            # Try analog channels: look for Fz or Force labels
            try:
                analog_labels = [l.strip() for l in
                                  c['parameters']['ANALOG']['LABELS']['value']]
                analog_data   = c['data']['analogs'][0]  # (n_ch, n_analog_frames)
                analog_rate   = float(
                    c['parameters']['ANALOG']['RATE']['value'][0])
                fz_indices = [i for i, l in enumerate(analog_labels)
                              if 'FZ' in l.upper() or 'Fz' in l]
                if fz_indices:
                    fz = np.abs(analog_data[fz_indices, :]).sum(axis=0)
                    # Downsample from analog rate to vicon_fps
                    step = int(round(analog_rate / vicon_fps))
                    grf_total = fz[::step][: int(
                        c['data']['points'].shape[2])]
            except Exception:
                pass

        if grf_total is None or grf_total.max() < 1e-6:
            return None

        # Derivative = |d(GRF)/dt|  — emphasises events, removes static loading
        grf_diff = np.abs(np.gradient(grf_total))
        return grf_diff.astype(np.float64)

    except Exception as exc:
        logger.debug(f"GRF extraction failed: {exc}")
        return None


def _marker_velocity_signal(vicon_csv: Path, vicon_fps: float) -> np.ndarray:
    """
    Total 3D marker velocity from the cleaned Vicon CSV.

    Sum of per-joint speed across all 21 output joints.  No calibration needed.

    Returns:
        np.ndarray shape (N,) at vicon_fps.
    """
    traj = read_vicon_trajectories(vicon_csv, fps=vicon_fps, mm_to_m=True)
    frames = traj['frames']
    N      = len(frames)
    total  = np.zeros(N, dtype=np.float64)

    for label, xyz in traj['data'].items():
        # Forward difference, NaN where marker missing
        vel = np.full(N, np.nan)
        for t in range(1, N - 1):
            p0, p1 = xyz[t - 1], xyz[t + 1]
            if not (np.any(np.isnan(p0)) or np.any(np.isnan(p1))):
                vel[t] = np.linalg.norm(p1 - p0) * vicon_fps / 2.0
        # Replace NaN with 0 so missing markers don't dominate
        vel = np.where(np.isnan(vel), 0.0, vel)
        total += vel

    return total


def _vicon_projected_motion(
        vicon_csv: Path, intrinsics_json: Path, extrinsics_json: Path,
        rough_sync_offset_s: float, manual_frame_offset: int,
        vicon_fps: float, video_fps: float, n_frames: int,
        progress_cb=None) -> np.ndarray:
    """
    Per-video-frame mean projected Vicon joint displacement (pixels).
    Requires intrinsic + extrinsic calibration JSON.
    """
    K, dist, _, _ = load_intrinsics(intrinsics_json)
    with open(extrinsics_json, encoding="utf-8") as f:
        ext = json.load(f)
    rvec = np.array(ext["rvec"],   dtype=np.float64).reshape(3, 1)
    tvec = np.array(ext["tvec_m"], dtype=np.float64).reshape(3, 1)

    vicon  = read_vicon_trajectories(vicon_csv)
    fr_arr = vicon["frames"]
    data   = vicon["data"]
    labels = vicon["labels"]
    vf_min = float(fr_arr[0])
    vf_max = float(fr_arr[-1])

    signal   = np.zeros(n_frames, dtype=np.float32)
    prev_pts: dict = {}

    for V in range(n_frames):
        t_s = (V + manual_frame_offset) / video_fps + rough_sync_offset_s
        vf  = t_s * vicon_fps
        if not (vf_min <= vf <= vf_max):
            prev_pts = {}
            continue
        pts = {}
        for label in labels:
            p = interp_marker_at_time(data[label], fr_arr, vicon_fps, t_s)
            if not np.any(np.isnan(p)):
                pt2, _ = cv2.projectPoints(
                    p.reshape(1, 1, 3), rvec, tvec, K, dist)
                pts[label] = pt2.reshape(2)
        if prev_pts and pts:
            common = [l for l in pts if l in prev_pts]
            if common:
                signal[V] = float(np.mean(
                    [np.linalg.norm(pts[l] - prev_pts[l]) for l in common]))
        prev_pts = pts
        if V % 100 == 0 and progress_cb and n_frames > 0:
            progress_cb(int(100 * V / n_frames))

    return signal


# ── Signal utilities ───────────────────────────────────────────────────────────

def _resample_vicon_to_video(
        vicon_sig: np.ndarray,
        vicon_fps: float, video_fps: float,
        n_video_frames: int,
        rough_sync_offset_s: float,
        manual_frame_offset: int,
        t0_vicon: float = 0.01) -> np.ndarray:
    """
    Resample a Vicon-rate signal (100 Hz) to video frame rate, aligned by
    the rough sync offset so each video frame maps to the correct Vicon sample.

    Args:
        vicon_sig:           Shape (N,) at vicon_fps.
        t0_vicon:            Start time of vicon_sig in seconds from XCP START_TIME.
                             Prefer traj['times'][0] (exact) over frame/fps approximation.
        rough_sync_offset_s: video_start_utc - vicon_start_utc (seconds).
        manual_frame_offset: Extra video-frame offset to apply.

    Returns:
        np.ndarray shape (n_video_frames,) — 0.0 where out of Vicon range.
    """
    N_vicon = len(vicon_sig)
    out     = np.zeros(n_video_frames, dtype=np.float64)

    for V in range(n_video_frames):
        t_s   = (V + manual_frame_offset) / video_fps + rough_sync_offset_s
        idx_f = (t_s - t0_vicon) * vicon_fps
        i0    = int(np.floor(idx_f))
        alpha = idx_f - i0
        if 0 <= i0 < N_vicon - 1:
            out[V] = (1 - alpha) * vicon_sig[i0] + alpha * vicon_sig[i0 + 1]
        elif 0 <= i0 < N_vicon:
            out[V] = vicon_sig[i0]
    return out


def _highpass(signal: np.ndarray, fps: float,
              cutoff_hz: float = 0.5) -> np.ndarray:
    """
    First-order Butterworth highpass — removes DC / postural sway baseline.
    Falls back to simple mean-subtraction if scipy is unavailable.
    """
    try:
        from scipy.signal import butter, filtfilt
        nyq = fps / 2.0
        if cutoff_hz >= nyq:
            return signal - signal.mean()
        b, a = butter(1, cutoff_hz / nyq, btype='high')
        return filtfilt(b, a, signal)
    except Exception:
        return signal - signal.mean()


def _normalise(s: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalisation."""
    s  = s.astype(np.float64) - s.mean()
    sd = s.std()
    return s / sd if sd > 1e-9 else s


def _subframe_offset(corr: np.ndarray, peak_idx: int) -> float:
    """Sub-frame refinement of a correlation peak by parabolic interpolation.

    argmax resolves the lag only to a whole frame — ±16.7 ms at 60 fps. That is
    the dominant sync error once the coarse lock is right (SUB01's fixed-point
    test converged to exactly this 1-frame floor), and it propagates straight
    into MPJVE, where a 16.7 ms pairing error at ~100 deg/s trunk speed is ~1.7
    deg of fabricated angle difference.

    A correlation peak is locally quadratic, so fitting a parabola through the
    peak and its two neighbours locates the true maximum between samples:
        delta = 0.5 * (c[-1] - c[+1]) / (c[-1] - 2*c[0] + c[+1])
    This is the standard estimator (identical to the one used for GCC/PHAT time
    delay and radar range refinement) and typically lands within ~0.1 frame
    (~2 ms) on a smooth peak.

    Returns 0.0 when the peak is at an array edge (no neighbours to fit) or the
    curvature is degenerate. The result is clamped to ±0.5 frames: a parabola
    fitted to a noisy or non-quadratic peak can otherwise extrapolate outside the
    bracketing samples, which would move the answer to a lag the data does not
    support.
    """
    n = len(corr)
    if peak_idx <= 0 or peak_idx >= n - 1:
        return 0.0
    c0, c1, c2 = float(corr[peak_idx - 1]), float(corr[peak_idx]), float(corr[peak_idx + 1])
    denom = c0 - 2.0 * c1 + c2
    if abs(denom) < 1e-12:
        return 0.0
    delta = 0.5 * (c0 - c2) / denom
    if not np.isfinite(delta):
        return 0.0
    return float(np.clip(delta, -0.5, 0.5))


def _run_ncc(video_sig: np.ndarray, vicon_sig: np.ndarray,
             search_frames: int) -> Tuple[float, float, np.ndarray]:
    """
    Normalised cross-correlation in a ±search_frames window.

    Returns:
        (peak_lag, r_at_peak, search_corr_curve)
        peak_lag > 0 → Vicon leads → subtract from offset
        peak_lag is FRACTIONAL (sub-frame) — see _subframe_offset().
        r_at_peak    → Pearson correlation in [-1, 1] at the winning lag.

    Confidence is the Pearson r at the peak, NOT a peak-prominence z-score.
    The previous metric was (peak - background_mean) / background_std, compared
    against _CONF_HIGH = 0.55 as though it were a correlation coefficient. It is
    not: it is an unbounded z-score, and a smooth oscillating correlation curve
    ALWAYS has a maximum ~2-3 SD above its own mean whether or not the alignment
    is real. Measured: pure white noise scores mean 2.89 (min 2.14) on that
    metric and therefore passed the 0.55 "HIGH" gate 100% of the time — and real
    trials scored 2.39-3.20, i.e. inside the noise distribution. The gate could
    never reject anything, and the "±8 ms when HIGH" claim rested on it.

    Both inputs are z-scored by _normalise(), so dividing the raw correlation by
    the sample count yields the Pearson r directly. This is the BIASED estimator
    (constant divisor N rather than the per-lag overlap N-|lag|): it shrinks r
    slightly toward zero at large |lag|, which is the desirable behaviour here —
    large lags overlap less data and should not be rewarded for it. Over the
    +/-2 s search the overlap varies by only ~13%, so the bias is small.
    """
    try:
        from scipy.signal import correlate as _corr
    except ImportError:
        raise ImportError(
            "scipy is required for cross-correlation refinement. "
            "Run: pip install scipy"
        )

    nv = _normalise(video_sig)
    nc = _normalise(vicon_sig)
    n  = min(len(nv), len(nc))
    if n < 2:
        return 0, 0.0, np.array([0.0])

    full_corr   = _corr(nv, nc, mode="full") / float(n)   # -> Pearson r per lag
    mid         = len(nc) - 1
    lo          = max(0, mid - search_frames)
    hi          = min(len(full_corr), mid + search_frames + 1)
    search_corr = full_corr[lo:hi]

    if len(search_corr) == 0:
        return 0, 0.0, np.array([0.0])

    best_idx  = int(np.argmax(search_corr))
    peak_lag  = float(best_idx - (mid - lo)) + _subframe_offset(search_corr, best_idx)
    r_at_peak = float(np.clip(search_corr[best_idx], -1.0, 1.0))

    return peak_lag, r_at_peak, search_corr


# ── Main refinement function ───────────────────────────────────────────────────

def refine_sync_by_crosscorr(
        video_path:            Path,
        vicon_csv:             Path,
        rough_sync_offset_s:   float,
        manual_frame_offset:   int,
        vicon_fps:             float,
        rotate:                str,
        search_range_s:        float = 4.0,
        c3d_path:              Optional[Path] = None,
        intrinsics_json:       Optional[Path] = None,
        extrinsics_json:       Optional[Path] = None,
        output_dir:            Optional[Path] = None,
        progress_cb=None) -> dict:
    """
    Multi-signal sync refinement.

    Tries up to three Vicon signals against video optical flow, selects the one
    with the highest peak-prominence confidence, and applies the correction only
    if confidence >= MEDIUM threshold.

    Args:
        c3d_path:        Optional C3D file for GRF extraction (best signal).
        intrinsics_json: Optional — enables projected-Vicon-motion signal.
        extrinsics_json: Optional — enables projected-Vicon-motion signal.
        output_dir:      Directory for diagnostic_sync.png.
        progress_cb:     Callback(pct: int, step: str).

    Returns:
        dict with keys:
          refined_offset_s    — float, seconds
          confidence          — float 0-1+ (peak prominence score)
          confidence_label    — 'HIGH' | 'MEDIUM' | 'LOW'
          signal_used         — name of winning signal
          correction_applied  — bool
          correction_frames   — float (sub-frame lag applied)
          all_signals         — {name: {confidence, lag_frames, lag_ms}}
    """
    def _cb(pct: int, step: str):
        if progress_cb:
            progress_cb(pct, step)

    # ── Step 1: optical flow from video (needed for all correlations) ──────────
    _cb(0, "Computing optical flow…")

    def _of_cb(pct):
        _cb(int(pct * 0.40), "Computing optical flow…")

    # When calibration is available, focus the flow on the subject's projected
    # bounding box so bystanders in the lab don't pollute the sync signal
    # (Finding 8). Needs the video frame count first, so do a light pre-pass.
    subject_bboxes = None
    if intrinsics_json and extrinsics_json:
        try:
            cap = cv2.VideoCapture(str(video_path))
            n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            v_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            subject_bboxes = _projected_bbox_per_frame(
                vicon_csv, intrinsics_json, extrinsics_json,
                rough_sync_offset_s, manual_frame_offset,
                vicon_fps, v_fps, n_vid, rotate)
            logger.info("Optical-flow ROI restricted to projected subject bbox "
                        f"({int(np.isfinite(subject_bboxes[:, 0]).sum())}/{n_vid} frames)")
        except Exception as exc:
            logger.warning(f"Subject-bbox pre-pass failed, using whole-frame flow: {exc}")
            subject_bboxes = None

    of_sig, video_fps = _optical_flow_signal(
        video_path, rotate, bboxes=subject_bboxes, progress_cb=_of_cb)
    n_frames     = len(of_sig)
    search_frames = int(search_range_s * video_fps)

    # Preprocess video signal
    of_hp = _highpass(of_sig, video_fps, cutoff_hz=0.3)

    _cb(40, "Preparing Vicon signals…")

    # ── Step 2: read Vicon data once ──────────────────────────────────────────
    traj = read_vicon_trajectories(vicon_csv, fps=vicon_fps, mm_to_m=True)
    # Use times[0] (exact seconds from XCP START_TIME) when available;
    # fall back to frame-number approximation for legacy CSVs without time_s.
    if traj.get('has_absolute_times') and len(traj['times']):
        t0_vicon = float(traj['times'][0])
    else:
        t0_vicon = int(traj['frames'][0]) / vicon_fps if len(traj['frames']) else 0.01

    def _to_video(vicon_sig: np.ndarray) -> np.ndarray:
        return _resample_vicon_to_video(
            vicon_sig, vicon_fps, video_fps, n_frames,
            rough_sync_offset_s, manual_frame_offset, t0_vicon)

    all_results: dict = {}

    # ── Signal A: GRF derivative (from C3D) ───────────────────────────────────
    if c3d_path and Path(c3d_path).exists():
        _cb(42, "Extracting GRF signal…")
        grf_raw = _grf_signal_from_c3d(Path(c3d_path), vicon_fps)
        if grf_raw is not None and grf_raw.max() > 1e-9:
            grf_hp   = _highpass(grf_raw, vicon_fps, cutoff_hz=0.3)
            grf_vid  = _to_video(grf_hp)
            lag, conf, corr = _run_ncc(of_hp, grf_vid, search_frames)
            all_results['GRF'] = {
                'lag_frames': lag, 'confidence': conf, 'corr': corr,
                'lag_ms': round(lag / video_fps * 1000, 1)}
            logger.info(f"GRF signal: lag={lag:+.2f} fr, confidence={conf:.3f}")

    # ── Signal B: 3D marker velocity (always available) ───────────────────────
    _cb(50, "Computing marker velocity signal…")
    try:
        mv_raw = _marker_velocity_signal(vicon_csv, vicon_fps)
        mv_hp  = _highpass(mv_raw, vicon_fps, cutoff_hz=0.3)
        mv_vid = _to_video(mv_hp)
        lag, conf, corr = _run_ncc(of_hp, mv_vid, search_frames)
        all_results['MarkerVelocity'] = {
            'lag_frames': lag, 'confidence': conf, 'corr': corr,
            'lag_ms': round(lag / video_fps * 1000, 1)}
        logger.info(f"MarkerVelocity: lag={lag:+.2f} fr, confidence={conf:.3f}")
    except Exception as exc:
        logger.warning(f"Marker velocity signal failed: {exc}")

    # ── Signal C: projected Vicon motion (needs calibration) ──────────────────
    if intrinsics_json and extrinsics_json:
        _cb(58, "Computing projected Vicon motion…")
        try:
            def _proj_cb(pct):
                _cb(58 + int(pct * 0.30), "Projected Vicon motion…")
            proj_raw = _vicon_projected_motion(
                vicon_csv, intrinsics_json, extrinsics_json,
                rough_sync_offset_s, manual_frame_offset,
                vicon_fps, video_fps, n_frames,
                progress_cb=_proj_cb)
            proj_hp = _highpass(proj_raw, video_fps, cutoff_hz=0.3)
            lag, conf, corr = _run_ncc(of_hp, proj_hp, search_frames)
            all_results['ProjectedVicon'] = {
                'lag_frames': lag, 'confidence': conf, 'corr': corr,
                'lag_ms': round(lag / video_fps * 1000, 1)}
            logger.info(f"ProjectedVicon: lag={lag:+.2f} fr, confidence={conf:.3f}")
        except Exception as exc:
            logger.warning(f"Projected Vicon signal failed: {exc}")

    _cb(90, "Selecting best signal…")

    # ── Step 3: select best signal ─────────────────────────────────────────────
    if not all_results:
        return {
            'refined_offset_s':   rough_sync_offset_s,
            'confidence':         0.0,
            'confidence_label':   'LOW',
            'signal_used':        'none',
            'correction_applied': False,
            'correction_frames':  0,
            'all_signals':        {},
        }

    # Only marker-derived signals may vote; GRF is diagnostic-only (see module
    # docstring — it is ~90 deg out of phase with optical flow by construction).
    selectable = {k: v for k, v in all_results.items()
                  if k not in _DIAGNOSTIC_ONLY_SIGNALS}
    if not selectable:
        logger.warning("Only diagnostic-only signals available — keeping "
                       "wall-clock offset unrefined.")
        return {
            'refined_offset_s':   rough_sync_offset_s,
            'confidence':         0.0,
            'confidence_label':   'LOW',
            'signal_used':        'none',
            'correction_applied': False,
            'correction_frames':  0,
            'all_signals':        {k: {kk: vv for kk, vv in v.items() if kk != 'corr'}
                                   for k, v in all_results.items()},
        }

    best_name = max(selectable, key=lambda k: selectable[k]['confidence'])
    best      = selectable[best_name]
    conf      = best['confidence']
    peak_lag  = best['lag_frames']

    # ── Signal agreement ──────────────────────────────────────────────────────
    # The real quality check. A high r only says one signal matched *something*;
    # two signals landing on the same lag says they matched the same event.
    lags        = [v['lag_frames'] for v in selectable.values()]
    lag_spread  = float(max(lags) - min(lags)) if len(lags) > 1 else 0.0
    agree       = lag_spread <= _AGREEMENT_TOL_FRAMES
    spread_ms   = round(lag_spread / video_fps * 1000, 1)

    # ── Search-boundary clipping ──────────────────────────────────────────────
    # A winning lag near the edge of +-search_frames means the true optimum may
    # sit outside the searched range — the correlation curve was very likely
    # still rising toward the edge rather than having turned over. Neither
    # agreement nor confidence catches this: both signals can agree and both
    # can score HIGH while clipped at the same wrong boundary (a narrow
    # window can clip at its edge while a wider one finds the real optimum past it).
    near_boundary = abs(peak_lag) >= _BOUNDARY_TOL_FRAC * search_frames

    if conf >= _CONF_HIGH:
        conf_label = 'HIGH'
    elif conf >= _CONF_MEDIUM:
        conf_label = 'MEDIUM'
    else:
        conf_label = 'LOW'

    # Disagreement caps the label at MEDIUM: the correction is still applied
    # (the wall-clock offset is not obviously better), but the trial is marked
    # for review rather than silently trusted.
    if not agree and conf_label == 'HIGH':
        conf_label = 'MEDIUM'
        logger.warning(
            f"Signals disagree by {lag_spread:.2f} frames ({spread_ms:+.1f} ms) > "
            f"tolerance {_AGREEMENT_TOL_FRAMES} — downgrading HIGH to MEDIUM. "
            f"Lags: " + ", ".join(f"{k}={v['lag_ms']:+.0f}ms"
                                  for k, v in selectable.items())
            + ". Review diagnostic_sync.png before trusting this trial."
        )

    if near_boundary and conf_label == 'HIGH':
        conf_label = 'MEDIUM'
        logger.warning(
            f"Winning lag {peak_lag:+.1f} frames sits within "
            f"{(1 - _BOUNDARY_TOL_FRAC) * 100:.0f}% of the +-{search_frames}-frame "
            f"search boundary — downgrading HIGH to MEDIUM. The true optimum may "
            f"lie outside the searched range; consider re-running with a wider "
            f"search_range_s. Review diagnostic_sync.png before trusting this trial."
        )
    elif near_boundary:
        logger.warning(
            f"Winning lag {peak_lag:+.1f} frames sits within "
            f"{(1 - _BOUNDARY_TOL_FRAC) * 100:.0f}% of the +-{search_frames}-frame "
            f"search boundary — the true optimum may lie outside the searched range."
        )

    apply_correction = conf >= _CONF_MEDIUM
    delta_s          = peak_lag / video_fps
    refined          = rough_sync_offset_s - delta_s if apply_correction \
                       else rough_sync_offset_s

    # GRF cannot vote, but as the only marker-independent signal its lag is
    # still worth logging next to the winner — a large gap is a hint that the
    # marker signals locked onto a repetition of the movement rather than the
    # right one.
    if 'GRF' in all_results:
        logger.info(f"GRF (diagnostic only): lag="
                    f"{all_results['GRF']['lag_ms']:+.1f} ms vs selected "
                    f"{best['lag_ms']:+.1f} ms")

    logger.info(
        f"Best signal: {best_name}  r={conf:.3f} ({conf_label})  "
        f"lag={peak_lag:+.2f} fr ({best['lag_ms']:+.1f} ms)  "
        f"spread={lag_spread:.2f} fr  agree={agree}  applied={apply_correction}"
    )

    # ── Step 4: diagnostic plot ────────────────────────────────────────────────
    if output_dir is not None:
        _save_diagnostic(
            of_sig, all_results, best_name, video_fps,
            search_frames, rough_sync_offset_s, output_dir)

    _cb(99, "Done")

    return {
        'refined_offset_s':   round(refined, 6),
        'confidence':         round(conf, 4),
        'confidence_metric':  'pearson_r',   # was an unbounded prominence z-score
        'confidence_label':   conf_label,
        'signal_used':        best_name,
        'correction_applied': apply_correction,
        'correction_frames':  round(peak_lag, 3) if apply_correction else 0.0,
        'correction_ms':      round(best['lag_ms'], 1) if apply_correction else 0.0,
        # Agreement between the selectable (marker-derived) signals. `agree=False`
        # means at least one signal locked onto the wrong feature — the trial
        # needs a human look at diagnostic_sync.png before its pairs are used for
        # training or validation.
        'signals_agree':      bool(agree),
        'lag_spread_frames':  round(lag_spread, 3),
        'lag_spread_ms':      spread_ms,
        'agreement_tol_frames': _AGREEMENT_TOL_FRAMES,
        # True when the winning lag sits within _BOUNDARY_TOL_FRAC of the
        # +-search_frames edge — the true optimum may lie outside the searched
        # range. Neither `signals_agree` nor `confidence` catches this on their
        # own (see the near-boundary check above). Re-run with a wider
        # search_range_s when this is true, then re-check.
        'near_search_boundary': bool(near_boundary),
        'search_range_frames': search_frames,
        'boundary_tol_frac':  _BOUNDARY_TOL_FRAC,
        'needs_review':       bool((not agree) or conf < _CONF_HIGH or near_boundary),
        'all_signals': {
            k: {'confidence': round(v['confidence'], 4),
                'lag_frames': round(v['lag_frames'], 3),
                'lag_ms':     v['lag_ms'],
                'diagnostic_only': k in _DIAGNOSTIC_ONLY_SIGNALS}
            for k, v in all_results.items()},
    }


# ── Diagnostic plot ────────────────────────────────────────────────────────────

def _save_diagnostic(of_sig, all_results, best_name, video_fps,
                     search_frames, rough_offset_s, out_dir: Path):
    """Save diagnostic_sync.png with motion signals and all NCC curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def _nd(s):
            s = np.asarray(s, dtype=float)
            s = s - s.min()
            return s / s.max() if s.max() > 0 else s

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Top panel: video optical flow
        t = np.arange(len(of_sig)) / video_fps
        axes[0].plot(t, _nd(of_sig), color='steelblue', alpha=0.7,
                     label='Optical flow (video)', linewidth=0.8)
        axes[0].set_xlabel('Time from video start (s)')
        axes[0].set_ylabel('Motion energy (normalised)')
        axes[0].set_title('Video motion signal')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Bottom panel: NCC curves for all signals
        colors = {'GRF': 'firebrick', 'MarkerVelocity': 'darkorange',
                  'ProjectedVicon': 'green'}
        for name, res in all_results.items():
            corr  = res['corr']
            lags  = (np.arange(len(corr)) - (len(corr) // 2)) / video_fps * 1000
            col   = colors.get(name, 'gray')
            lw    = 2.0 if name == best_name else 1.0
            alpha = 1.0 if name == best_name else 0.5
            label = (f"{name}  r={res['confidence']:.2f}  "
                     f"lag={res['lag_ms']:+.0f} ms")
            if name in _DIAGNOSTIC_ONLY_SIGNALS:
                label += "  (diagnostic only — does not vote)"
            if name == best_name:
                label += "  ← SELECTED"
            axes[1].plot(lags, corr, color=col, linewidth=lw, alpha=alpha,
                         label=label)
            # Mark peak
            peak_ms = res['lag_ms']
            axes[1].axvline(peak_ms, color=col, linestyle='--', alpha=0.6,
                            linewidth=lw)

        axes[1].axvline(0, color='gray', linestyle=':', alpha=0.5,
                        label='No correction')
        axes[1].set_xlabel('Correction lag (ms)  [+ = Vicon leads video]')
        axes[1].set_ylabel('Pearson r')
        _sel = {k: v for k, v in all_results.items()
                if k not in _DIAGNOSTIC_ONLY_SIGNALS}
        _lags = [v['lag_frames'] for v in _sel.values()]
        _spread_ms = ((max(_lags) - min(_lags)) / video_fps * 1000) if len(_lags) > 1 else 0.0
        _warn = ('' if abs(_spread_ms) <= _AGREEMENT_TOL_FRAMES / video_fps * 1000
                 else f'   ⚠ voting signals disagree by {_spread_ms:.0f} ms — REVIEW')
        axes[1].set_title(
            f'Cross-correlation — best: {best_name}  '
            f'r={all_results[best_name]["confidence"]:.2f}  '
            f'lag={all_results[best_name]["lag_ms"]:+.0f} ms  '
            f'rough_offset={rough_offset_s:+.3f} s{_warn}')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        # QA artifact: verification/debugging only, never an analysis input, so
        # it lives in qa/ rather than beside the data files.
        qa = out_dir / Artifacts.QA_DIR
        qa.mkdir(parents=True, exist_ok=True)
        out_png = qa / Artifacts.QA_SYNC_DIAG
        plt.savefig(str(out_png), dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Diagnostic saved → {out_png}")
    except Exception as exc:
        logger.warning(f"Diagnostic plot failed: {exc}")
