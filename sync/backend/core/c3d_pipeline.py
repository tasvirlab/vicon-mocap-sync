"""
c3d_pipeline.py
===============
Read a Vicon Nexus C3D file and produce a clean 21-joint CSV in the same
format as vicon_clean.py.

When Plug-in Gait (PiG) model outputs are present in the C3D, true
anatomical joint centres are used instead of surface markers:

  Verified PiG virtual marker mapping (Z-heights confirmed from actual C3D):
  Joint        | Surface (fallback)        | PiG (preferred)
  -------------|---------------------------|--------------------------------------------
  RHIP/LHIP    | Harrington 2007 (ALWAYS)  | NONE — see "O vs A/L/P" below. NOT RFEP/LFEP.
  RKNE/LKNE    | RKNE/LKNE surface marker  | RFEO/LFEO  distal  femur = KJC  (Z≈435mm)
  RANK/LANK    | RANK/LANK surface marker  | RTIO/LTIO  distal  tibia = AJC  (Z≈74mm)
  RELB/LELB    | RELB/LELB surface marker  | RHUO/LHUO  distal humerus = EJC (Z≈1070mm)
  RWR/LWR      | (RWRA+RWRB)/2             | RRAO/LRAO  distal  radius = WJC (Z≈851mm)
  RSHO/LSHO    | RSHO/LSHO acromion marker | (no change — PiG shoulder not stored as point)
  RHAND/LHAND  | RFIN/LFIN finger marker   | (no change — RHNO/LHNO too distal)
  PELCEN       | mean(LASI/RASI/LPSI/RPSI) | (no change — PELO sits at hip level, ~100mm
               |                           |  below the physical ASIS/PSIS centroid)

  ── O vs A/L/P: only "O" outputs are joint centres ──────────────────────────
  Vicon exports FOUR points per segment: <SEG>O/A/L/P = Origin, Anterior,
  Lateral, Proximal. Only the ORIGIN is an anatomical joint centre. A/L/P are
  AXIS-DIRECTION markers, each placed at origin + unit_axis * segment_scale.

  Proof (SUB01/lifting): |O->A| == |O->L| == |O->P| exactly, per segment —
  pelvis 45.6/45.6/45.6 mm, right femur 442.2/442.2/442.2, head 254.9 x3.
  Three mutually perpendicular axes cannot be coincidentally equidistant.

  Consequence: RFEP is the femur's proximal-AXIS marker, NOT the hip joint
  centre. |RFEP-LFEP| = 45.6mm on SUB01 — hip centres 23mm either side of the
  midline, which is anatomically impossible (this subject's knees are 240mm
  apart, ankles 221mm, shoulders 311mm; real hip width is ~180-210mm).

  A previous note here read "Verified: RFEP/LFEP Z≈884mm → hip joint centre
  level". That check passed and was still wrong: the femur's proximal axis does
  point toward the hip, so RFEP lands at roughly hip HEIGHT. The height was
  verified; the width never was. Using it collapsed the pelvis frame's
  mediolateral axis onto a 45.6mm baseline, amplifying any joint noise ~11x
  into the trunk lateral/twist angles built on that frame.

  RFEO/RTIO/RHUO/RRAO/PELO above are all ORIGINS, so they remain correct and
  in use. Never substitute an A/L/P point for a joint centre.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


# ── Output column order (same as vicon_clean.py) ──────────────────────────────

OUTPUT_COLUMNS: List[str] = [
    "Frame",
    "HD_X",     "HD_Y",     "HD_Z",
    "C7_X",     "C7_Y",     "C7_Z",
    "RSHO_X",   "RSHO_Y",   "RSHO_Z",
    "LSHO_X",   "LSHO_Y",   "LSHO_Z",
    "RELB_X",   "RELB_Y",   "RELB_Z",
    "LELB_X",   "LELB_Y",   "LELB_Z",
    "RWR_X",    "RWR_Y",    "RWR_Z",
    "LWR_X",    "LWR_Y",    "LWR_Z",
    "RHAND_X",  "RHAND_Y",  "RHAND_Z",
    "LHAND_X",  "LHAND_Y",  "LHAND_Z",
    "T10_X",    "T10_Y",    "T10_Z",
    "L45_X",    "L45_Y",    "L45_Z",
    "PELCEN_X", "PELCEN_Y", "PELCEN_Z",
    "RHIP_X",   "RHIP_Y",   "RHIP_Z",
    "LHIP_X",   "LHIP_Y",   "LHIP_Z",
    "RKNE_X",   "RKNE_Y",   "RKNE_Z",
    "LKNE_X",   "LKNE_Y",   "LKNE_Z",
    "RANK_X",   "RANK_Y",   "RANK_Z",
    "LANK_X",   "LANK_Y",   "LANK_Z",
    "RTOE_X",   "RTOE_Y",   "RTOE_Z",
    "LTOE_X",   "LTOE_Y",   "LTOE_Z",
]

MAX_INTERP_GAP = 10

# ── Biomechanical sanity check + smoothing constants ────────────────────────────
VELOCITY_THRESHOLD_MPS = 5.0   # max plausible joint speed, frame-to-frame (m/s)
BUTTERWORTH_CUTOFF_HZ  = 6.0   # default low-pass cutoff; must stay below Nyquist
BUTTERWORTH_ORDER      = 4


# ── C3D loading ────────────────────────────────────────────────────────────────

def _load_c3d_markers(c3d_path: Path) -> Tuple[dict, float, int, int]:
    """
    Load all marker trajectories from a C3D file.

    Returns:
        markers: {label: np.ndarray (N, 3)} in mm, NaN where occluded.
        frame_rate: float (Hz)
        n_frames: int
        actual_start_field: int (TRIAL.ACTUAL_START_FIELD, default 1)
    """
    import ezc3d
    c = ezc3d.c3d(str(c3d_path))

    points     = c['data']['points']            # (4, n_markers, n_frames)
    labels     = c['parameters']['POINT']['LABELS']['value']
    frame_rate = float(c['parameters']['POINT']['RATE']['value'][0])
    n_frames   = points.shape[2]

    try:
        start = int(c['parameters']['TRIAL']['ACTUAL_START_FIELD']['value'][0])
    except Exception:
        start = 1

    markers: dict = {}
    for i, label in enumerate(labels):
        label = label.strip()
        xyz       = points[:3, i, :].T.copy()   # (N, 3) in mm
        residuals = points[3, i, :]              # (N,)
        # Mark invalid frames: Vicon uses residual < 0 for occluded markers.
        # Also guard against all-zero coordinates (another Vicon convention).
        # Also guard against frames where Vicon stored NaN coordinates despite
        # a positive residual (seen in Task2 LPSI: residual=1.0 but XYZ=NaN).
        invalid = (residuals < 0) | (
            (xyz[:, 0] == 0) & (xyz[:, 1] == 0) & (xyz[:, 2] == 0)
            & (residuals == 0)
        ) | np.any(~np.isfinite(xyz), axis=1)
        xyz[invalid] = np.nan
        markers[label] = xyz

    # Strip biomechanically implausible frame-to-frame jumps before any
    # derivation (Harrington/PiG) consumes these markers — see despike_velocity().
    for label in markers:
        markers[label] = despike_velocity(markers[label], frame_rate)

    return markers, frame_rate, n_frames, start


# ── Biomechanical sanity check ──────────────────────────────────────────────────

def despike_velocity(
    arr: np.ndarray, frame_rate: float, max_speed_mps: float = VELOCITY_THRESHOLD_MPS,
) -> np.ndarray:
    """Strip frame-to-frame jumps faster than biologically plausible to NaN.

    Applied to raw markers BEFORE Harrington/PiG derivation — a single mislabeled
    or jittered marker should invalidate that frame's derived joint, not silently
    corrupt the regression. Stripped frames re-enter the normal gap-fill /
    fallback-CSV pipeline like any other occlusion.

    [ASSUMPTION] When a jump between frame i and i+1 exceeds the threshold, the
    LATER frame (i+1) is marked NaN, not the earlier one — frame i was already
    accepted by the previous iteration of this check.

    NaN-safe by construction: a displacement computed against an already-NaN
    neighbour is itself NaN, and `NaN > threshold` is False, so genuinely
    occluded frames are never mistakenly flagged (and never mask a real spike
    on their other side either).

    Args:
        arr:           (N, 3) float64, millimeters.
        frame_rate:    Hz, from the C3D/Nexus header.
        max_speed_mps: threshold in meters/second.

    Returns:
        (N, 3) copy of arr with implausible-jump frames set to NaN.
    """
    result = arr.copy()
    with np.errstate(invalid="ignore"):
        disp_mm = np.linalg.norm(np.diff(result, axis=0), axis=1)   # (N-1,)
        speed_mps = disp_mm / 1000.0 * frame_rate
        bad = speed_mps > max_speed_mps
    result[1:][bad] = np.nan
    return result


# ── Harrington 2007 HJC + pelvis helpers (fallback) ───────────────────────────

def _safe_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norms > 0, v / norms, np.nan)


# Plausible inter-HJC distance for an adult. Real hip width is ~180-210mm;
# these bounds are deliberately wide so only a genuinely broken derivation trips
# them, not an unusual build. Derived hips outside this range invalidate the
# pelvis frame (right = unit(RHIP - LHIP)) and therefore every trunk angle built
# on it, so this is a hard failure rather than a warning — silently continuing is
# exactly how the RFEP/LFEP mix-up (45.6mm) reached the angle stage unnoticed.
HIP_WIDTH_MIN_MM = 120.0
HIP_WIDTH_MAX_MM = 260.0


def check_hip_width(rhip: np.ndarray, lhip: np.ndarray) -> float:
    """Validate derived hip joint centres and return the median hip width (mm).

    Args:
        rhip, lhip: (N, 3) hip joint centre trajectories in mm.

    Returns:
        Median inter-HJC distance in mm, over frames where both are finite.

    Raises:
        ValueError: if no frame has both hips finite, or if the median hip width
            falls outside [HIP_WIDTH_MIN_MM, HIP_WIDTH_MAX_MM].
    """
    w = np.linalg.norm(rhip - lhip, axis=1)
    valid = w[np.isfinite(w)]
    if valid.size == 0:
        raise ValueError(
            "Hip joint centres are NaN on every frame — the pelvis markers "
            "(LASI/RASI/LPSI/RPSI) are missing or unlabelled in this trial, so "
            "no pelvis frame can be built and no trunk angle is defined."
        )
    median = float(np.median(valid))
    if not (HIP_WIDTH_MIN_MM <= median <= HIP_WIDTH_MAX_MM):
        raise ValueError(
            f"Derived hip width {median:.1f} mm is outside the plausible range "
            f"[{HIP_WIDTH_MIN_MM:.0f}, {HIP_WIDTH_MAX_MM:.0f}] mm. The pelvis "
            f"frame (right = unit(RHIP - LHIP)) and every trunk angle built on "
            f"it would be unreliable. Check that the hips come from the "
            f"ASIS/PSIS markers via Harrington and NOT from RFEP/LFEP, which "
            f"are axis-direction markers, not joint centres (see module "
            f"docstring)."
        )
    return median


def _harrington_hjc(lasi, rasi, lpsi, rpsi):
    """Harrington 2007 HJC regression. Returns rhjc, lhjc, pelcen, l45."""
    MASI = (lasi + rasi) / 2.0
    MPSI = (lpsi + rpsi) / 2.0
    C    = np.linalg.norm(rasi - lasi, axis=1)
    PD   = np.linalg.norm(MASI - MPSI, axis=1)
    x_hat   = _safe_normalize(rasi - lasi)
    ant_u   = _safe_normalize(MASI - MPSI)
    z_hat   = _safe_normalize(np.cross(x_hat, ant_u))
    y_hat   = _safe_normalize(np.cross(z_hat, x_hat))
    # MEDIOLATERAL half-offset. Was `0.24 * C - 9.9`, which is Harrington's
    # ANTEROPOSTERIOR coefficient applied to the mediolateral axis — it yielded
    # a 121mm hip width where anatomy gives ~180-210mm (this subject's knees are
    # 240mm apart, ankles 221mm).
    #
    # `0.33 * C + 7.3` is Harrington's published mediolateral term and is
    # confirmed empirically here, not just by citation: sweeping the half-offset
    # and scoring the resulting knee angle against the C3D's own PiG
    # RKneeAngles gives a smooth minimum at hip width 200-220mm
    # (SUB01/lifting, 1573 frames):
    #     hipW 120mm -> knee MAE 7.00 deg   (old coefficient)
    #     hipW 200mm -> knee MAE 5.47 deg   (this coefficient, = 208mm)
    #     hipW 220mm -> knee MAE 5.47 deg   (optimum)
    #     hipW 260mm -> knee MAE 6.03 deg
    # The hip joint centre sets the knee angle (knee = angle(RHIP, RKNE, RANK)),
    # so this directly improves the REBA leg score.
    x_R  = (0.33 * C + 7.3)[:, None]
    # [UNVERIFIED] The y (inferior) and z (posterior) terms below are carried
    # over unchanged. y = -0.30*C - 10.9 matches Harrington's published inferior
    # term. The z form (-0.22*C - 0.16*PD - 7.0) does NOT match the published
    # single-variable AP term (-0.24*PD - 9.9) and may be a different regression
    # variant. The knee-MAE test above constrains the MEDIOLATERAL axis only and
    # cannot arbitrate these — verify against Harrington (2007) directly before
    # relying on absolute hip-centre depth/height.
    y_off = (-0.30 * C - 10.9)[:, None]
    z_off = (-0.22 * C - 0.16 * PD - 7.0)[:, None]
    rhjc = MASI + x_R * x_hat + y_off * y_hat + z_off * z_hat
    lhjc = MASI - x_R * x_hat + y_off * y_hat + z_off * z_hat
    pelcen = (lasi + rasi + lpsi + rpsi) / 4.0
    PD2  = PD[:, None]
    l45  = MPSI + 0.30 * PD2 * z_hat + 0.15 * PD2 * y_hat
    return rhjc, lhjc, pelcen, l45


# ── Gap interpolation (same as vicon_clean.py) ────────────────────────────────

def _fill_gaps(arr: np.ndarray, max_gap: Optional[int] = MAX_INTERP_GAP) -> np.ndarray:
    """
    Linear interpolation for interior NaN gaps, all-or-nothing per run.

    A contiguous NaN run is filled ENTIRELY if its length <= max_gap and it
    has valid data on both sides; otherwise left entirely as NaN. This matches
    the behavior of vicon_reba/load.py and avoids partial fills where only the
    first N frames of a longer gap are interpolated.

    Args:
        arr:     (N, 3) float64 array.
        max_gap: Maximum consecutive NaN frames to fill.
                 None = unlimited (fill all interior gaps regardless of length).
    """
    result = arr.copy()
    n = len(arr)
    for col in range(3):
        col_data = result[:, col]
        is_nan = np.isnan(col_data)
        if not is_nan.any():
            continue
        i = 0
        while i < n:
            if not is_nan[i]:
                i += 1
                continue
            j = i
            while j < n and is_nan[j]:
                j += 1
            if i > 0 and j < n and (max_gap is None or (j - i) <= max_gap):
                col_data[i:j] = np.interp(
                    np.arange(i, j), [i - 1, j], [col_data[i - 1], col_data[j]]
                )
            i = j
    return result


# ── Low-pass smoothing (Step 4.5) ────────────────────────────────────────────────

def _filter_1d_segments(x: np.ndarray, b: np.ndarray, a: np.ndarray, min_len: int) -> np.ndarray:
    """Zero-phase filtfilt a 1-D series, restricted to contiguous finite runs.

    Runs long enough for a stable filtfilt (>= min_len, scipy's own padlen
    requirement) are filtered in place. Shorter runs are left as their
    raw/interpolated values — real data, just unsmoothed, rather than crashing
    or silently corrupting neighbouring runs with edge-artifact bleed. NaN runs
    are left as NaN.
    """
    out = x.copy()
    n = len(x)
    is_finite = np.isfinite(x)
    i = 0
    while i < n:
        if not is_finite[i]:
            i += 1
            continue
        j = i
        while j < n and is_finite[j]:
            j += 1
        if (j - i) >= min_len:
            out[i:j] = filtfilt(b, a, x[i:j])
        i = j
    return out


def apply_butterworth_filter(
    joint_arrays: Dict[str, np.ndarray],
    frame_rate: float,
    cutoff_hz: float = BUTTERWORTH_CUTOFF_HZ,
    order: int = BUTTERWORTH_ORDER,
) -> Dict[str, np.ndarray]:
    """4th-order zero-phase Butterworth low-pass filter, per joint per axis.

    Run AFTER gap-filling (Step 4.5) so short interpolated gaps get smoothed
    along with real data. Long NaN blocks that gap-fill couldn't close are left
    untouched — filtering is restricted to contiguous valid runs
    (_filter_1d_segments) so it never crashes on remaining NaN and never lets
    one run's edge artifacts bleed into another across a gap.

    Args:
        joint_arrays: {joint_name: (N, 3) ndarray}, mm, post gap-fill.
        frame_rate:   Hz, from the C3D/Nexus header — cutoff is expressed
                      relative to this, not a fixed sample assumption.
        cutoff_hz:    low-pass cutoff (default 6.0 Hz).
        order:        Butterworth filter order (default 4).

    Returns:
        New dict with the same keys, filtered values.

    Raises:
        ValueError: if cutoff_hz is at or above the Nyquist frequency for
            frame_rate (would produce an unstable/invalid filter design).
    """
    nyquist = frame_rate / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError(
            f"Butterworth cutoff {cutoff_hz} Hz must be below the Nyquist "
            f"frequency ({nyquist} Hz at {frame_rate} Hz capture rate)."
        )
    b, a = butter(order, cutoff_hz / nyquist, btype="low")
    min_len = 3 * max(len(a), len(b)) + 1   # scipy filtfilt's default padlen + 1

    filtered: Dict[str, np.ndarray] = {}
    for name, arr in joint_arrays.items():
        out = arr.copy()
        for axis in range(3):
            out[:, axis] = _filter_1d_segments(arr[:, axis], b, a, min_len)
        filtered[name] = out
    return filtered


# ── Raw Nexus CSV processing (vicon_clean.py logic) ───────────────────────────

_REQUIRED_MARKERS = [
    "LFHD","RFHD","LBHD","RBHD",
    "C7","T10",
    "LSHO","LELB","LWRA","LWRB","LFIN",
    "RSHO","RELB","RWRA","RWRB","RFIN",
    "LASI","RASI","LPSI","RPSI",
    "LKNE","RKNE","LANK","RANK","LTOE","RTOE",
]


def _parse_nexus_csv(path: Path) -> Tuple[pd.DataFrame, int]:
    """
    Parse a raw Nexus multi-section CSV (same as vicon_clean.py parse_vicon_csv).
    Returns (df, sample_rate) where df has Frame + MARKER_X/Y/Z columns.
    """
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()

    traj_idx: Optional[int] = None
    for i, line in enumerate(raw):
        if line.strip() == "Trajectories":
            traj_idx = i
            break
    if traj_idx is None:
        raise ValueError(f"'Trajectories' section not found in {path.name}")

    sample_rate = int(raw[traj_idx + 1].strip())
    name_fields = raw[traj_idx + 2].split(",")

    col_names: List[str] = ["Frame", "SubFrame"]
    j = 2
    while j < len(name_fields):
        raw_name = name_fields[j].strip()
        if ":" in raw_name:
            raw_name = raw_name.split(":")[-1]
        raw_name = raw_name.strip().replace(" ", "_") or f"_UNK{j}"
        col_names += [f"{raw_name}_X", f"{raw_name}_Y", f"{raw_name}_Z"]
        j += 3

    data_start = traj_idx + 5
    records = []
    ncols = len(col_names)
    for line in raw[data_start:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(",")
        try:
            int(parts[0])
        except (ValueError, IndexError):
            break
        if len(parts) < ncols:
            parts += [""] * (ncols - len(parts))
        records.append(parts[:ncols])

    df = pd.DataFrame(records, columns=col_names)
    df = df[df["SubFrame"].str.strip() == "0"].copy().reset_index(drop=True)
    for col in col_names:
        if col != "SubFrame":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.drop(columns=["SubFrame"], inplace=True)
    return df, sample_rate


def process_raw_nexus_csv(
    csv_path: Path,
    max_gap: int = MAX_INTERP_GAP,
    fallback_mode: bool = False,
) -> Tuple[pd.DataFrame, int]:
    """
    Process a raw Nexus multi-section CSV using vicon_clean.py logic.

    Applies: Trajectories parse → gap fill → 21-joint computation
    (HD mean, wrist midpoints, Harrington 2007 HJC).

    Args:
        csv_path:      Path to raw Nexus multi-section CSV.
        max_gap:       Max gap frames for C3D gap fill (standalone mode only).
        fallback_mode: When True, the CSV is used as a fallback source for the
                       C3D merge step.  Two key differences from standalone mode:
                         1. No first-valid-frame trimming — every frame in the
                            CSV is kept so the merge can reach any C3D frame.
                         2. Unlimited interior gap interpolation (limit=None) so
                            that any valid data surrounded by gaps is still
                            available to fill C3D NaN frames.

    Returns:
        (df, sample_rate) — df in OUTPUT_COLUMNS format, NaN where markers were
        missing; sample_rate is the Hz parsed from the CSV's own Trajectories
        header (used by _validate_fallback_csv to check it matches the C3D).
    """
    df_raw, sample_rate = _parse_nexus_csv(csv_path)
    present = [m for m in _REQUIRED_MARKERS if f"{m}_X" in df_raw.columns]

    keep = ["Frame"] + [f"{m}_{ax}" for m in present for ax in ("X","Y","Z")]
    df = df_raw[[c for c in keep if c in df_raw.columns]].copy()

    if not fallback_mode:
        # Standalone mode: trim leading frames where not all usable markers valid
        N = len(df)
        MIN_VALID = max(25, int(0.05 * N))
        perm_missing = [m for m in present
                        if df[f"{m}_X"].notna().sum() < MIN_VALID]
        usable = [m for m in present if m not in perm_missing]
        if not usable:
            raise ValueError("All required markers are permanently missing in the CSV.")
        valid_mask = pd.Series(True, index=df.index)
        for m in usable:
            valid_mask &= df[f"{m}_X"].notna()
        first_idx = int(valid_mask.idxmax()) if valid_mask.any() else 0
        df = df.loc[first_idx:].copy().reset_index(drop=True)
        gap_limit: Optional[int] = max_gap
    else:
        # Fallback mode: keep ALL frames so the merge can reach any C3D frame,
        # but cap interior interpolation at the SAME finite limit as the main
        # path. Unlimited interpolation (the previous behaviour) fabricated
        # arbitrarily long stretches of data and presented them as valid
        # Vicon-derived targets — the downstream gap caps (dataset_export
        # MAX_TARGET_GAP) cannot detect them because the gap is already filled.
        # A long dropout must stay NaN so it is correctly marked invalid, not
        # invented. (Leading/trailing NaN are never filled — only interior gaps.)
        gap_limit = max_gap  # finite; was None (unlimited) — do not restore

    def _gm(name: str) -> np.ndarray:
        if name not in present:
            return np.full((len(df), 3), np.nan)
        arr = df[[f"{name}_X", f"{name}_Y", f"{name}_Z"]].to_numpy(dtype=float)
        # Despike BEFORE gap-fill, same as the main C3D path — a single
        # implausible jump must not corrupt the Harrington pelvis regression.
        arr = despike_velocity(arr, float(sample_rate))
        return _fill_gaps(arr, gap_limit)

    HD     = np.nanmean(np.stack([_gm("LFHD"),_gm("RFHD"),_gm("LBHD"),_gm("RBHD")],axis=2),axis=2)
    C7     = _gm("C7")
    RSHO   = _gm("RSHO");  LSHO  = _gm("LSHO")
    RELB   = _gm("RELB");  LELB  = _gm("LELB")
    RWR    = (_gm("RWRA") + _gm("RWRB")) / 2.0
    LWR    = (_gm("LWRA") + _gm("LWRB")) / 2.0
    RHAND  = _gm("RFIN");  LHAND = _gm("LFIN")
    T10    = _gm("T10")
    RHIP, LHIP, PELCEN, L45 = _harrington_hjc(
        _gm("LASI"), _gm("RASI"), _gm("LPSI"), _gm("RPSI"))
    check_hip_width(RHIP, LHIP)   # same guard as the main C3D path
    RKNE   = _gm("RKNE");  LKNE  = _gm("LKNE")
    RANK   = _gm("RANK");  LANK  = _gm("LANK")
    RTOE   = _gm("RTOE");  LTOE  = _gm("LTOE")

    joint_arrays = {
        "HD": HD, "C7": C7, "RSHO": RSHO, "LSHO": LSHO,
        "RELB": RELB, "LELB": LELB, "RWR": RWR, "LWR": LWR,
        "RHAND": RHAND, "LHAND": LHAND, "T10": T10, "L45": L45,
        "PELCEN": PELCEN, "RHIP": RHIP, "LHIP": LHIP,
        "RKNE": RKNE, "LKNE": LKNE, "RANK": RANK, "LANK": LANK,
        "RTOE": RTOE, "LTOE": LTOE,
    }
    # Same Step 4.5 smoothing as the main C3D path — keeps merged output
    # consistently smoothed regardless of whether a frame came from C3D or
    # this fallback CSV (see despike_velocity()/apply_butterworth_filter()).
    joint_arrays = apply_butterworth_filter(joint_arrays, float(sample_rate))

    frames = df["Frame"].to_numpy(dtype=np.int64)
    out: dict = {"Frame": frames}
    for name, arr in joint_arrays.items():
        out[f"{name}_X"] = arr[:, 0]
        out[f"{name}_Y"] = arr[:, 1]
        out[f"{name}_Z"] = arr[:, 2]

    return pd.DataFrame(out)[OUTPUT_COLUMNS], sample_rate


# ── Fallback DataFrame merge ───────────────────────────────────────────────────

def _merge_fallback(
    df_c3d: pd.DataFrame,
    df_fallback: pd.DataFrame,
) -> tuple:
    """
    Frame-by-frame merge: C3D values take priority, NaN filled from fallback.

    Args:
        df_c3d:      C3D output DataFrame (OUTPUT_COLUMNS).
        df_fallback: Fallback DataFrame in OUTPUT_COLUMNS format (already gap-filled).

    Returns:
        (merged_df, filled_stats, unfillable_stats, recovered_mask) where:
          filled_stats     = {joint: n_frames_actually_filled_from_fallback}
          unfillable_stats = {joint: n_frames_nan_in_both_c3d_and_fallback}
          recovered_mask   = {joint: (F,) bool} — True on frames whose X value
                              was NaN going into this merge and became finite
                              from the fallback CSV. Feeds the {joint}_Source
                              4-state column in c3d_to_clean_csv (c3d /
                              interpolated / recovered_csv / invalid).
    """
    joint_names = [c[:-2] for c in OUTPUT_COLUMNS if c.endswith("_X")]

    merged = df_c3d.merge(
        df_fallback[["Frame"] + [c for c in OUTPUT_COLUMNS[1:] if c in df_fallback.columns]],
        on="Frame", how="left", suffixes=("", "_fb"),
    )

    filled_stats: dict = {}
    unfillable_stats: dict = {}
    recovered_mask: dict = {}
    for name in joint_names:
        filled = 0
        unfillable = 0
        for ax in ("_X", "_Y", "_Z"):
            col    = f"{name}{ax}"
            fb_col = f"{col}_fb"
            if fb_col in merged.columns:
                was_nan = merged[col].isna()
                fb_also_nan = merged[fb_col].isna()
                merged[col] = merged[col].fillna(merged[fb_col])
                if ax == "_X":
                    filled     = int((was_nan & ~fb_also_nan).sum())
                    unfillable = int((was_nan & fb_also_nan).sum())
                    recovered_mask[name] = (was_nan & ~fb_also_nan).to_numpy()
        if filled:
            filled_stats[name] = filled
        if unfillable:
            unfillable_stats[name] = unfillable
        for ax in ("_X", "_Y", "_Z"):
            merged.drop(columns=[f"{name}{ax}_fb"], errors="ignore", inplace=True)

    return merged[OUTPUT_COLUMNS], filled_stats, unfillable_stats, recovered_mask


def find_fully_valid_window(
    df: pd.DataFrame, x_cols: List[str], min_valid_frac: float = 0.05,
    source: str = "",
) -> Tuple[int, int]:
    """Return (first_idx, last_idx) — the row range where every consistently-
    tracked joint's X column is non-NaN, inclusive on both ends.

    A joint present in fewer than `min_valid_frac` of all rows is treated as
    effectively never placed/tracked and excluded from the "everything must
    be valid" requirement, so one permanently-missing marker doesn't make the
    whole trial untrimmable — the same carve-out
    process_raw_nexus_csv's standalone-mode trim already uses.

    Args:
        df:      DataFrame with one `{joint}_X` column per joint.
        x_cols:  the `{joint}_X` column names to check.
        source:  filename to name in a raised error, for a readable message.

    Raises:
        ValueError: if every joint is effectively missing, or no row has all
            the consistently-tracked joints valid simultaneously.
    """
    n = len(df)
    required_cols = [c for c in x_cols if df[c].notna().sum() >= min_valid_frac * n]
    if not required_cols:
        raise ValueError(
            f"{source}: every joint is present in fewer than "
            f"{min_valid_frac:.0%} of frames — nothing usable to trim."
        )
    fully_valid_rows = df[required_cols].notna().all(axis=1).to_numpy()
    if not fully_valid_rows.any():
        raise ValueError(
            f"{source}: no frame has all {len(required_cols)} "
            f"consistently-tracked joints valid at the same time — cannot "
            f"determine a usable start/end. Check for missing markers or "
            f"provide a fallback Nexus CSV."
        )
    valid_idx = np.flatnonzero(fully_valid_rows)
    return int(valid_idx[0]), int(valid_idx[-1])


def classify_joint_source(
    orig_valid: np.ndarray, interpolated: np.ndarray,
    recovered_csv: np.ndarray, final_valid: np.ndarray,
) -> np.ndarray:
    """Combine four per-frame boolean masks into one 4-state label array.

    The three input masks are causally exclusive by construction (each stage
    only acts on what the previous stage left NaN), but this does not assume
    that — `final_valid` always wins, so any future change to the pipeline
    order can't silently mislabel a frame.

    Returns:
        (F,) object array of "c3d" | "interpolated" | "recovered_csv" | "invalid".
    """
    n = len(final_valid)
    source = np.full(n, "invalid", dtype=object)
    source[orig_valid] = "c3d"
    source[interpolated & ~orig_valid] = "interpolated"
    source[recovered_csv & ~orig_valid & ~interpolated] = "recovered_csv"
    source[~final_valid] = "invalid"
    return source


def _validate_fallback_csv(
    fallback_sample_rate: int,
    c3d_frame_rate: float,
    fallback_frames: np.ndarray,
    c3d_frames: np.ndarray,
    df_fallback: pd.DataFrame,
    task_name: Optional[str] = None,
    fallback_filename: Optional[str] = None,
) -> List[str]:
    """Check the fallback CSV is plausibly THE SAME TRIAL as the C3D before
    it is allowed to fill any frame.

    Raises ValueError for mismatches that mean the two files cannot possibly
    be the same recording (different capture rate, or zero frame-number
    overlap at all) — these are facts about the data, not a naming slip, and
    merging anyway would silently blend two different trials.

    Returns a list of non-fatal warning strings for softer signals (frame
    count, coordinate magnitude, filename) that are worth a researcher's
    attention but are not, on their own, proof of a wrong file — e.g. Nexus
    exports commonly carry different pre/post-trial padding than the C3D.
    """
    warnings_out: List[str] = []

    if abs(float(fallback_sample_rate) - float(c3d_frame_rate)) > 1e-6:
        raise ValueError(
            f"Fallback CSV sample rate ({fallback_sample_rate} Hz) does not "
            f"match the C3D frame rate ({c3d_frame_rate} Hz) — refusing to "
            f"merge files recorded at different rates."
        )

    fb_lo, fb_hi = int(fallback_frames.min()), int(fallback_frames.max())
    c3d_lo, c3d_hi = int(c3d_frames.min()), int(c3d_frames.max())
    if fb_hi < c3d_lo or fb_lo > c3d_hi:
        raise ValueError(
            f"Fallback CSV frame range ({fb_lo}-{fb_hi}) does not overlap "
            f"the C3D frame range ({c3d_lo}-{c3d_hi}) at all — this looks "
            f"like a different recording, not a fallback for this trial."
        )

    fb_n, c3d_n = fb_hi - fb_lo + 1, c3d_hi - c3d_lo + 1
    if c3d_n > 0 and abs(fb_n - c3d_n) / c3d_n > 0.5:
        warnings_out.append(
            f"Fallback CSV frame count ({fb_n}) differs from the C3D "
            f"({c3d_n}) by more than 50% — verify this is the same trial."
        )

    # No explicit unit/coordinate-system field exists in a raw Nexus export,
    # so use a coarse anatomical plausibility check instead: adult marker
    # heights (Z) in mm are on the order of a few hundred to ~1900mm. Metres
    # instead of mm, or a badly wrong axis convention, misses this by ~1000x.
    z_cols = [c for c in df_fallback.columns if c.endswith("_Z")]
    if z_cols:
        z_vals = df_fallback[z_cols].to_numpy(dtype=float)
        z_finite = z_vals[np.isfinite(z_vals)]
        if z_finite.size:
            z_median = float(np.median(np.abs(z_finite)))
            if z_median < 50 or z_median > 5000:
                warnings_out.append(
                    f"Fallback CSV marker Z-magnitude looks implausible for "
                    f"millimetres (median |Z|={z_median:.0f}) — check units "
                    f"and coordinate system before trusting the merge."
                )

    # Identity — best-effort only. A raw Nexus export carries no embedded
    # subject/task/trial field, so this can only compare the uploaded
    # filename's tokens against the task name. Warn, don't block: researchers
    # don't always follow a strict file-naming convention.
    if task_name and fallback_filename:
        stem = Path(fallback_filename).stem.lower()
        task_tokens = set(re.split(r"[_\-\s]+", task_name.lower())) - {""}
        fb_tokens = set(re.split(r"[_\-\s]+", stem)) - {""}
        if task_tokens and not (task_tokens & fb_tokens):
            warnings_out.append(
                f"Fallback CSV filename ('{fallback_filename}') shares no "
                f"identifying tokens with the task name ('{task_name}') — "
                f"double check this is the correct trial's Nexus export."
            )

    return warnings_out


# ── Main pipeline ──────────────────────────────────────────────────────────────

def c3d_to_clean_csv(
    c3d_path: Path,
    out_dir: Path,
    max_gap: int = MAX_INTERP_GAP,
    raw_nexus_csv: Optional[Path] = None,
    progress_cb=None,
    out_stem: Optional[str] = None,
    cutoff_hz: float = BUTTERWORTH_CUTOFF_HZ,
    raw_nexus_csv_filename: Optional[str] = None,
) -> dict:
    """
    Read a Vicon C3D file and write a clean 21-joint CSV.

    Args:
        c3d_path:    Path to the .c3d file.
        out_dir:     Output directory.
        max_gap:     Max gap frames to fill (default 10).
        progress_cb: Optional callback(pct: int, step: str).
        raw_nexus_csv_filename: the fallback CSV's ORIGINAL uploaded filename
            (the saved path on disk is a generic "nexus_raw.csv" with no
            identifying info) — used only for the best-effort filename/task
            identity check in _validate_fallback_csv.
        cutoff_hz:   Butterworth low-pass cutoff in Hz, applied to the markers
            at the native capture rate (default 6.0, Winter's gait convention).
            Exposed as a parameter because this value is not cosmetic: it sets
            the bandwidth of the ground truth. Differentiation amplifies noise
            by ~f per derivative, so the cutoff decides what trunk angular
            VELOCITY and ACCELERATION mean downstream (see
            vicon_reba/kinematics.py). Two consequences:
              - It must stay well below the Nyquist of any later resampling
                (6 Hz << 30 Hz for the 100 Hz -> 60 Hz video-frame resample),
                which is what makes that resample alias-free.
              - Any RGB/model-derived angle series it is compared against must
                be filtered to the SAME cutoff, or the comparison is rigged.
            Vary it only for a deliberate sensitivity/ablation analysis, and
            record the value used alongside the outputs.

    Returns:
        Report dict with processing details and output path.
    """
    def _cb(pct, step):
        if progress_cb:
            progress_cb(pct, step)

    _cb(5, "Reading C3D file…")
    markers, frame_rate, n_frames, start = _load_c3d_markers(c3d_path)

    has = lambda name: name in markers and not np.all(np.isnan(markers[name]))

    # ── Detect Plug-in Gait outputs ────────────────────────────────────────────
    pig_detected = any(has(m) for m in ("RFEP", "LFEP", "RFEO", "LFEO", "RTIO", "LTIO"))
    pig_used: List[str] = []

    _cb(15, "Detecting Plug-in Gait model outputs…")

    def _get(name: str) -> np.ndarray:
        """Return (N,3) for a marker or NaN array if absent."""
        if name in markers:
            return markers[name].copy()
        return np.full((n_frames, 3), np.nan)

    # ── Build frame index ──────────────────────────────────────────────────────
    frames = np.arange(start, start + n_frames, dtype=np.int64)

    _cb(25, "Computing joint positions…")

    # ── HEAD ──────────────────────────────────────────────────────────────────
    head_stk = [_get(m) for m in ("LFHD", "RFHD", "LBHD", "RBHD")]
    HD = np.nanmean(np.stack(head_stk, axis=2), axis=2)

    # ── C7 (cervical spine) ───────────────────────────────────────────────────
    C7 = _get("C7")

    # ── SHOULDERS ─────────────────────────────────────────────────────────────
    # Verified: RHUO/LHUO are at ELBOW height (Z≈1070mm), NOT shoulder.
    # Physical acromion markers (RSHO/LSHO) are closest to GH center.
    RSHO = _get("RSHO")
    LSHO = _get("LSHO")

    # ── ELBOWS ────────────────────────────────────────────────────────────────
    # Verified: RHUO/LHUO Z≈1074/1053mm = elbow level (matches RELB≈1082/1061mm)
    # These are the distal humerus / elbow joint centres from PiG.
    if has("RHUO") and has("LHUO"):
        RELB = _get("RHUO")
        LELB = _get("LHUO")
        pig_used.append("RELB/LELB ← RHUO/LHUO (humerus distal = elbow JC, Z≈1070mm)")
    else:
        RELB = _get("RELB")
        LELB = _get("LELB")

    # ── WRISTS ────────────────────────────────────────────────────────────────
    # Verified: RRAO/LRAO Z≈849/853mm = wrist level (matches (RWRA+RWRB)/2≈850mm)
    if has("RRAO") and has("LRAO"):
        RWR = _get("RRAO")
        LWR = _get("LRAO")
        pig_used.append("RWR/LWR ← RRAO/LRAO (radius distal = wrist JC, Z≈851mm)")
    else:
        RWR = (_get("RWRA") + _get("RWRB")) / 2.0
        LWR = (_get("LWRA") + _get("LWRB")) / 2.0

    # ── HANDS ─────────────────────────────────────────────────────────────────
    # RHNO/LHNO are below RFIN/LFIN (Z≈687 vs 775mm) — keep physical finger markers.
    RHAND = _get("RFIN")
    LHAND = _get("LFIN")

    # ── THORACIC SPINE ────────────────────────────────────────────────────────
    T10 = _get("T10")

    # ── PELVIS + HIP JOINT CENTRES ────────────────────────────────────────────
    # ALWAYS Harrington from the ASIS/PSIS markers — never RFEP/LFEP, which are
    # the femur's proximal-AXIS markers and not joint centres (see the "O vs
    # A/L/P" section of the module docstring). The surface pelvis markers are
    # real and healthy (SUB01 inter-ASIS = 293.5mm), so this is a strictly
    # better source, not a fallback.
    #
    # PELO is likewise not used for PELCEN: it sits at hip level (Z≈877mm),
    # ~100mm below the physical ASIS/PSIS centroid (Z≈977mm).
    RHIP, LHIP, PELCEN, L45 = _harrington_hjc(
        _get("LASI"), _get("RASI"), _get("LPSI"), _get("RPSI"))
    hip_width_mm = check_hip_width(RHIP, LHIP)

    # ── KNEES ─────────────────────────────────────────────────────────────────
    # Verified: RFEO/LFEO Z≈444/425mm = knee joint centre level (matches RKNE≈456/435mm)
    if has("RFEO") and has("LFEO"):
        RKNE = _get("RFEO")
        LKNE = _get("LFEO")
        pig_used.append("RKNE/LKNE ← RFEO/LFEO (distal femur = true KJC, Z≈435mm)")
    else:
        RKNE = _get("RKNE")
        LKNE = _get("LKNE")

    # ── ANKLES ────────────────────────────────────────────────────────────────
    # Verified: RTIO/LTIO Z≈75/72mm = ankle level (exact match with RANK≈75mm)
    if has("RTIO") and has("LTIO"):
        RANK = _get("RTIO")
        LANK = _get("LTIO")
        pig_used.append("RANK/LANK ← RTIO/LTIO (tibia distal = true AJC, Z≈74mm)")
    else:
        RANK = _get("RANK")
        LANK = _get("LANK")

    # ── TOES ──────────────────────────────────────────────────────────────────
    RTOE = _get("RTOE")
    LTOE = _get("LTOE")

    _cb(45, "Filling gaps…")

    # ── Gap filling ────────────────────────────────────────────────────────────
    joint_arrays = {
        "HD": HD, "C7": C7,
        "RSHO": RSHO, "LSHO": LSHO,
        "RELB": RELB, "LELB": LELB,
        "RWR":  RWR,  "LWR":  LWR,
        "RHAND": RHAND, "LHAND": LHAND,
        "T10": T10, "L45": L45,
        "PELCEN": PELCEN,
        "RHIP": RHIP, "LHIP": LHIP,
        "RKNE": RKNE, "LKNE": LKNE,
        "RANK": RANK, "LANK": LANK,
        "RTOE": RTOE, "LTOE": LTOE,
    }
    # Snapshot per-joint validity BEFORE gap-fill (post-despike, since despike
    # runs inside _load_c3d_markers): this is what "c3d" means in the 4-state
    # {joint}_Source column written below. Anything still NaN here that
    # becomes finite after _fill_gaps() is "interpolated"; anything still NaN
    # after that but finite once _merge_fallback() runs is "recovered_csv".
    c3d_original_valid = {name: ~np.isnan(arr[:, 0]) for name, arr in joint_arrays.items()}

    for name, arr in joint_arrays.items():
        joint_arrays[name] = _fill_gaps(arr, max_gap)

    interpolated_by_gapfill = {
        name: (~c3d_original_valid[name]) & (~np.isnan(arr[:, 0]))
        for name, arr in joint_arrays.items()
    }

    _cb(55, "Smoothing (Butterworth low-pass)…")

    # ── Step 4.5: zero-phase low-pass filter ──────────────────────────────────
    # Applied AFTER gap-filling so short interpolated gaps get smoothed along
    # with real data. Restricted to contiguous valid runs (see
    # apply_butterworth_filter / _filter_1d_segments) so long unfilled NaN
    # blocks — now more common since Step 5 no longer truncates them away —
    # never crash filtfilt or bleed edge artifacts into neighbouring runs.
    joint_arrays = apply_butterworth_filter(joint_arrays, frame_rate,
                                            cutoff_hz=cutoff_hz)

    # The clean CSV is trimmed to the fully-valid window further down, AFTER
    # the fallback-CSV merge (a joint missing here may still be recovered
    # from the fallback before that boundary is decided). frames_trimmed
    # keeps its name for the diff below but nothing is cut yet at this point.
    frames_trimmed = frames
    n_out = len(frames_trimmed)

    # ── Compute time_s relative to Vicon capture start ────────────────────────
    # time_s[i] = (frame[i] - ACTUAL_START_FIELD) / frame_rate
    #
    # This is ALWAYS anchored to the absolute recording start (frame ==
    # ACTUAL_START_FIELD -> time_s == 0), NEVER re-based to "row 0 of this
    # file". That anchor is load-bearing, not cosmetic: sync_pipeline.py's
    # wall-clock offset and overlay_pipeline.py's video<->Vicon time mapping
    # both assume time_s measures seconds since the SAME fixed point (XCP
    # START_TIME) in every synced_60hz.csv this project has ever produced —
    # re-zeroing it at the trim boundary below would silently decouple this
    # file's clock from that shared reference and corrupt every downstream
    # timestamp comparison. Since the trim below cuts the leading portion,
    # row 0 of the OUTPUT will show a small positive time_s (e.g. 0.30s),
    # not 0.0 — that is correct, not a bug (see "first_valid_time_s" in the
    # returned report, which documents this explicitly).
    time_s_arr = (frames_trimmed - start) / frame_rate  # start = ACTUAL_START_FIELD

    _cb(75, "Building output CSV…")

    # ── Assemble output DataFrame ──────────────────────────────────────────────
    out_data: dict = {"Frame": frames_trimmed}
    for name, arr in joint_arrays.items():
        out_data[f"{name}_X"] = arr[:, 0]
        out_data[f"{name}_Y"] = arr[:, 1]
        out_data[f"{name}_Z"] = arr[:, 2]

    df_out = pd.DataFrame(out_data)[OUTPUT_COLUMNS]
    # Insert time_s immediately after Frame — do NOT add to OUTPUT_COLUMNS so
    # that process_raw_nexus_csv (fallback path) doesn't need to provide it.
    df_out.insert(1, "time_s", time_s_arr)

    # ── Process raw Nexus CSV and merge (frame-by-frame) ─────────────────────
    # fallback_mode=True: keep all CSV frames + unlimited interior gap fill so
    # every valid CSV measurement is available to cover C3D NaN frames.
    fallback_filled: dict = {}
    fallback_unfillable: dict = {}
    fallback_recovered_mask: dict = {}
    fallback_warnings: List[str] = []
    if raw_nexus_csv is not None and Path(raw_nexus_csv).exists():
        _cb(78, "Running Nexus CSV pipeline (fallback mode)…")
        df_fallback, fallback_sample_rate = process_raw_nexus_csv(
            Path(raw_nexus_csv), max_gap, fallback_mode=True)

        _cb(82, "Validating fallback CSV against the C3D…")
        fallback_warnings = _validate_fallback_csv(
            fallback_sample_rate, frame_rate,
            df_fallback["Frame"].to_numpy(), frames,
            df_fallback, task_name=out_stem, fallback_filename=raw_nexus_csv_filename,
        )

        _cb(85, "Merging C3D + Nexus CSV…")
        df_out, fallback_filled, fallback_unfillable, fallback_recovered_mask = \
            _merge_fallback(df_out, df_fallback)
        # _merge_fallback returns OUTPUT_COLUMNS only — re-attach time_s
        df_out.insert(1, "time_s", time_s_arr)

    _cb(86, "Computing per-joint source classification…")

    # ── 4-state per-joint source mask — computed AFTER the fallback merge ────
    # Replaces the old single {joint}_Valid boolean, which collapsed three
    # very different origins (real C3D measurement, short-gap linear
    # interpolation, recovered from the fallback Nexus CSV) into one
    # "present" flag. A downstream consumer needs to know WHICH of these
    # produced a value — e.g. to exclude interpolated/recovered frames from a
    # "measured accuracy" claim while still using them for training.
    # "invalid" = none of the three; the frame stays NaN in X/Y/Z.
    # Appended after OUTPUT_COLUMNS/time_s rather than inserted into
    # OUTPUT_COLUMNS itself, so _merge_fallback's exact-column-list merge
    # above is unaffected by this addition.
    joint_names_for_mask = [c[:-2] for c in OUTPUT_COLUMNS if c.endswith("_X")]
    for name in joint_names_for_mask:
        final_valid = df_out[f"{name}_X"].notna().to_numpy()
        orig_valid  = c3d_original_valid.get(name, np.zeros(n_out, dtype=bool))
        interp      = interpolated_by_gapfill.get(name, np.zeros(n_out, dtype=bool))
        recovered   = fallback_recovered_mask.get(name, np.zeros(n_out, dtype=bool))
        df_out[f"{name}_Source"] = classify_joint_source(orig_valid, interp, recovered, final_valid)

    _cb(87, "Trimming to the fully-valid window…")

    # ── Trim to the fully-valid window ────────────────────────────────────────
    # The clean CSV starts at the first frame — and ends at the last frame —
    # where every joint has a value in the FINAL output (c3d, interpolated, or
    # recovered_csv; never "invalid"). This is the single canonical Vicon
    # source every downstream stage (Sync, REBA, posture duration, trunk
    # dynamics, training targets) reads directly, so its own boundary must
    # already be clean rather than making every consumer re-derive "was this
    # frame actually usable" for itself.
    #
    # Interior gaps are NOT touched here: a joint dropping out mid-trial
    # after this window starts doesn't fragment the frame index — it shows up
    # in {joint}_Source/{joint}_X for that joint at that frame only.
    x_cols_out = [f"{n}_X" for n in joint_names_for_mask]
    trim_lo, trim_hi = find_fully_valid_window(df_out, x_cols_out, source=c3d_path.name)
    n_trimmed_leading  = trim_lo
    n_trimmed_trailing = n_out - 1 - trim_hi
    df_out = df_out.iloc[trim_lo:trim_hi + 1].reset_index(drop=True)
    n_out = len(df_out)

    _cb(88, "Writing CSV…")

    # ── Write output ───────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    # out_stem overrides the C3D filename so the CSV is named after the task.
    stem     = out_stem.replace(" ", "_") if out_stem else c3d_path.stem.replace(" ", "_")
    out_path = out_dir / f"{stem}_clean.csv"
    df_out.to_csv(out_path, index=False, float_format="%.6f")

    _cb(100, "Done")

    # ── Per-joint NaN statistics and origin classification ────────────────────
    joint_names_out = [c[:-2] for c in OUTPUT_COLUMNS if c.endswith("_X")]
    nan_pcts: dict = {}
    nan_origin: dict = {}   # "ok" | "c3d_only" | "both" | "partial"

    for name in joint_names_out:
        col = f"{name}_X"
        arr = df_out[[col]].to_numpy()
        n_nan = int(np.isnan(arr[:, 0]).sum())
        pct   = round(100.0 * n_nan / max(n_out, 1), 2)
        nan_pcts[name] = pct

        if n_nan == 0:
            nan_origin[name] = "ok"
        elif raw_nexus_csv is not None:
            # Determine if CSV could have helped
            if name in fallback_filled:
                # partially filled — some frames came from CSV, rest still NaN
                nan_origin[name] = "partial"
            elif name in fallback_unfillable and n_nan > 0:
                # CSV was present but also had NaN for those frames → both missing
                nan_origin[name] = "both"
            else:
                # C3D had NaN, CSV had nothing to offer → absent from both
                nan_origin[name] = "both"
        else:
            # No CSV provided
            nan_origin[name] = "c3d_only"

    permanently_nan = [n for n, o in nan_origin.items() if nan_pcts[n] == 100.0]
    partially_nan   = [n for n, o in nan_origin.items()
                       if 0 < nan_pcts[n] < 100.0]

    # first_valid_frame/first_valid_time_s/last_valid_time_s now describe the
    # ACTUAL trim boundary written to the CSV (see "Trim to the fully-valid
    # window" above) — total_c3d_frames stays the RAW C3D frame count so the
    # difference against output_frames shows how much was cut.
    #
    # time_s_reference spells out, in the report itself, that time_s is NOT
    # re-zeroed at the trim boundary — first_valid_time_s being > 0 is
    # expected whenever leading frames were cut, not a sign anything is wrong.
    return {
        "output_path":            str(out_path),
        "output_filename":        out_path.name,
        "frame_rate_hz":          frame_rate,
        "total_c3d_frames":       n_frames,
        "actual_start_field":     int(start),
        "time_s_reference": (
            "time_s = (Frame - actual_start_field) / frame_rate_hz — anchored "
            "to the raw recording's absolute start, NOT to row 0 of this "
            "trimmed CSV. Row 0's time_s equals first_valid_time_s below, "
            "which is > 0 whenever leading frames were trimmed. Sync/Overlay "
            "require this fixed anchor; do not re-zero it."
        ),
        "first_valid_frame":      int(frames[trim_lo]),
        "first_valid_time_s":     float(time_s_arr[trim_lo]),
        "last_valid_time_s":      float(time_s_arr[trim_hi]),
        "trimmed_leading_frames": n_trimmed_leading,
        "trimmed_trailing_frames": n_trimmed_trailing,
        "output_frames":          n_out,
        "pig_detected":           pig_detected,
        "pig_outputs_used":       pig_used,
        "nan_pct_per_joint":      nan_pcts,
        "nan_origin_per_joint":   nan_origin,
        "permanently_nan_joints": permanently_nan,
        "partially_nan_joints":   partially_nan,
        "max_gap_filled":         max_gap,
        # Recorded because it defines the ground-truth bandwidth every
        # downstream velocity/acceleration inherits — an unrecorded filter
        # setting makes the derived kinematics unreproducible.
        "butterworth_cutoff_hz":  cutoff_hz,
        "butterworth_order":      BUTTERWORTH_ORDER,
        # Derived hip width (mm), validated by check_hip_width(). Recorded so a
        # per-subject anatomical outlier is visible in the report rather than
        # only surfacing as a strange trunk angle much further downstream.
        "hip_width_mm":           round(hip_width_mm, 1),
        "fallback_csv_used":          raw_nexus_csv is not None,
        "fallback_frames_filled":     fallback_filled,
        "fallback_frames_unfillable": fallback_unfillable,
        # Non-fatal cross-checks from _validate_fallback_csv (sample rate and
        # frame-range overlap are hard errors raised before this point, so
        # any trial that reaches here already passed those).
        "fallback_validation_warnings": fallback_warnings,
    }


# ── Standalone CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json
    # ASCII arrow: a literal "→" here makes --help raise UnicodeEncodeError on a
    # Windows cp1252 console, which is where this CLI actually gets run.
    p = argparse.ArgumentParser(
        description="Vicon C3D -> clean 21-joint CSV (same format as vicon_clean.py)")
    p.add_argument("--input",   type=Path, required=True, help="Input .c3d file")
    p.add_argument("--out_dir", type=Path, default=None,  help="Output directory")
    p.add_argument("--max_gap", type=int,  default=10,    help="Max gap to fill (frames)")
    p.add_argument("--cutoff_hz", type=float, default=BUTTERWORTH_CUTOFF_HZ,
                   help=("Butterworth low-pass cutoff (Hz) applied at the native "
                         "capture rate. Sets the ground-truth bandwidth inherited "
                         "by all downstream velocity/acceleration; vary only for "
                         "a deliberate sensitivity analysis."))
    args = p.parse_args()

    out = args.out_dir or (args.input.parent / "cleaned")
    report = c3d_to_clean_csv(args.input, out, args.max_gap,
                              cutoff_hz=args.cutoff_hz)
    print(json.dumps(report, indent=2))
