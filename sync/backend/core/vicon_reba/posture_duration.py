"""High-risk posture-duration assessment.

Musculoskeletal-disorder risk depends not only on posture severity but on how long an awkward
posture is sustained. This module finds, per body segment, contiguous
high-risk bouts using HYSTERESIS (separate enter/exit debounce windows so
single-frame detector noise cannot fragment one continuous posture into
several short events, or truncate a real bout early), then compares each
bout's mean inclination against the ISO 11226:2000 angle-dependent
holding-time limit for that segment.

Expects pre-smoothed angle series (e.g. kinematics.lowpass_series) as input -- this module does not
re-smooth; that is the caller's responsibility, same separation used
elsewhere in this package (angles.py computes angles, kinematics.py
differentiates, this module only classifies/aggregates duration).

Two design choices are made explicit here:

1. Per-segment "non-neutral" trigger angle for entering a bout is set equal
   to that segment's ISO 11226 lower holding-time anchor (20 deg trunk/
   upper-arm, 25 deg neck) -- i.e. a bout only starts counting once the
   posture has crossed into ISO's duration-limited-or-worse zone, rather
   than an arbitrary separate threshold.
2. Hysteresis enter/exit windows are specified in SECONDS and converted to frame counts via fps,
   not hardcoded frame counts.

ISO 11226:2000 angle-dependent holding-time limits L_s(theta) -- linear
between the two published anchor points per segment (Figs. 4/7/10):
  trunk:      ~4 min @ 20 deg  ->  ~1 min @ 60 deg inclination
  neck:       ~8 min @ 25 deg  ->  ~1 min @ 85 deg head inclination
  upper_arm:  ~4 min @ 20 deg  ->  ~1 min @ 60 deg elevation
Below the lower anchor: ISO classifies this "acceptable", no duration
criterion at all (returned as None = unlimited). Strictly beyond the upper
anchor (the anchor itself is still on the duration-limited curve):
"not recommended" regardless of duration (returned as 0.0 seconds -- any
observed duration exceeds it).

Known approximation: the trunk vector used here is pelvis-center -> shoulder-midpoint;
ISO's trunk line is greater-trochanter(hip) -> C7. Close but not identical --
state it as an approximation, or switch angles.py's trunk vector to match ISO
exactly, before citing this module's output as ISO-11226-compliant.
"""
from __future__ import annotations

import numpy as np

_ISO_CURVES = {
    "trunk":     {"lo_deg": 20.0, "lo_s": 240.0, "hi_deg": 60.0, "hi_s": 60.0},
    "neck":      {"lo_deg": 25.0, "lo_s": 480.0, "hi_deg": 85.0, "hi_s": 60.0},
    "upper_arm": {"lo_deg": 20.0, "lo_s": 240.0, "hi_deg": 60.0, "hi_s": 60.0},
}


def holding_time_limit_s(segment: str, mean_angle_deg: float) -> float | None:
    """ISO 11226 acceptable holding-time limit L_s(theta), in seconds.

    Returns None below the segment's lower anchor (ISO: "acceptable", no
    duration criterion -- unlimited, not merely generous). Returns 0.0
    strictly beyond the upper anchor ("not recommended" regardless of
    duration; the anchor itself still sits on the duration-limited curve).
    Linearly interpolated between the two anchors otherwise.
    """
    if segment not in _ISO_CURVES:
        raise KeyError(f"No ISO 11226 holding-time curve for segment {segment!r}; "
                        f"available: {sorted(_ISO_CURVES)}")
    c = _ISO_CURVES[segment]
    theta = abs(mean_angle_deg)
    if theta < c["lo_deg"]:
        return None
    if theta > c["hi_deg"]:
        return 0.0
    frac = (theta - c["lo_deg"]) / (c["hi_deg"] - c["lo_deg"])
    return c["lo_s"] + frac * (c["hi_s"] - c["lo_s"])


def hysteresis_bouts(
    high_risk: np.ndarray,
    enter_frames: int,
    exit_frames: int,
) -> list[tuple[int, int]]:
    """Debounced high-risk bout detection.

    A bout is CONFIRMED (and back-dated to its true first frame) only after
    `enter_frames` consecutive True frames, and CLOSED (at its true last
    high-risk frame, not the trailing confirmation frames) only after
    `exit_frames` consecutive False frames following it. Brief dips shorter
    than `exit_frames` do not fragment an otherwise-sustained bout.

    Args:
        high_risk: (F,) bool array, True where the frame is high-risk.
        enter_frames: consecutive high-risk frames required to open a bout (>=1).
        exit_frames: consecutive low-risk frames required to close a bout (>=1).

    Returns:
        List of (start_idx, end_idx) INCLUSIVE index pairs, in original
        frame indexing.
    """
    if enter_frames < 1 or exit_frames < 1:
        raise ValueError("enter_frames and exit_frames must both be >= 1")

    bouts: list[tuple[int, int]] = []
    confirmed = False
    hot_run = 0
    cold_run = 0
    candidate_start: int | None = None
    bout_start: int | None = None
    last_hot_idx: int | None = None

    for i, h in enumerate(np.asarray(high_risk, dtype=bool)):
        if not confirmed:
            if h:
                if hot_run == 0:
                    candidate_start = i
                hot_run += 1
                if hot_run >= enter_frames:
                    confirmed = True
                    bout_start = candidate_start
                    last_hot_idx = i
                    cold_run = 0
            else:
                hot_run = 0
                candidate_start = None
        else:
            if h:
                last_hot_idx = i
                cold_run = 0
            else:
                cold_run += 1
                if cold_run >= exit_frames:
                    bouts.append((bout_start, last_hot_idx))
                    confirmed = False
                    hot_run = 0
                    candidate_start = None
                    bout_start = None

    if confirmed:
        bouts.append((bout_start, last_hot_idx))
    return bouts


def angle_status(segment: str, mean_angle_deg: float) -> str:
    """ISO 11226 ANGLE verdict for a posture, independent of how long it is held.

    "acceptable"       -- below the segment's lower anchor: ISO applies no
                          duration criterion at all.
    "duration_limited" -- between the anchors: acceptability depends on how long
                          the posture is held (that is what holding_time_limit_s
                          quantifies).
    "not_recommended"  -- beyond the upper anchor: ISO rejects the posture
                          REGARDLESS of duration. This is an ANGLE verdict and
                          must never be reported as a holding-time exceedance.
    """
    c = _ISO_CURVES[segment]
    theta = abs(mean_angle_deg)
    if theta < c["lo_deg"]:
        return "acceptable"
    if theta > c["hi_deg"]:
        return "not_recommended"
    return "duration_limited"


# ── Entry / exit thresholds, derived from REBA segment categories ────────────
#
# Per-segment tuned: each segment's ENTRY threshold is the angle at which that
# segment's REBA base score first reaches its substantially-awkward category —
# score >= 3 where the scale has four levels (trunk, upper arm), or the top
# category (2) where the scale has only two (neck). Boundaries are read off the
# same reba.py tables the severity channel uses, so the duration layer and the
# severity layer cannot drift apart.
#
#   trunk      REBA trunk_base:      >20 deg -> 3 (moderate flexion)   -> entry 20
#   neck       REBA neck_base:       >20 deg -> 2 (top category)       -> entry 20
#   upper_arm  REBA upper_arm_base:  >45 deg -> 3 (45-90 elevation)    -> entry 45
#
# The EXIT threshold sits a fixed band below entry (Schmitt trigger) so a signal
# hovering at the boundary cannot chatter between states. The band is an
# ALGORITHM parameter, pre-specified before analysis and subject to sensitivity
# analysis — it is NOT an ergonomic threshold and is not claimed to be validated.
HYSTERESIS_BAND_DEG = 5.0

REBA_ENTRY_DEG = {
    "trunk":     20.0,   # REBA trunk score 3 boundary
    "neck":      20.0,   # REBA neck score 2 boundary
    "upper_arm": 45.0,   # REBA upper-arm score 3 boundary
}
REBA_CATEGORY_NOTE = {
    "trunk":     "REBA trunk_base >20 deg -> score 3 (moderate flexion)",
    "neck":      "REBA neck_base >20 deg -> score 2 (top category)",
    "upper_arm": "REBA upper_arm_base >45 deg -> score 3 (45-90 deg elevation)",
}


def entry_exit_thresholds(segment: str,
                          entry_deg: float | None = None,
                          band_deg: float = HYSTERESIS_BAND_DEG,
                          ) -> tuple[float, float]:
    """(entry, exit) angle thresholds for a segment, in degrees.

    Exit is always strictly below entry; a caller-supplied entry overrides the
    REBA default (for sensitivity analysis), never silently.
    """
    if segment not in REBA_ENTRY_DEG:
        raise KeyError(f"Unknown segment {segment!r}; available: {sorted(REBA_ENTRY_DEG)}")
    entry = REBA_ENTRY_DEG[segment] if entry_deg is None else float(entry_deg)
    return entry, max(entry - abs(band_deg), 0.0)


def schmitt_state(angle_deg: np.ndarray, entry_deg: float, exit_deg: float) -> np.ndarray:
    """Dual-threshold (Schmitt) high-risk state.

    Enters the high state at |angle| >= entry and leaves it only at
    |angle| < exit, so a signal hovering at a single boundary cannot chatter.
    NaN (occluded) is always low and never bridges a bout. This is the ANGLE
    hysteresis; the separate d_on/d_off frame gating is the TIME hysteresis.
    """
    if exit_deg > entry_deg:
        raise ValueError(f"exit threshold {exit_deg} must not exceed entry {entry_deg}")
    a = np.abs(np.asarray(angle_deg, dtype=float))
    state = np.zeros(a.shape, dtype=bool)
    hot = False
    for i, v in enumerate(a):
        if not np.isfinite(v):
            hot = False
        elif hot:
            hot = v >= exit_deg
        else:
            hot = v >= entry_deg
        state[i] = hot
    return state


# ── Conditional static-hold branch ───────────────────────────────────────────

# A bout must last longer than this to be eligible for static classification.
# Fixed a priori by the study protocol, not fitted to results.
STATIC_MIN_DURATION_S = 4.0

# ISO 11226 is scoped to postures under NO OR MINIMAL external force. The REBA
# load/force score is the entered task metadata that encodes this, so a bout is
# force-eligible only at load score 0. Never inferred from pose.
ISO_MAX_LOAD_SCORE = 0


def classify_static(duration_s: float,
                    angle_window: np.ndarray,
                    vel_window: np.ndarray | None,
                    eps_omega: float | None,
                    eps_theta: float | None) -> tuple[bool | None, dict]:
    """S_s,i — is this bout a STATIC high-risk hold?

    S = 1 iff duration > STATIC_MIN_DURATION_S
            AND median|omega| < eps_omega
            AND SD(theta)    < eps_theta

    Returns (S, detail). S is None — "not evaluated" — when the low-motion
    thresholds have not been derived and frozen, or the velocity series is
    absent. It is NEVER guessed: an unset threshold must not silently become a
    static claim. Criterion 5 (instrument applicability / external force) is
    evaluated by the caller, which holds the task metadata.
    """
    ang = np.asarray(angle_window, dtype=float)
    ang = ang[np.isfinite(ang)]
    sd_theta = float(np.std(ang)) if ang.size else float("nan")
    med_omega = float("nan")
    if vel_window is not None:
        v = np.abs(np.asarray(vel_window, dtype=float))
        v = v[np.isfinite(v)]
        if v.size:
            med_omega = float(np.median(v))

    detail = {
        "duration_criterion_s": STATIC_MIN_DURATION_S,
        "duration_ok": bool(duration_s > STATIC_MIN_DURATION_S),
        "median_abs_velocity_deg_s": (None if not np.isfinite(med_omega)
                                      else round(med_omega, 3)),
        "sd_angle_deg": None if not np.isfinite(sd_theta) else round(sd_theta, 3),
        "eps_omega_deg_s": eps_omega,
        "eps_theta_deg": eps_theta,
    }
    if eps_omega is None or eps_theta is None or not np.isfinite(med_omega) \
            or not np.isfinite(sd_theta):
        detail["reason"] = (
            "low-motion thresholds not frozen (or velocity unavailable) — static "
            "classification not evaluated. Derive eps via "
            "estimate_static_thresholds() on train-pool static calibration trials.")
        return None, detail

    detail["velocity_ok"] = bool(med_omega < eps_omega)
    detail["variability_ok"] = bool(sd_theta < eps_theta)
    S = bool(detail["duration_ok"] and detail["velocity_ok"] and detail["variability_ok"])
    return S, detail


def holding_time_ratio(segment: str, mean_angle_deg: float,
                       duration_s: float) -> tuple[float | None, str]:
    """R_hold = H / L_s(theta_bar) for an ISO-applicable STATIC bout.

    Returns (ratio, interpretation). ratio is None when ISO defines no finite
    limit for that angle:
      - below the lower anchor  -> "no_duration_criterion" (ISO: acceptable)
      - beyond the upper anchor -> "angle_not_recommended"  (rejected at ANY
        duration — an ANGLE verdict, never reported as a duration exceedance)
    """
    limit = holding_time_limit_s(segment, mean_angle_deg)
    if limit is None:
        return None, "no_duration_criterion"
    if limit == 0.0:
        return None, "angle_not_recommended"
    ratio = duration_s / limit
    return round(ratio, 4), ("limit_exceeded" if ratio > 1.0 else "within_limit")


# ── Freezing the low-motion thresholds ───────────────────────────────────────

def estimate_static_thresholds(calibration_angles: dict[str, list[np.ndarray]],
                               fps: float,
                               percentile: float = 95.0,
                               cutoff_hz: float | None = None) -> dict:
    """Derive eps_omega / eps_theta from STATIC CALIBRATION trials.

    Args:
        calibration_angles: {segment: [angle series per train-pool trial]} —
            taken ONLY from the static standing/calibration trial of subjects in
            the training/validation pool. Passing a test-pool subject here is a
            leakage error the caller must prevent; this function cannot detect it.
        fps: sample rate of those series.
        percentile: upper percentile of the observed static distribution kept as
            the threshold. 95 by default — pre-specified, and the parameter to
            vary in the sensitivity analysis.
        cutoff_hz: passed to the same kinematics path the dynamics channel uses,
            so a "static" verdict cannot disagree with Channel 3. None for the
            Vicon path (markers already 6 Hz zero-phase filtered upstream).

    Returns a dict ready to freeze to JSON, recording the provenance of every
    number so a reader can tell what the threshold was estimated from.
    """
    from . import kinematics as _kin

    out: dict = {"fps": round(float(fps), 4), "percentile": percentile,
                 "cutoff_hz": cutoff_hz, "segments": {}}
    for seg, series_list in calibration_angles.items():
        med_v, sds, n_frames = [], [], 0
        for series in series_list:
            s = np.asarray(series, dtype=float)
            if s.size < 3:
                continue
            kin = _kin.compute_kinematics({"x_deg": s}, fs=fps, cutoff_hz=cutoff_hz,
                                          channels=("x_deg",))
            v = np.abs(kin["x_vel_deg_s"])
            v = v[np.isfinite(v)]
            fin = s[np.isfinite(s)]
            if v.size:
                med_v.append(float(np.median(v)))
            if fin.size:
                sds.append(float(np.std(fin)))
            n_frames += int(fin.size)
        if not med_v or not sds:
            continue
        out["segments"][seg] = {
            "eps_omega_deg_s": round(float(np.percentile(med_v, percentile)), 4),
            "eps_theta_deg": round(float(np.percentile(sds, percentile)), 4),
            "n_trials": len(med_v),
            "n_frames": n_frames,
        }
    return out


def load_frozen_thresholds(path) -> dict[str, tuple[float, float]]:
    """Read a frozen threshold file -> {segment: (eps_omega, eps_theta)}.

    Returns {} when the file does not exist, so the static branch reports
    "not evaluated" rather than falling back to an invented default.
    """
    import json
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return {}
    doc = json.loads(p.read_text(encoding="utf-8"))
    return {seg: (v["eps_omega_deg_s"], v["eps_theta_deg"])
            for seg, v in doc.get("segments", {}).items()}


def segment_duration_bouts(
    segment: str,
    angle_deg: np.ndarray,
    fps: float,
    *,
    angle_threshold_deg: float | None = None,
    exit_threshold_deg: float | None = None,
    enter_s: float = 0.5,
    exit_s: float = 0.5,
    angular_vel_deg_s: np.ndarray | None = None,
    eps_omega: float | None = None,
    eps_theta: float | None = None,
    load_score: int | None = None,
) -> dict:
    """Within-Task High-Risk Posture-Duration Assessment for ONE segment.

    PRIMARY OUTPUT (always produced, instrument-independent): the set of
    above-threshold bouts B_s,i = [t_on, t_off] with durations H_s,i, plus
    T_s^high (total), the bout count, the longest bout, and P_s^high (percentage
    of observed task time). These are TEMPORAL EXPOSURE DESCRIPTORS, not a new
    ergonomic risk score.

    SECONDARY, CONDITIONAL: each bout is additionally tested for static-hold
    status S_s,i (duration > 4 s AND low median angular velocity AND low angular
    variability). Only a bout with S = 1 AND minimal external force is eligible
    for the ISO 11226 holding-time ratio R_hold = H / L(theta_bar).

    A dynamic transition that crosses the posture threshold is a legitimate
    bout. It is NOT thereby a sustained static load — that is what S_s,i decides.

    Args:
        segment: "trunk", "neck", or "upper_arm".
        angle_deg: (F,) pre-smoothed SIGNED angle series; NaN = occluded.
        fps: sample rate, for converting d_on/d_off to frames.
        angle_threshold_deg: entry threshold; defaults to the segment's REBA
            category boundary (see REBA_ENTRY_DEG).
        exit_threshold_deg: exit threshold; defaults to entry - HYSTERESIS_BAND_DEG.
        enter_s / exit_s: d_on / d_off minimum durations. Onset and offset are
            BACK-DATED to the first and final qualifying frames.
        angular_vel_deg_s: |omega| series for the static criterion.
        eps_omega / eps_theta: frozen low-motion thresholds. Absent -> static
            status is "not evaluated", never assumed.
        load_score: entered REBA load/force score. Criterion 5 — ISO applies
            only under no or minimal external force. Absent -> unknown, and the
            ISO ratio is withheld.
    """
    entry, exit_thr = entry_exit_thresholds(segment, angle_threshold_deg)
    if exit_threshold_deg is not None:
        exit_thr = float(exit_threshold_deg)
        if exit_thr > entry:
            raise ValueError(f"exit threshold {exit_thr} must not exceed entry {entry}")

    angle_deg = np.asarray(angle_deg, dtype=float)
    state = schmitt_state(angle_deg, entry, exit_thr)

    enter_frames = max(1, round(enter_s * fps))
    exit_frames = max(1, round(exit_s * fps))
    dt = 1.0 / fps

    force_ok = None if load_score is None else bool(load_score <= ISO_MAX_LOAD_SCORE)

    bouts_out = []
    n_static = 0
    n_iso_exceeded = 0
    for start, end in hysteresis_bouts(state, enter_frames, exit_frames):
        window = angle_deg[start:end + 1]
        finite = window[np.isfinite(window)]
        if finite.size == 0:
            continue
        mean_abs = float(np.mean(np.abs(finite)))
        peak_abs = float(np.max(np.abs(finite)))
        duration_s = (end - start + 1) * dt

        vel_window = (None if angular_vel_deg_s is None
                      else np.asarray(angular_vel_deg_s, dtype=float)[start:end + 1])
        S, static_detail = classify_static(duration_s, window, vel_window,
                                           eps_omega, eps_theta)
        if S:
            n_static += 1

        # ISO ratio: static AND minimal external force. Both must hold; either
        # unknown means the ratio is withheld rather than assumed.
        r_hold, r_interp = None, "not_applicable"
        if S is None:
            r_interp = "static_status_not_evaluated"
        elif not S:
            r_interp = "not_static"
        elif force_ok is None:
            r_interp = "external_force_unknown"
        elif not force_ok:
            r_interp = "external_force_exceeds_iso_scope"
        else:
            r_hold, r_interp = holding_time_ratio(segment, mean_abs, duration_s)
            if r_interp == "limit_exceeded":
                n_iso_exceeded += 1

        bouts_out.append({
            "onset_frame": int(start),
            "offset_frame": int(end),
            "onset_s": round(start * dt, 3),
            "offset_s": round((end + 1) * dt, 3),
            "duration_s": round(duration_s, 3),
            "mean_angle_deg": round(mean_abs, 2),
            "peak_angle_deg": round(peak_abs, 2),
            "is_static_hold": S,
            "static_detail": static_detail,
            "iso_holding_time_ratio": r_hold,
            "iso_interpretation": r_interp,
        })

    n_bouts = len(bouts_out)
    total_s = round(sum(b["duration_s"] for b in bouts_out), 3)
    n_frames = int(angle_deg.size)
    task_s = n_frames * dt

    # ── Exposure partition of the observed interval ──────────────────────────
    # THREE-WAY, not two-way. An occluded frame is neither hazardous nor
    # acceptable — it is UNMEASURED, and folding it into "acceptable posture
    # time" would assert a safe posture that was never observed. Computing
    # acceptable = observed - hazardous does exactly that.
    measured = np.isfinite(angle_deg)
    n_measured = int(measured.sum())
    n_unmeasured = n_frames - n_measured
    measured_s = n_measured * dt
    unmeasured_s = n_unmeasured * dt
    acceptable_s = max(measured_s - total_s, 0.0)

    # Raw frame-wise count for comparison. This is NOT the same as the bout
    # total: hysteresis and d_off deliberately hold a bout open across a brief
    # dip below threshold, so hazardous_exposure_s >= above_threshold_frame_s.
    # Reporting both keeps that difference visible instead of arguable.
    n_above = int(np.count_nonzero(state & measured))

    def _pct(x, denom):
        return round(100.0 * x / denom, 2) if denom > 0 else None

    exposure = {
        "observed_task_s": round(task_s, 3),
        "measured_s": round(measured_s, 3),
        "unmeasured_s": round(unmeasured_s, 3),
        "hazardous_exposure_s": total_s,
        "acceptable_posture_s": round(acceptable_s, 3),
        # Denominators are stated because they differ: percent_of_observed
        # includes occluded time, percent_of_measured excludes it.
        "percent_hazardous_of_observed": _pct(total_s, task_s),
        "percent_acceptable_of_observed": _pct(acceptable_s, task_s),
        "percent_unmeasured_of_observed": _pct(unmeasured_s, task_s),
        "percent_hazardous_of_measured": _pct(total_s, measured_s),
        "percent_acceptable_of_measured": _pct(acceptable_s, measured_s),
        "n_frames_total": n_frames,
        "n_frames_measured": n_measured,
        "n_frames_unmeasured": n_unmeasured,
        "above_threshold_frame_s": round(n_above * dt, 3),
        "bout_minus_framewise_s": round(total_s - n_above * dt, 3),
    }
    return {
        "exposure": exposure,
        "segment": segment,
        "entry_threshold_deg": entry,
        "exit_threshold_deg": exit_thr,
        "threshold_basis": REBA_CATEGORY_NOTE.get(segment, "caller-supplied"),
        "d_on_s": enter_s,
        "d_off_s": exit_s,
        "bouts": bouts_out,
        # Primary temporal exposure descriptors
        "n_bouts": n_bouts,
        "total_high_risk_s": total_s,
        "longest_bout_s": round(max((b["duration_s"] for b in bouts_out), default=0.0), 3),
        "percent_task_time": round(100.0 * total_s / task_s, 2) if task_s > 0 else 0.0,
        "observed_task_s": round(task_s, 3),
        # Conditional secondary
        "n_static_holds": n_static,
        "static_evaluated": eps_omega is not None and eps_theta is not None,
        "n_iso_limit_exceeded": n_iso_exceeded,
        "external_force_within_iso_scope": force_ok,
    }


# ── Self-check ───────────────────────────────────────────────────────────────

def _demo() -> None:
    """Assert-based self-check on synthetic signals with known ground truth."""
    fps = 60.0

    # 1. holding_time_limit_s: anchors, interpolation, and both tail regimes.
    assert holding_time_limit_s("trunk", 20.0) == 240.0
    assert holding_time_limit_s("trunk", 60.0) == 60.0
    assert holding_time_limit_s("trunk", 40.0) == 150.0   # midpoint -> midpoint of 240/60
    assert holding_time_limit_s("trunk", 10.0) is None    # below lower anchor -> unlimited
    assert holding_time_limit_s("trunk", 75.0) == 0.0     # beyond upper anchor
    assert holding_time_limit_s("neck", 25.0) == 480.0

    # 2. hysteresis_bouts (TIME gating): noise on either edge must not fragment
    #    or extend a bout, and onset/offset are back-dated.
    mask = np.array([0, 1, 1, 1, 0, 1, 1, 1, 1, 0], dtype=bool)
    assert hysteresis_bouts(mask, 3, 2) == [(1, 8)]
    mask2 = np.array([1, 1, 1, 0, 0, 1, 1, 1], dtype=bool)
    assert hysteresis_bouts(mask2, 3, 2) == [(0, 2), (5, 7)]
    assert hysteresis_bouts(np.array([1, 1, 0, 0, 0], dtype=bool), 3, 2) == []

    # 3. Entry thresholds come from REBA categories, exit sits a band below.
    assert entry_exit_thresholds("trunk") == (20.0, 15.0)
    assert entry_exit_thresholds("neck") == (20.0, 15.0)
    assert entry_exit_thresholds("upper_arm") == (45.0, 40.0)

    # 4. ANGLE hysteresis (Schmitt): a signal oscillating across a single
    #    threshold chatters; the dual threshold holds one state.
    t = np.arange(0, 4, 1 / fps)
    hovering = 20 + 1.5 * np.sin(2 * np.pi * 3 * t)
    single = int(np.sum(np.diff((np.abs(hovering) >= 20).astype(int)) != 0))
    dual = int(np.sum(np.diff(schmitt_state(hovering, 20.0, 15.0).astype(int)) != 0))
    assert single > 10, single          # the problem is real
    assert dual == 0, dual              # and the Schmitt trigger removes it

    # 5. Primary descriptors: two 3 s bouts inside a 10 s observation.
    t10 = np.arange(0, 10, 1 / fps)
    square = np.where((t10 % 5) < 3, 50.0, 0.0)
    r = segment_duration_bouts("trunk", square, fps)
    assert r["n_bouts"] == 2, r["n_bouts"]
    assert abs(r["total_high_risk_s"] - 6.0) < 0.1, r["total_high_risk_s"]
    assert abs(r["longest_bout_s"] - 3.0) < 0.1, r["longest_bout_s"]
    assert abs(r["percent_task_time"] - 60.0) < 1.0, r["percent_task_time"]
    assert r["bouts"][0]["onset_s"] == 0.0 and r["bouts"][0]["offset_s"] > 2.9

    # 6. STATIC BRANCH — never guessed. Without frozen eps, every bout reports
    #    "not evaluated"; a dynamic transition is never silently called static.
    assert r["static_evaluated"] is False
    assert all(b["is_static_hold"] is None for b in r["bouts"])
    assert all(b["iso_interpretation"] == "static_status_not_evaluated" for b in r["bouts"])

    # 7. With eps frozen: a smooth sustained hold IS static; the same threshold
    #    crossed dynamically is NOT.
    t12 = np.arange(0, 12, 1 / fps)
    hold = np.where(t12 < 8, 30.0 + 0.3 * np.sin(2 * np.pi * 0.05 * t12), 0.0)
    vel_hold = np.gradient(hold, 1 / fps)
    rh = segment_duration_bouts("trunk", hold, fps, angular_vel_deg_s=vel_hold,
                                eps_omega=2.0, eps_theta=1.0, load_score=0)
    assert rh["bouts"][0]["is_static_hold"] is True, rh["bouts"][0]["static_detail"]
    assert rh["n_static_holds"] == 1

    swinging = 40 + 40 * np.sin(2 * np.pi * 0.4 * t12)
    vel_sw = np.gradient(swinging, 1 / fps)
    rd = segment_duration_bouts("trunk", swinging, fps, angular_vel_deg_s=vel_sw,
                                eps_omega=2.0, eps_theta=1.0, load_score=0)
    assert rd["n_bouts"] >= 3, rd["n_bouts"]
    assert rd["n_static_holds"] == 0, "a dynamic transition must never be a static hold"

    # 8. A still hold SHORTER than the 4 s criterion fails on duration alone.
    short = np.where(t12 < 3, 30.0, 0.0)
    rs = segment_duration_bouts("trunk", short, fps, angular_vel_deg_s=np.gradient(short, 1 / fps),
                                eps_omega=2.0, eps_theta=1.0, load_score=0)
    assert rs["bouts"][0]["is_static_hold"] is False
    assert rs["bouts"][0]["static_detail"]["duration_ok"] is False

    # 9. R_hold is gated by BOTH static status and external force (criterion 5).
    assert rh["bouts"][0]["iso_holding_time_ratio"] is not None      # load 0 -> computed
    for load, expect in ((2, "external_force_exceeds_iso_scope"), (None, "external_force_unknown")):
        rr = segment_duration_bouts("trunk", hold, fps, angular_vel_deg_s=vel_hold,
                                    eps_omega=2.0, eps_theta=1.0, load_score=load)
        assert rr["bouts"][0]["iso_holding_time_ratio"] is None
        assert rr["bouts"][0]["iso_interpretation"] == expect, rr["bouts"][0]["iso_interpretation"]

    # 10. R_hold semantics: > 1 exceeds, <= 1 within, and an ANGLE beyond the
    #     upper anchor is reported as an angle verdict, never as a duration one.
    ratio, interp = holding_time_ratio("trunk", 40.0, 200.0)   # limit 150 s
    assert interp == "limit_exceeded" and ratio > 1.0
    ratio, interp = holding_time_ratio("trunk", 40.0, 10.0)
    assert interp == "within_limit" and ratio < 1.0
    ratio, interp = holding_time_ratio("trunk", 75.0, 10.0)
    assert ratio is None and interp == "angle_not_recommended"
    ratio, interp = holding_time_ratio("trunk", 5.0, 10.0)
    assert ratio is None and interp == "no_duration_criterion"

    # 11. NaN (occluded) frames are low, never bridged, and never counted.
    gapped = square.copy()
    gapped[100:110] = np.nan
    assert segment_duration_bouts("trunk", gapped, fps)["n_bouts"] >= 1

    # 12. EXPOSURE PARTITION — hazardous vs acceptable vs UNMEASURED.
    #     Occluded frames must never be banked as "acceptable posture time":
    #     that asserts a safe posture nobody observed.
    t10 = np.arange(0, 10, 1 / fps)
    part = np.where((t10 % 5) < 3, 50.0, 0.0)
    part[:60] = np.nan                       # 1.0 s occluded
    rp = segment_duration_bouts("trunk", part, fps)
    ex = rp["exposure"]
    assert abs(ex["observed_task_s"] - 10.0) < 0.05, ex
    assert abs(ex["unmeasured_s"] - 1.0) < 0.05, ex
    assert abs(ex["measured_s"] - 9.0) < 0.05, ex
    # The three parts must exactly reconstitute the observed interval.
    assert abs((ex["hazardous_exposure_s"] + ex["acceptable_posture_s"]
                + ex["unmeasured_s"]) - ex["observed_task_s"]) < 1e-6, ex
    # ...and acceptable must EXCLUDE the occluded second, not absorb it.
    assert abs(ex["acceptable_posture_s"]
               - (ex["measured_s"] - ex["hazardous_exposure_s"])) < 1e-6, ex
    assert ex["n_frames_unmeasured"] == 60, ex
    # Percentages state their denominator and differ when occlusion exists.
    assert ex["percent_hazardous_of_measured"] > ex["percent_hazardous_of_observed"], ex

    #     Bout time >= raw frame-wise time, because hysteresis holds a bout open
    #     across a brief dip. The difference is reported, not hidden.
    dip = np.full(int(6 * fps), 50.0)
    dip[int(3 * fps):int(3 * fps) + 5] = 0.0      # 5-frame dip, shorter than d_off
    rd2 = segment_duration_bouts("trunk", dip, fps)
    assert rd2["n_bouts"] == 1, rd2["n_bouts"]
    assert rd2["exposure"]["bout_minus_framewise_s"] > 0, rd2["exposure"]

    #     A fully-occluded trial yields no exposure claim of either kind.
    allnan = segment_duration_bouts("trunk", np.full(300, np.nan), fps)
    assert allnan["exposure"]["measured_s"] == 0.0
    assert allnan["exposure"]["hazardous_exposure_s"] == 0.0
    assert allnan["exposure"]["acceptable_posture_s"] == 0.0
    assert allnan["exposure"]["percent_hazardous_of_measured"] is None

    # 13. eps estimation returns provenance, and an absent frozen file yields {}
    #     rather than a fabricated default.
    est = estimate_static_thresholds({"trunk": [hold[:480], hold[:480]]}, fps)
    assert est["segments"]["trunk"]["n_trials"] == 2
    assert est["segments"]["trunk"]["eps_omega_deg_s"] > 0
    assert load_frozen_thresholds("/nonexistent/thresholds.json") == {}

    print("posture_duration self-check OK")


if __name__ == "__main__":
    _demo()
