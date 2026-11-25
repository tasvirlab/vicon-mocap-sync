"""Channel 2 (within-task posture duration) and Channel 3 (trunk-motion
dynamics) assembly — the two ergonomic channels beyond frame-wise REBA.

This is a thin orchestrator over the already-verified modules:
  - kinematics.py       — signed trunk angular velocity / acceleration
  - marras.py           — Marras (1993) five-factor LBD high-risk-group probability
  - posture_duration.py — hysteresis bout detection + ISO 11226 holding-time

The REBA router calls this AFTER computing angles so every trial persists all
three channels, not only posture severity (Channel 1). Kept out of reba.py to
keep that router under its size limit.

Load-bearing choices (see docs/engineering-notes.md):
  - Vicon-derived path passes cutoff_hz=None: the markers are already 6 Hz
    zero-phase filtered upstream by c3d_pipeline, so filtering again biases
    velocity low. The future RGB/model-derived path MUST pass cutoff_hz=6.0.
  - Marras probability is computed ONLY for eligible repetitive material-handling
    tasks AND only when all five inputs are available. lift_rate and load moment
    are not derivable from pose alone (no repetition counter / load metadata
    wired yet), so it currently reports applicable=False rather than guessing.
  - ISO 11226 holding-time anchors are PROVISIONAL (unverified against the
    standard). Bout onset/offset/duration/count are the primary trustworthy
    outputs; ISO exceedance flags carry iso_11226_values_verified=False.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import kinematics as _kin
from . import marras as _marras
from . import posture_duration as _dur
from . import repetition as _rep

# Tasks eligible for the Marras repetitive-MMH model (Marras et al. 1993).
# Substring-matched against the task_id. "handling" is deliberately excluded
# until its protocol is confirmed to involve a countable repeated lift cycle.
MARRAS_ELIGIBLE_TASKS = ("lifting",)

# Tasks whose movement is repetitive/dynamic and externally loaded, i.e. OUTSIDE
# ISO 11226's scope (static postures, no or minimal external force). For these
# the holding-time comparison is withheld entirely — see iso_applicability().
ISO_OUT_OF_SCOPE_TASKS = ("lifting", "carrying", "handling")

# Segments with an ISO 11226 holding-time curve, mapped to their signed angle
# channel in the angles dict. upper_arm_l/r both use the "upper_arm" ISO curve.
_DURATION_SEGMENTS = {
    "trunk":       ("trunk", "trunk_sagittal_deg"),
    "neck":        ("neck", "neck_flex_deg"),
    "upper_arm_r": ("upper_arm", "upper_arm_r_deg"),
    "upper_arm_l": ("upper_arm", "upper_arm_l_deg"),
}

_ISO_NOTE = (
    "ISO 11226 holding-time anchors are provisional approximations pending "
    "verification against ISO 11226:2000 Figs 4/7/10. Bout onset/offset/"
    "duration/count/total are the primary trustworthy outputs; the "
    "'exceeded' / iso_limit_s fields are provisional and must not be reported "
    "as validated endurance-limit exceedance until the anchors are checked."
)


def infer_fps(time_s: np.ndarray | None, default: float = 60.0) -> float:
    """Frame rate from a time column (median inter-frame dt), or a default."""
    if time_s is None:
        return default
    t = np.asarray(time_s, dtype=float)
    if t.size < 2:
        return default
    dt = np.median(np.diff(t))
    return float(1.0 / dt) if dt > 0 else default


# ── Channel 2 ──────────────────────────────────────────────────────────────────

def iso_applicability(task_id: str = "") -> dict:
    """Whether an ISO 11226 holding-time comparison can be considered AT ALL.

    Applicability is decided PER BOUT, not per task: a bout qualifies only when
    it is classified static (S_s,i = 1) AND the external force is within ISO's
    scope (entered REBA load score <= ISO_MAX_LOAD_SCORE). Both tests live in
    posture_duration.segment_duration_bouts().

    The earlier task-NAME exclusion list was replaced: a task label is a weak
    proxy for what the standard actually requires, and it both over-blocked
    (a genuinely static hold inside a "handling" trial) and under-blocked (a
    loaded static hold in a task not on the list). What remains here is the
    single global fact a reader needs.
    """
    return {
        "applicable": False,
        "decided_per_bout": True,
        "reason": ("ISO 11226 holding-time comparison is evaluated PER BOUT, not per task: a "
                   "bout must be classified static (duration > 4 s, low median angular "
                   "velocity, low angular variability) AND carry no or minimal external "
                   "force. Anchors remain unverified against ISO 11226:2000."),
    }


def compute_posture_duration(angles: dict[str, np.ndarray], fps: float,
                             *, enter_s: float = 0.5, exit_s: float = 0.5,
                             task_id: str = "",
                             load_score: int | None = None,
                             frozen_thresholds: dict | None = None,
                             hysteresis_band_deg: float | None = None) -> dict:
    """Within-Task High-Risk Posture-Duration Assessment Layer (Channel 2).

    Converts a continuous filtered angle sequence into stable temporal events
    and reports, per segment: bout onset, offset, individual duration, count,
    longest bout, total high-risk time, and percentage of observed task time.
    These are temporal EXPOSURE DESCRIPTORS, not a new ergonomic risk score.

    Applies to dynamic AND static tasks. Static-hold classification is a
    CONDITIONAL SECONDARY branch decided per bout (never per task), and the ISO
    11226 holding-time ratio is computed only for a bout that is both static and
    within ISO's minimal-external-force scope.

    Args:
        load_score: entered REBA load/force score for the trial. Criterion 5 of
            the static branch — never inferred from pose. None -> ISO ratio
            withheld as "external_force_unknown".
        frozen_thresholds: {segment: (eps_omega, eps_theta)} from
            posture_duration.load_frozen_thresholds(). Absent -> every bout
            reports is_static_hold = None ("not evaluated"), never a guess.
    """
    frozen = frozen_thresholds or {}

    # Velocity for the static criterion comes from the SAME kinematics path the
    # dynamics channel uses, so a "static" verdict cannot disagree with Ch3.
    vel_channels = tuple(k for _, k in _DURATION_SEGMENTS.values() if k in angles)
    kin = (_kin.compute_kinematics(angles, fs=fps, cutoff_hz=None, channels=vel_channels)
           if vel_channels else {})

    segments_out: dict[str, dict] = {}
    for seg_key, (iso_seg, angle_key) in _DURATION_SEGMENTS.items():
        if angle_key not in angles:
            continue
        eps_w, eps_t = frozen.get(iso_seg, (None, None))
        res = _dur.segment_duration_bouts(
            iso_seg, angles[angle_key], fps,
            enter_s=enter_s, exit_s=exit_s,
            exit_threshold_deg=(None if hysteresis_band_deg is None else
                                max(_dur.REBA_ENTRY_DEG[iso_seg] - abs(hysteresis_band_deg), 0.0)),
            angular_vel_deg_s=kin.get(f"{angle_key[:-4]}_vel_deg_s"),
            eps_omega=eps_w, eps_theta=eps_t,
            load_score=load_score)
        res["angle_channel"] = angle_key
        segments_out[seg_key] = res

    any_static_eval = any(s["static_evaluated"] for s in segments_out.values())
    return {
        "channel": "posture_duration",
        "layer_name": "Within-Task High-Risk Posture-Duration Assessment Layer",
        "fps": round(fps, 4),
        "task_id": task_id,
        "hysteresis": {"enter_s": enter_s, "exit_s": exit_s,
                       "onset_backdated": True, "offset_backdated": True,
                       "band_deg": (hysteresis_band_deg if hysteresis_band_deg is not None
                                    else _dur.HYSTERESIS_BAND_DEG)},
        "threshold_basis": "REBA per-segment category boundaries (see posture_duration.REBA_ENTRY_DEG)",
        "static_criteria": {
            "min_duration_s": _dur.STATIC_MIN_DURATION_S,
            "iso_max_load_score": _dur.ISO_MAX_LOAD_SCORE,
            "thresholds_frozen": bool(frozen),
            "evaluated": any_static_eval,
        },
        "iso_11226_values_verified": False,
        "iso_assessment": iso_applicability(task_id),
        "iso_note": _ISO_NOTE,
        "bout_definition": ("above-threshold posture bout: includes the rising and falling "
                            "phases of a movement. A bout is a static hold only when "
                            "is_static_hold is True."),
        "segments": segments_out,
    }


# ── Channel 3 ──────────────────────────────────────────────────────────────────

def compute_trunk_dynamics(angles: dict[str, np.ndarray], fps: float,
                           task_id: str,
                           *, lift_rate_per_hr: float | None = None,
                           max_moment_nm: float | None = None
                           ) -> tuple[pd.DataFrame, dict]:
    """Channel 3: signed trunk angles + angular velocity/acceleration + Marras.

    Returns (per_frame_df, summary_dict). Vicon path → cutoff_hz=None.
    """
    kin = _kin.compute_kinematics(angles, fs=fps, cutoff_hz=None,
                                  channels=_kin.TRUNK_CHANNELS)
    predictors = _kin.summarise_trunk_motion(kin)

    # Per-frame CSV: angle + velocity + acceleration for each trunk plane.
    per_frame = {}
    for c in _kin.TRUNK_CHANNELS:
        stem = c[:-4]  # strip "_deg"
        per_frame[c] = kin[c]
        per_frame[f"{stem}_vel_deg_s"] = kin[f"{stem}_vel_deg_s"]
        per_frame[f"{stem}_acc_deg_s2"] = kin[f"{stem}_acc_deg_s2"]
    per_frame_df = pd.DataFrame(per_frame)

    # Repetition counting on trunk sagittal flexion. Informative for every task
    # (cadence/recovery); supplies the Marras lift_rate for eligible tasks.
    # Task-aware: picks the plane whose peaks define a cycle for this task, and
    # labels the result a lift or a movement cycle. A non-lift task returns
    # lift_rate_per_hr = None so its cadence can never reach the Marras model.
    rep = _rep.count_task_repetitions(angles, fps, task_id)

    # Marras: eligible task AND all five inputs present, else applicable=False.
    eligible = any(t in task_id for t in MARRAS_ELIGIBLE_TASKS)
    if eligible:
        # Prefer an explicitly supplied (e.g. field-observed) lift rate; else use
        # the counted within-trial rate — which carries its own extrapolation
        # caveat (see repetition.py). max_moment is still task metadata.
        lr = lift_rate_per_hr if lift_rate_per_hr is not None else rep.get("lift_rate_per_hr")
        marras = _marras.summarise_lbd_risk(
            lift_rate_per_hr=lr,
            avg_twisting_vel_deg_s=predictors.get("mean_twist_vel_deg_s"),
            max_moment_nm=max_moment_nm,
            max_sagittal_flexion_deg=predictors.get("peak_trunk_sagittal_deg"),
            max_lateral_vel_deg_s=predictors.get("peak_lateral_vel_deg_s"),
        )
        marras["lift_rate_source"] = ("supplied" if lift_rate_per_hr is not None
                                      else "within_trial_extrapolation")
        if not marras["applicable"]:
            marras["reason"] = (
                marras.get("reason", "")
                + " (max_moment_nm is task metadata not derivable from pose alone "
                  "— supply a load mass + pose-derived moment arm to enable the "
                  "Marras probability; lift_rate is now provided by the rep counter)."
            )
    else:
        marras = {
            "applicable": False,
            "reason": (f"task '{task_id}' is not in the Marras-eligible set "
                       f"{MARRAS_ELIGIBLE_TASKS}; dynamics reported without "
                       f"a high-risk-group probability (Marras et al. 1993)."),
            "probability": None,
        }

    summary = {
        "channel": "trunk_dynamics",
        "fps": round(fps, 4),
        "cutoff_hz": None,   # Vicon path — already filtered upstream
        "repetition": rep,
        "marras_predictors": predictors,
        "marras_probability": marras,
    }
    return per_frame_df, summary


# ── Persistence ────────────────────────────────────────────────────────────────

def _json_default(o):
    """JSON encoder fallback for stray numpy scalars/arrays."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serializable: {type(o)}")


def run_and_write(angles: dict[str, np.ndarray], fps: float, task_id: str,
                  reba_out, *, frame_col=None, time_col=None,
                  load_score: int | None = None, frozen_thresholds: dict | None = None,
                  lift_rate_per_hr: float | None = None,
                  max_moment_nm: float | None = None) -> dict:
    """Compute + persist Channels 2 & 3 into reba_out. Returns compact summaries
    for embedding in result.json (not the full per-bout / per-frame payloads)."""
    import json
    from utils.task_paths import Artifacts

    ch2 = compute_posture_duration(angles, fps, task_id=task_id,
                                   load_score=load_score,
                                   frozen_thresholds=frozen_thresholds)
    dyn_df, ch3 = compute_trunk_dynamics(
        angles, fps, task_id,
        lift_rate_per_hr=lift_rate_per_hr, max_moment_nm=max_moment_nm)

    if frame_col is not None:
        dyn_df.insert(0, "frame", frame_col)
    if time_col is not None:
        dyn_df.insert(1 if frame_col is not None else 0, "time_s", time_col)

    (reba_out / Artifacts.REBA_POSTURE_DURATION).write_text(
        json.dumps(ch2, indent=2, default=_json_default), encoding="utf-8")
    dyn_df.to_csv(reba_out / Artifacts.REBA_TRUNK_DYNAMICS_CSV,
                  index=False, float_format="%.4f")
    (reba_out / Artifacts.REBA_TRUNK_DYNAMICS_JSON).write_text(
        json.dumps(ch3, indent=2, default=_json_default), encoding="utf-8")

    dur_summary = {
        seg: {
            # Primary temporal exposure descriptors (instrument-independent)
            "n_bouts": d["n_bouts"],
            "total_high_risk_s": d["total_high_risk_s"],
            "longest_bout_s": d["longest_bout_s"],
            "percent_task_time": d["percent_task_time"],
            "exposure": d["exposure"],
            "entry_threshold_deg": d["entry_threshold_deg"],
            # Conditional secondary
            "n_static_holds": d["n_static_holds"],
            "static_evaluated": d["static_evaluated"],
            "n_iso_limit_exceeded": d["n_iso_limit_exceeded"],
        }
        for seg, d in ch2["segments"].items()
    }
    _rep_block = ch3["repetition"]
    return {
        "posture_duration": {"iso_11226_values_verified": False,
                             "iso_assessment": ch2["iso_assessment"],
                             "segments": dur_summary},
        "trunk_dynamics": {
            "repetition": {"n_reps": _rep_block["n_reps"],
                           "cycle_kind": _rep_block.get("cycle_kind"),
                           "counting_channel": _rep_block.get("counting_channel"),
                           "cycle_rate_per_hr": _rep_block.get("cycle_rate_per_hr"),
                           "lift_rate_per_hr": _rep_block.get("lift_rate_per_hr"),
                           "plane_warning": _rep_block.get("plane_warning"),
                           "cycle_timing": _rep_block.get("cycle_timing"),
                           "mean_period_s": _rep_block["mean_period_s"],
                           "within_trial_extrapolation": True},
            "marras_predictors": ch3["marras_predictors"],
            "marras_probability": ch3["marras_probability"]},
    }


# ── Self-check ───────────────────────────────────────────────────────────────

def _demo() -> None:
    """Assert-based self-check on synthetic angles (structure + eligibility)."""
    fps = 60.0
    t = np.arange(0, 6, 1 / fps)
    # A repeated deep-bend sagittal signal; small lateral/twist.
    sag = 60 + 60 * np.sin(2 * np.pi * 0.4 * t)      # swings 0..120
    small = 5 * np.sin(2 * np.pi * 0.4 * t)
    angles = {
        "trunk_sagittal_deg": sag,
        "trunk_lateral_deg": small,
        "trunk_twist_deg": small,
        "neck_flex_deg": 30 + small,
        "upper_arm_r_deg": 70 + small,
        "upper_arm_l_deg": 15 + small,
    }

    ch2 = compute_posture_duration(angles, fps)
    assert ch2["iso_11226_values_verified"] is False
    assert "trunk" in ch2["segments"] and ch2["segments"]["trunk"]["n_bouts"] >= 1

    # Eligible task → Marras attempted (applicable False here, inputs missing).
    df_lift, sum_lift = compute_trunk_dynamics(angles, fps, "SUB01_lifting_T01")
    assert list(df_lift.columns) == [
        "trunk_sagittal_deg", "trunk_sagittal_vel_deg_s", "trunk_sagittal_acc_deg_s2",
        "trunk_lateral_deg", "trunk_lateral_vel_deg_s", "trunk_lateral_acc_deg_s2",
        "trunk_twist_deg", "trunk_twist_vel_deg_s", "trunk_twist_acc_deg_s2",
    ], list(df_lift.columns)
    assert sum_lift["marras_probability"]["applicable"] is False  # max_moment missing
    assert "peak_trunk_sagittal_deg" in sum_lift["marras_predictors"]
    # Repetition counting is wired in and lift_rate now flows to Marras.
    assert "repetition" in sum_lift and sum_lift["repetition"]["n_reps"] >= 1
    assert sum_lift["marras_probability"]["lift_rate_source"] == "within_trial_extrapolation"

    # Ineligible task → Marras not attempted at all.
    _, sum_bend = compute_trunk_dynamics(angles, fps, "SUB01_forward_trunk_bending_T03")
    assert sum_bend["marras_probability"]["applicable"] is False
    assert "not in the Marras-eligible set" in sum_bend["marras_probability"]["reason"]

    # Marras probability DOES compute when eligible + all inputs supplied.
    _, sum_full = compute_trunk_dynamics(
        angles, fps, "SUB01_lifting_T01",
        lift_rate_per_hr=120.0, max_moment_nm=60.0)
    assert sum_full["marras_probability"]["applicable"] is True
    assert 0.0 <= sum_full["marras_probability"]["probability"] <= 1.0

    print("channels self-check OK")


if __name__ == "__main__":
    _demo()
