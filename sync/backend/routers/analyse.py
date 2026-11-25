"""
analyse.py — /api/analyse endpoints
======================================
Upload one CSV, get the Channel 2 (posture duration) and Channel 3 (trunk
dynamics) figures back, rendered by core/vicon_reba/channel_figures.py.

This is a READ-ONLY inspection surface: nothing is written to outputs/, no
metadata stage is marked, no trial is modified. It exists so a trial's duration
and dynamics channels can be looked at as figures — during QC, or when pulling
a figure panel — without re-running the REBA stage.

Three accepted uploads, auto-detected by their columns:
  - reba/angles.csv       — signed angle channels; Channels 2+3 recomputed here
  - synced_60hz.csv       — 3D joints; angles computed first, then as above
  - reba/trunk_dynamics.csv — pre-computed Channel 3 only (angle/vel/acc columns),
                              paired with the trial's trunk_dynamics.json if the
                              user also uploads it

Load-bearing: the recompute path passes cutoff_hz=None via channels.py, exactly
as the REBA stage does for Vicon-derived angles (markers are already 6 Hz
zero-phase filtered by c3d_pipeline). A figure produced here therefore matches
the trial's own trunk_dynamics.csv rather than presenting a differently-filtered
second version of the same signal.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from core.vicon_reba import angles as vr_angles
from core.vicon_reba import channel_figures as vr_figs
from core.vicon_reba import channels as vr_channels
from core.vicon_reba import load as vr_load
from core.vicon_reba import neutral_offset as vr_off
from core.vicon_reba import posture_duration as vr_dur
from core.vicon_reba import reba as vr_reba
from utils import app_config

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_UPLOAD_MB = 200

# Frozen low-motion thresholds for the static-hold branch. Absent until derived
# from train-pool static calibration trials, in which case static classification
# reports "not evaluated" rather than guessing.
def _calib_dir() -> Path:
    return app_config.get_outputs_dir() / "_calibration"


# Per-subject neutral-reference offsets. Absent -> angles stay RAW and the
# response says so; never someone else's neutral pose, never a guess.

# A signed angle CSV must carry at least the trunk sagittal channel; a joint CSV
# is recognised by the 3D position columns the angle computation needs.
_ANGLE_MARKER = "trunk_sagittal_deg"
_JOINT_MARKERS = ("PELCEN_X", "RHIP_X", "LSHO_X")
_DYN_MARKER = "trunk_sagittal_vel_deg_s"


def _read_upload(f: UploadFile) -> pd.DataFrame:
    raw = f.file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"{f.filename} exceeds {MAX_UPLOAD_MB} MB")
    if not raw:
        raise HTTPException(400, f"{f.filename} is empty")
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Could not parse {f.filename} as CSV: {e}")


def detect_kind(df: pd.DataFrame) -> str:
    """Classify an uploaded CSV: 'dynamics' | 'angles' | 'joints'.

    Order matters — trunk_dynamics.csv also contains `trunk_sagittal_deg`, so
    the velocity column must be checked first or it would be misread as an
    angles.csv and silently re-differentiated.
    """
    cols = set(df.columns)
    if _DYN_MARKER in cols:
        return "dynamics"
    if _ANGLE_MARKER in cols:
        return "angles"
    if all(c in cols for c in _JOINT_MARKERS):
        return "joints"
    raise HTTPException(
        400,
        "Unrecognised CSV. Expected one of: reba/angles.csv (has "
        f"'{_ANGLE_MARKER}'), synced_60hz.csv (has {', '.join(_JOINT_MARKERS)}), "
        f"or reba/trunk_dynamics.csv (has '{_DYN_MARKER}'). "
        f"Got columns: {', '.join(list(df.columns)[:12])}...")


def _time_column(df: pd.DataFrame) -> Optional[np.ndarray]:
    for c in ("time_s", "vicon_time_s"):
        if c in df.columns:
            return df[c].to_numpy(dtype=float)
    return None


def _infer_fps(df: pd.DataFrame, override: Optional[float]) -> float:
    if override and override > 0:
        return float(override)
    return vr_channels.infer_fps(_time_column(df))


def _angles_from(df: pd.DataFrame, kind: str) -> dict[str, np.ndarray]:
    """Signed angle channels, either read straight from the CSV or computed."""
    if kind == "joints":
        joints = vr_load.csv_to_joints(vr_load.normalize_frame(df))
        return vr_angles.compute_reba_angles(joints)
    return {c: df[c].to_numpy(dtype=float)
            for c in df.columns if c.endswith("_deg")}


@router.post("/figures")
async def analyse_figures(
    file: UploadFile = File(...),
    dynamics_json: Optional[UploadFile] = File(default=None),
    fps: Optional[float] = Form(default=None),
    enter_s: float = Form(default=0.5),
    exit_s: float = Form(default=0.5),
    task_id: str = Form(default=""),
    load: int = Form(default=0),
    coupling: int = Form(default=0),
    activity: int = Form(default=1),
    unilateral: bool = Form(default=False),
):
    """Render Channel 2 + Channel 3 figures from one uploaded CSV.

    Returns base64 PNGs plus the numeric summaries behind them, so the caller
    can show the figure and the numbers it was drawn from side by side.
    """
    df = _read_upload(file)
    kind = detect_kind(df)
    sample_rate = _infer_fps(df, fps)
    t_col = _time_column(df)

    # Two DIFFERENT things, deliberately not one variable:
    #   `label`   — cosmetic figure title; falling back to the filename is fine.
    #   `task_key`— the real trial identity that decides ISO applicability and
    #               Marras eligibility. It must stay EMPTY when the user gave no
    #               label, or the filename ("angles.csv") sails through the
    #               dynamic-task check and re-enables the ISO layer for an
    #               unidentified trial — the exact fail-open this guard exists to
    #               prevent. A filename is not a task identity.
    task_key = task_id.strip()
    label = task_key or file.filename or ""

    out: dict = {
        "input_kind": kind,
        "filename": file.filename,
        "n_frames": int(len(df)),
        "fps": round(sample_rate, 4),
        "task_id": label,
        "figures": {},
        "warnings": [],
    }

    if kind == "dynamics":
        # Pre-computed Channel 3 only. Its own JSON carries the Marras block;
        # without it we plot the traces and say so rather than inventing one.
        ch3: dict = {"fps": sample_rate, "cutoff_hz": None,
                     "marras_predictors": {}, "repetition": {},
                     "marras_probability": {"applicable": False,
                                            "reason": "trunk_dynamics.json not uploaded"}}
        if dynamics_json is not None:
            try:
                ch3 = json.loads(dynamics_json.file.read().decode("utf-8"))
            except Exception as e:
                raise HTTPException(400, f"Could not parse trunk_dynamics.json: {e}")
        else:
            out["warnings"].append(
                "Only trunk_dynamics.csv was uploaded — the Marras predictor panel is "
                "empty. Upload the trial's trunk_dynamics.json alongside it, or upload "
                "angles.csv instead to recompute both channels.")
        out["figures"]["trunk_dynamics"] = base64.b64encode(
            vr_figs.plot_trunk_dynamics(
                df, ch3, time_s=t_col, title=label,
                rep_marks=(ch3.get("repetition") or {}).get("peak_times_s"))).decode()
        out["trunk_dynamics"] = {k: v for k, v in ch3.items() if k != "per_frame"}
        out["warnings"].append(
            "Channel 2 (posture duration) needs the segment angle series — upload "
            "reba/angles.csv or synced_60hz.csv to get the duration figure too.")
        return out

    angles = _angles_from(df, kind)

    # ── Neutral-reference correction ─────────────────────────────────────────
    # neck_flex_deg carries a +27..+33 deg anatomical bias (the C7->head-centroid
    # vector is tilted forward), so 0 deg is not anatomical neutral and both the
    # REBA neck category and this layer's neck threshold act on a biased signal.
    # Correct it from the subject's own static capture. Arms are excluded inside
    # apply_offsets(): the static pose holds them abducted.
    _subject = vr_off.parse_subject(task_key)
    _offsets = vr_off.load_offsets(_calib_dir() / "neutral_offsets.json", _subject)
    angles, _off_rec = vr_off.apply_offsets(angles, _offsets)
    out["neutral_offset"] = {"subject": _subject, **_off_rec}
    _angle_ref = (
        "Angles expressed relative to the subject's measured NEUTRAL pose ("
        + ", ".join(f"{c.replace('_deg','')} {v:+.1f}deg" for c, v in _off_rec["applied"].items())
        + " subtracted; arms deliberately not offset)."
        if _off_rec["any_applied"] else
        "Angles are RAW (no neutral-reference offset applied) — neck_flex carries a "
        "+27..+33deg anatomical bias.")
    if not _off_rec["any_applied"]:
        out["warnings"].append(
            "Angles are RAW — no neutral-reference offset was applied"
            + (f" for {_subject}" if _subject else " (no subject in the trial label)")
            + ". neck_flex_deg carries a +27..+33 deg anatomical bias, so neck "
              "severity and neck exposure are NOT comparable across subjects until "
              "offsets are derived (python -m tools.derive_static_thresholds).")
    if _ANGLE_MARKER not in angles:
        raise HTTPException(400, f"'{_ANGLE_MARKER}' is missing after parsing — "
                                 f"cannot compute trunk dynamics.")

    # `task_id` decides ISO applicability (a repetitive dynamic lift is outside
    # ISO 11226's static-posture scope), so the label is not merely cosmetic here.
    # `load` is criterion 5 of the static branch (ISO's minimal-external-force
    # scope) as well as a REBA input — the same entered metadata, used once.
    ch2 = vr_channels.compute_posture_duration(
        angles, sample_rate, enter_s=enter_s, exit_s=exit_s, task_id=task_key,
        load_score=load,
        frozen_thresholds=vr_dur.load_frozen_thresholds(_calib_dir() / "static_thresholds.json"))
    dyn_df, ch3 = vr_channels.compute_trunk_dynamics(angles, sample_rate, task_key)
    rep_marks = (ch3.get("repetition") or {}).get("peak_times_s")

    # ── Channel 1: REBA posture severity ────────────────────────────────────
    # load/coupling/activity/unilateral are ENTERED task metadata, never inferred
    # from pose. The defaults are the form's, so a caller who
    # leaves them alone gets a posture-only score — flagged below, not silently
    # presented as if the trial's real load had been applied.
    try:
        reba_df = vr_reba.score_timeseries(
            angles, load=load, coupling=coupling,
            activity=activity, unilateral=unilateral)
        reba_summary = vr_reba.summarise(reba_df)
        out["figures"]["reba"] = base64.b64encode(
            vr_figs.plot_reba(reba_df, reba_summary, fps=sample_rate, time_s=t_col,
                              title=label, rep_marks=rep_marks,
                              angle_reference=_angle_ref)).decode()
        out["reba"] = {
            "summary": reba_summary,
            "params": {"load": load, "coupling": coupling,
                       "activity": activity, "unilateral": unilateral},
        }
        if load == 0 and coupling == 0:
            out["warnings"].append(
                "REBA was computed with load = 0 and coupling = 0 (the form defaults). "
                "If this trial actually involved a load, enter it — REBA severity is not "
                "comparable across trials scored with different task metadata.")
    except (ValueError, KeyError) as e:
        # An all-occluded trial, or an angle set lacking a REBA input, must not
        # take the other two channels down with it.
        out["warnings"].append(f"Channel 1 (REBA) could not be computed: {e}")

    out["figures"]["posture_duration"] = base64.b64encode(
        vr_figs.plot_posture_duration(ch2, angles, sample_rate, time_s=t_col,
                                      title=label, rep_marks=rep_marks,
                                      angle_reference=_angle_ref)).decode()
    out["figures"]["trunk_dynamics"] = base64.b64encode(
        vr_figs.plot_trunk_dynamics(dyn_df, ch3, time_s=t_col, title=label,
                                    rep_marks=rep_marks)).decode()

    out["posture_duration"] = {
        "iso_11226_values_verified": ch2["iso_11226_values_verified"],
        "iso_assessment": ch2["iso_assessment"],
        "layer_name": ch2["layer_name"],
        "bout_definition": ch2["bout_definition"],
        "static_criteria": ch2["static_criteria"],
        "hysteresis": ch2["hysteresis"],
        "segments": {
            seg: {
                "n_bouts": d["n_bouts"],
                "total_high_risk_s": d["total_high_risk_s"],
                "longest_bout_s": d["longest_bout_s"],
                "percent_task_time": d["percent_task_time"],
                "observed_task_s": d["observed_task_s"],
                "exposure": d["exposure"],
                "entry_threshold_deg": d["entry_threshold_deg"],
                "exit_threshold_deg": d["exit_threshold_deg"],
                "threshold_basis": d["threshold_basis"],
                "n_static_holds": d["n_static_holds"],
                "static_evaluated": d["static_evaluated"],
                "n_iso_limit_exceeded": d["n_iso_limit_exceeded"],
                "bouts": d["bouts"],
            }
            for seg, d in ch2["segments"].items()
        },
    }
    out["trunk_dynamics"] = {
        "cutoff_hz": ch3["cutoff_hz"],
        "repetition": ch3["repetition"],
        "marras_predictors": ch3["marras_predictors"],
        "marras_probability": ch3["marras_probability"],
    }
    if kind == "joints":
        out["warnings"].append(
            "Angles were recomputed from 3D joint positions in this upload. For the "
            "figure to match a trial's stored reba/ outputs exactly, upload that "
            "trial's angles.csv instead.")
    return out
