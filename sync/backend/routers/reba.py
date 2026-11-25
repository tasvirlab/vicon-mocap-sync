"""
reba.py — /api/reba endpoints
================================
Compute REBA ergonomic risk scores from a task's video_synced.csv.

The video_synced.csv is produced by the Overlay step and already contains
Vicon joint positions aligned to video frame timestamps.  No calibration
files are needed here — it is purely angle computation + REBA scoring.

Result is cached as  outputs/<task_name>/reba/result.json  so repeated
requests within a session are instant.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import StreamingResponse

from utils import app_config
from utils.job_manager import manager
from utils.task_paths import task_dir as resolve_task_dir, Artifacts, update_stage
from core.vicon_reba import load as vr_load
from core.vicon_reba import angles as vr_angles
from core.vicon_reba import reba as vr_reba
from core.vicon_reba import pig_angles as vr_pig
from core.vicon_reba import channels as vr_channels

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _stats(arr: np.ndarray) -> dict:
    """Return mean / min / max / median / std for a 1-D array."""
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": 0, "min": 0, "max": 0, "median": 0, "std": 0}
    return {
        "mean":   round(float(np.mean(arr)),   3),
        "min":    round(float(np.min(arr)),    3),
        "max":    round(float(np.max(arr)),    3),
        "median": round(float(np.median(arr)), 3),
        "std":    round(float(np.std(arr)),    3),
    }


def _int_stats(arr: np.ndarray) -> dict:
    """Like _stats but rounds to 2 decimal places (for integer-like scores)."""
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": 0, "min": 0, "max": 0, "median": 0, "std": 0}
    return {
        "mean":   round(float(np.mean(arr)),   2),
        "min":    int(np.min(arr)),
        "max":    int(np.max(arr)),
        "median": round(float(np.median(arr)), 2),
        "std":    round(float(np.std(arr)),    2),
    }


# ── Background worker ──────────────────────────────────────────────────────────

def _run_reba_job(
    job_id:    str,
    task_name: str,
    synced_csv: Path,
    output_dir: Path,
    load:      int,
    coupling:  int,
    activity:  int,
    unilateral: bool,
    c3d_path:  Optional[Path] = None,
    auto_load: bool = False,
):
    t0 = time.time()

    def _cb(pct: int, step: str):
        manager.update_job(job_id, progress=pct, step=step)

    try:
        _cb(5, "Loading Vicon CSV…")
        joints, df = vr_load.load(synced_csv)
        n_frames = len(df)
        logger.info(f"REBA [{task_name}]: {n_frames} frames loaded")

        _cb(20, "Computing joint angles…")
        angles  = vr_angles.compute_reba_angles(joints)
        frame_col = df["frame"].to_numpy() if "frame" in df.columns else np.arange(n_frames)
        time_col  = df["time_s"].to_numpy() if "time_s" in df.columns else None
        angles_df = vr_angles.angles_to_dataframe(angles, frame_col=frame_col, time_col=time_col)

        # ── Angle stage: prefer validated Plug-in Gait leg angles when a C3D is
        # available. PiG knee flexion (Grood-Suntay, anatomical) replaces the
        # marker-geometry knee feeding REBA legs; we also record the marker-vs-PiG
        # agreement as Study-1 validation evidence and can estimate load from the
        # force plates. Trunk/neck/arms stay marker-derived (PiG is lower-body only).
        pig_report: dict = {"available": False}
        if c3d_path is not None and Path(c3d_path).exists() and time_col is not None:
            _cb(35, "Extracting Plug-in Gait angles…")
            pig = vr_pig.extract_pig_angles(Path(c3d_path))
            if pig.get("available"):
                marker_knee = angles["knee_max_deg"].copy()
                pig_report = vr_pig.crosscheck_marker_vs_pig(pig, marker_knee, time_col)
                pig_knee = vr_pig.knee_flexion_at_times(pig, time_col)
                if pig_knee is not None:
                    # Use PiG where finite, keep marker value where PiG is occluded.
                    fused = np.where(np.isfinite(pig_knee), pig_knee, marker_knee)
                    angles["knee_max_deg"] = fused
                    angles_df["knee_max_deg_pig"] = pig_knee
                    pig_report["knee_source"] = "pig_with_marker_fallback"
            if auto_load:
                _cb(40, "Estimating load from force plates…")
                load_est = vr_pig.estimate_load_score(Path(c3d_path))
                if load_est.get("available"):
                    load = load_est["load_score"]
                    pig_report["load_estimate"] = load_est

        _cb(50, "Computing REBA scores…")
        reba_df = vr_reba.score_timeseries(
            angles,
            load=load, coupling=coupling, activity=activity, unilateral=unilateral,
        )
        if "frame"  in angles_df.columns: reba_df.insert(0, "frame",  frame_col)
        if time_col is not None:          reba_df.insert(1 if "frame" in reba_df.columns else 0, "time_s", time_col)

        _cb(65, "Saving CSV outputs…")
        reba_out = output_dir / Artifacts.REBA_DIR
        reba_out.mkdir(parents=True, exist_ok=True)
        angles_df.to_csv(reba_out / Artifacts.REBA_ANGLES, index=False, float_format="%.4f")
        reba_df.to_csv(  reba_out / Artifacts.REBA_SCORES, index=False)

        # ── Channels 2 & 3: within-task posture duration + trunk-motion dynamics.
        # Writes posture_duration.json, trunk_dynamics.csv, trunk_dynamics.json.
        # Vicon path → cutoff_hz=None inside channels. Marras is attempted only
        # for eligible tasks and reports applicable=False when its inputs are
        # missing (lift_rate/load moment not yet wired). See vicon_reba/channels.py.
        _cb(70, "Computing posture-duration + trunk-dynamics channels…")
        fps = vr_channels.infer_fps(time_col)
        channels_summary = vr_channels.run_and_write(
            angles, fps, task_name, reba_out,
            frame_col=frame_col, time_col=time_col,
        )

        _cb(75, "Computing statistics…")
        summary = vr_reba.summarise(reba_df)

        # Per-angle summary statistics
        angle_keys = [k for k in angles if k.endswith("_deg")]
        angle_summary = {k: _stats(angles[k]) for k in angle_keys}

        # Per-REBA-component statistics (scores, not angles)
        score_cols = [
            "trunk_score", "neck_score", "legs_score",
            "upper_arm_adj", "lower_arm_score", "wrist_adj",
            "score_a", "score_b", "reba_total",
        ]
        component_stats = {
            col: _int_stats(reba_df[col].to_numpy(dtype=float))
            for col in score_cols if col in reba_df.columns
        }

        # Peak frame details
        peak_idx  = int(reba_df["reba_total"].idxmax())
        peak_row  = reba_df.iloc[peak_idx]
        peak_detail = {col: (int(peak_row[col]) if col in score_cols else round(float(peak_row[col]), 3))
                       for col in reba_df.columns if col not in ("frame", "time_s", "risk_level", "dominant_arm")}
        peak_detail["risk_level"]   = str(peak_row["risk_level"])
        peak_detail["dominant_arm"] = str(peak_row["dominant_arm"])
        peak_detail["frame"]        = int(frame_col[peak_idx])
        if time_col is not None:
            peak_detail["time_s"]   = round(float(time_col[peak_idx]), 4)

        # ── Load/coupling review flag (placeholder until verified per-task
        # metadata exists) ─────────────────────────────────────────────────
        # No confirmed per-task load/coupling table exists yet — inventing
        # one from the task name was explicitly ruled out until real,
        # verified protocol data is available. The one safe check available
        # in the meantime: flag when load/coupling look like they were left
        # at the form's defaults (0, 0) rather than deliberately set for
        # THIS trial. This is a "did you forget to set this?" nudge, not a
        # validation against any ground truth.
        load_was_estimated = bool(auto_load and pig_report.get("load_estimate", {}).get("available"))
        reba_params_review_needed = (load == 0 and not load_was_estimated) or coupling == 0

        result = {
            "job_id":            job_id,
            "task_name":         task_name,
            "n_frames":          n_frames,
            "params":            {"load": load, "coupling": coupling,
                                  "activity": activity, "unilateral": unilateral},
            "reba_params_review_needed": reba_params_review_needed,
            "reba_summary":      summary,          # Channel 1: posture severity
            "component_stats":   component_stats,
            "angle_summary":     angle_summary,
            "pig_validation":    pig_report,
            "peak_detail":       peak_detail,
            "channels_2_3":      channels_summary,  # Channel 2 (duration) + Channel 3 (dynamics)
            "output_dir":        str(reba_out),
            "processing_time_s": round(time.time() - t0, 2),
        }

        # Cache to disk
        (reba_out / Artifacts.REBA_RESULT).write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        update_stage(app_config.get_outputs_dir(), task_name, "reba", "completed",
                     reba_load=load, reba_coupling=coupling,
                     reba_activity=activity, reba_unilateral=unilateral,
                     reba_params_review_needed=reba_params_review_needed)

        _cb(100, "Done")
        manager.update_job(job_id, status="completed", progress=100,
                            step="REBA complete", result=result)
        logger.info(f"REBA job {job_id} done in {result['processing_time_s']:.1f}s")

    except Exception as exc:
        logger.error(f"REBA job {job_id} failed: {exc}", exc_info=True)
        manager.update_job(job_id, status="failed", step="Failed", error=str(exc))


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/start")
async def start_reba(
    task_name:  str  = Form(...),
    load:       int  = Form(default=0),
    coupling:   int  = Form(default=0),
    activity:   int  = Form(default=1),
    unilateral: bool = Form(default=False),
    c3d_path:   str  = Form(default=""),
    auto_load:  bool = Form(default=False),
):
    """Start a REBA computation job for a completed task.

    Optional Angle-stage inputs:
      c3d_path:  path to the trial's Vicon C3D. When it contains Plug-in Gait
                 output, validated knee flexion is used for REBA legs and a
                 marker-vs-PiG agreement report is added to the result.
      auto_load: estimate the REBA load score (0/1/2) from the C3D force plates
                 + subject Bodymass instead of the fixed `load` value.
    """
    task_dir   = resolve_task_dir(app_config.get_outputs_dir(), task_name)
    synced_csv = task_dir / Artifacts.SYNCED_60HZ

    if not synced_csv.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"{Artifacts.SYNCED_60HZ} not found for task '{task_name}'. "
                "Run Overlay first to generate the synced CSV."
            ),
        )

    c3d = Path(c3d_path) if c3d_path else None
    if c3d is not None and not c3d.exists():
        raise HTTPException(status_code=404, detail=f"C3D not found: {c3d_path}")

    job_id = manager.create_job()
    manager.update_job(job_id, status="queued", step="Starting…")

    threading.Thread(
        target=_run_reba_job,
        args=(job_id, task_name, synced_csv, task_dir,
              load, coupling, activity, unilateral, c3d, auto_load),
        daemon=True,
    ).start()

    return {"job_id": job_id, "message": f"REBA started for task '{task_name}'"}


@router.get("/{job_id}/stream")
async def stream_reba(job_id: str):
    """SSE progress stream for a REBA job."""
    if manager.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return StreamingResponse(
        manager.stream_job(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{job_id}/result")
async def get_reba_result(job_id: str):
    """Return the completed REBA result dict."""
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == "failed":
        raise HTTPException(status_code=422, detail=job.error or "REBA failed")
    if job.status != "completed":
        raise HTTPException(status_code=202, detail="Job not yet complete")
    return job.result


@router.get("/{task_name}/saved")
async def get_saved_reba(task_name: str):
    """Return cached REBA result if already computed for this task."""
    result_path = (resolve_task_dir(app_config.get_outputs_dir(), task_name)
                   / Artifacts.REBA_DIR / Artifacts.REBA_RESULT)
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="No saved REBA result for this task.")
    with result_path.open(encoding="utf-8") as f:
        return json.load(f)
