"""
pose2d.py — /api/pose2d endpoints
====================================
Run RTMPose-WholeBody over a batch of trials' <task>_rgb.mp4 to produce
<task>_pose2d.csv — the 2D keypoint input dataset_export.py consumes.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import StreamingResponse

from utils import app_config
from utils.job_manager import manager
from utils.task_paths import (discover_task_dirs, task_dir as resolve_task_dir,
                              Artifacts, update_stage)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Background worker ────────────────────────────────────────────────────────────

def _run_pose2d_job(
    job_id: str,
    task_names: list[str],
    conf_threshold: float,
    t_t10: float,
    t_l45: float,
    device: str,
    mode: str,
):
    t0 = time.time()
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from core import rtmpose_pipeline as rp

        manager.update_job(job_id, status="running", progress=1, step="Starting…")

        def _cb(pct: int, step: str):
            manager.update_job(job_id, progress=pct, step=step)

        trials = [(name, resolve_task_dir(app_config.get_outputs_dir(), name)) for name in task_names]
        results = rp.process_trials_batch(
            trials, progress_cb=_cb,
            conf_threshold=conf_threshold, t_t10=t_t10, t_l45=t_l45,
            device=device, mode=mode,
        )

        n_ok = sum(1 for r in results if r["status"] == "ok")
        result = {
            "job_id": job_id,
            "n_trials": len(task_names),
            "n_succeeded": n_ok,
            "n_failed": len(task_names) - n_ok,
            "trials": results,
            "params": {"conf_threshold": conf_threshold, "t_t10": t_t10, "t_l45": t_l45,
                       "device": device, "mode": mode},
            "processing_time_s": round(time.time() - t0, 2),
        }

        for _tid, _ in trials:
            update_stage(app_config.get_outputs_dir(), _tid, "pose2d", "completed")
        manager.update_job(job_id, status="completed", progress=100,
                            step="Done", result=result)
        logger.info(f"pose2d job {job_id}: {n_ok}/{len(task_names)} trials succeeded")

    except Exception as exc:
        logger.error(f"pose2d job {job_id} failed: {exc}", exc_info=True)
        manager.update_job(job_id, status="failed", step="Failed", error=str(exc))


# ── Endpoints ────────────────────────────────────────────────────────────────────

@router.get("/tasks")
def list_pose2d_tasks():
    """List trial folders that have a <task>_rgb.mp4 (eligible for pose2d), with
    their current <task>_pose2d.csv status."""
    tasks = []
    for task_id, trial_dir in sorted(discover_task_dirs(app_config.get_outputs_dir())):
        rgb = trial_dir / Artifacts.RGB_VIDEO
        if not rgb.exists():
            continue
        pose2d = trial_dir / Artifacts.POSE2D_RTMPOSE
        tasks.append({
            "task_name": task_id,
            "has_pose2d": pose2d.exists(),
        })
    return {"tasks": tasks}


@router.post("/start")
async def start_pose2d(
    task_names: str = Form(...),         # JSON-encoded list[str]
    conf_threshold: float = Form(default=0.3),
    # Keep in sync with DEFAULT_T_T10 / DEFAULT_T_L45 in core/rtmpose_pipeline.py
    # (not imported here: rtmpose_pipeline is deferred so this router loads
    # without the rtmlib dependency installed).
    t_t10: float = Form(default=0.35),
    t_l45: float = Form(default=0.95),
    device: str = Form(default="cpu"),
    mode: str = Form(default="balanced"),
):
    """Start an RTMPose-WholeBody batch job over the given trials."""
    try:
        names: list[str] = json.loads(task_names)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="task_names must be a JSON-encoded list of strings")
    if not names:
        raise HTTPException(status_code=400, detail="No trials selected")

    missing = [n for n in names if not (resolve_task_dir(app_config.get_outputs_dir(), n) / Artifacts.RGB_VIDEO).exists()]
    if missing:
        raise HTTPException(status_code=404, detail=f"Missing rgb.mp4 for: {missing}")

    job_id = manager.create_job()
    manager.update_job(job_id, status="queued", step="Starting…")

    threading.Thread(
        target=_run_pose2d_job,
        args=(job_id, names, conf_threshold, t_t10, t_l45, device, mode),
        daemon=True,
    ).start()

    return {"job_id": job_id, "message": f"RTMPose-WholeBody started for {len(names)} trial(s)"}


@router.get("/{job_id}/stream")
async def stream_pose2d(job_id: str):
    if manager.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return StreamingResponse(
        manager.stream_job(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{job_id}/result")
async def get_pose2d_result(job_id: str):
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == "failed":
        raise HTTPException(status_code=422, detail=job.error or "pose2d job failed")
    if job.status != "completed":
        raise HTTPException(status_code=202, detail="Job not yet complete")
    return job.result


@router.post("/visualize")
async def visualize_pose2d(
    task_name: str = Form(...),
    conf_threshold: float = Form(default=0.3),
    # Keep in sync with DEFAULT_T_T10 / DEFAULT_T_L45 in core/rtmpose_pipeline.py
    # (not imported here: rtmpose_pipeline is deferred so this router loads
    # without the rtmlib dependency installed).
    t_t10: float = Form(default=0.35),
    t_l45: float = Form(default=0.95),
    device: str = Form(default="cpu"),
    mode: str = Form(default="balanced"),
):
    """Run RTMPose-WholeBody on just the first frame and return an annotated
    preview image (base64 JPEG) — use this to verify the joint mapping before
    running a full batch. Does not write any pose2d.csv."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core import rtmpose_pipeline as rp

    trial_dir = resolve_task_dir(app_config.get_outputs_dir(), task_name)
    video_path = trial_dir / Artifacts.RGB_VIDEO
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"No rgb.mp4 found for task '{task_name}'")

    out_path = trial_dir / Artifacts.QA_DIR / Artifacts.QA_POSE2D_CHECK
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rp.visualize_first_frame(
            video_path, out_path,
            conf_threshold=conf_threshold, t_t10=t_t10, t_l45=t_l45,
            device=device, mode=mode,
        )
    except (FileNotFoundError, RuntimeError) as e:
        raise HTTPException(status_code=422, detail=str(e))

    image_b64 = base64.b64encode(out_path.read_bytes()).decode("ascii")
    return {"image_b64": image_b64}


@router.get("/live")
def live_pose2d(
    task_name: str,
    conf_threshold: float = 0.3,
    # Keep in sync with DEFAULT_T_T10 / DEFAULT_T_L45 in core/rtmpose_pipeline.py
    t_t10: float = 0.35,
    t_l45: float = 0.95,
    device: str = "cpu",
    mode: str = "balanced",
):
    """Live MJPEG stream of the RTMPose 21-joint skeleton for one trial, frame by
    frame at model speed. Transient — writes no CSV/video. The stream (and its
    inference) stops when the client closes the connection (Stop button / navigate away)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core import rtmpose_pipeline as rp

    trial_dir = resolve_task_dir(app_config.get_outputs_dir(), task_name)
    video_path = trial_dir / Artifacts.RGB_VIDEO
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"No rgb.mp4 found for task '{task_name}'")

    gen = rp.stream_video_annotated(
        video_path, conf_threshold=conf_threshold, t_t10=t_t10, t_l45=t_l45,
        device=device, mode=mode,
    )
    return StreamingResponse(
        gen,
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
