"""
overlay.py — /api/overlay endpoints
=====================================
Generate skeleton overlay video from a completed sync task.

Uses the sync JSON saved under outputs/<task_name>/ to locate
video and Vicon CSV paths, then runs the full overlay pipeline.
"""

from __future__ import annotations
import base64
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from models.schemas import JobStarted
from utils import app_config
from utils.job_manager import manager
from utils.task_paths import task_dir as resolve_task_dir, Artifacts, update_stage
from core.overlay_pipeline import (run_overlay, preview_single_frame,
                                   NORMALIZED_WIDTH, NORMALIZED_HEIGHT)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Background worker ──────────────────────────────────────────────────────────

def _run_overlay_job(
    job_id: str,
    task_name: str,
    video_path: Path,
    vicon_csv: Path,
    intrinsics_path: Path,
    extrinsics_path: Path,
    sync_json: Path,
    output_dir: Path,
    rotate: Optional[str],
    export_plain: bool = True,
    blur_faces_rgb: bool = False,
    export_2d_csv: bool = True,
):
    t0 = time.time()
    try:
        def _cb(pct, step):
            manager.update_job(job_id, progress=pct, step=step)

        manager.update_job(job_id, status="running", progress=3,
                            step="Loading calibration…")

        stats = run_overlay(
            task_name=task_name,
            video_path=video_path,
            vicon_csv=vicon_csv,
            intrinsics_path=intrinsics_path,
            extrinsics_path=extrinsics_path,
            sync_path=sync_json,
            output_dir=output_dir,
            rotate=rotate,
            export_plain=export_plain,
            blur_faces_rgb=blur_faces_rgb,
            export_2d_csv=export_2d_csv,
            progress_cb=_cb,
        )

        has_diag = (output_dir / Artifacts.QA_DIR / Artifacts.QA_SYNC_DIAG).exists()
        elapsed  = round(time.time() - t0, 2)

        # Record the video spec the overlay actually measured/produced, so the
        # trial states its own resolution and true frame rate (60.0283, not the
        # nominal 60) without anyone re-probing the file.
        update_stage(
            app_config.get_outputs_dir(), task_name, "overlay", "completed",
            video_fps=stats.get("video_fps"),
            video_resolution=f"{NORMALIZED_WIDTH}x{NORMALIZED_HEIGHT}",
            vicon_fps=stats.get("vicon_fps"),
        )

        result = {
            **stats,
            "job_id":             job_id,
            "has_diagnostic_png": has_diag,
            "processing_time_s":  elapsed,
        }

        manager.update_job(job_id, status="completed", progress=100,
                            step="Overlay complete", result=result)
        logger.info(f"Overlay job {job_id} done in {elapsed:.1f}s  task={task_name}")

    except Exception as exc:
        logger.error(f"Overlay job {job_id} failed: {exc}", exc_info=True)
        manager.update_job(job_id, status="failed", step="Failed", error=str(exc))


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/start", response_model=JobStarted)
async def start_overlay(
    task_name:      str  = Form(...),
    rotate:         Optional[str] = Form(default=None),
    export_plain:   bool = Form(default=True),
    blur_faces_rgb: bool = Form(default=False),
    export_2d_csv:  bool = Form(default=True),
    intrinsics:     Optional[UploadFile] = File(default=None),
    extrinsics:     Optional[UploadFile] = File(default=None),
):
    """
    Start a skeleton overlay job for a previously synced task.

    Required:
      task_name — must have a completed sync in outputs/<task_name>/

    Optional:
      intrinsics     — upload intrinsic.json if not in task folder
      extrinsics     — upload extrinsics.json if not in task folder
      rotate         — override video rotation ('ccw90', 'cw90', '180', 'none')
      blur_faces_rgb — blur every detected face in <task>_rgb.mp4 (default False)
      export_2d_csv  — also write synced_vicon2d_60hz.csv (Vicon-PROJECTED 2D,
                       NOT RTMPose) alongside the 3D synced.csv (default True)

    Video and Vicon CSV paths are read from the saved sync JSON.
    """
    task_dir  = resolve_task_dir(app_config.get_outputs_dir(), task_name)
    sync_json = task_dir / Artifacts.SYNC_JSON

    if not sync_json.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No sync data for task '{task_name}'. Run Sync first."
        )

    with sync_json.open(encoding="utf-8") as f:
        sync_meta = json.load(f)

    # Locate video and vicon CSV from sync metadata
    video_path = Path(sync_meta["video_path"])
    vicon_path = Path(sync_meta["vicon_csv_path"])

    if not video_path.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Video file not found at {video_path}. "
                "The temp folder may have been cleaned. Re-run Sync to re-upload."
            )
        )
    if not vicon_path.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Vicon CSV not found at {vicon_path}. "
                "Re-run Sync to re-upload."
            )
        )

    # Resolve intrinsics/extrinsics.
    # Priority order — MUST match /preview's and /extrinsic-correction/save's
    # resolution exactly, or a spatial correction saved into *_overlay.json can
    # render correctly in the live preview while the full render silently uses
    # the other (uncorrected) file:
    #   1. Fresh upload in this overlay request (allows re-use of different calibration)
    #   2. An *_overlay.json already on disk — e.g. from a previous override upload,
    #      or from core/extrinsic_correction.py's save_correction() baking a saved
    #      spatial correction into it
    #   3. temp/<task_name>/intrinsic.json / extrinsics.json — uploaded during Sync
    # Calibration files are NOT stored in the outputs/ folder by design.
    task_temp = app_config.get_temp_dir() / task_name
    task_temp.mkdir(parents=True, exist_ok=True)

    intr_overlay_path = task_temp / "intrinsic_overlay.json"
    if intrinsics:
        intr_overlay_path.write_bytes(await intrinsics.read())
        intr_path = intr_overlay_path
    elif intr_overlay_path.exists():
        intr_path = intr_overlay_path
    else:
        intr_path = task_temp / "intrinsic.json"   # uploaded during Sync step

    extr_overlay_path = task_temp / "extrinsics_overlay.json"
    if extrinsics:
        extr_overlay_path.write_bytes(await extrinsics.read())
        extr_path = extr_overlay_path
    elif extr_overlay_path.exists():
        extr_path = extr_overlay_path
    else:
        extr_path = task_temp / "extrinsics.json"  # uploaded during Sync step

    if not intr_path.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                "Intrinsics not found. Re-run the Sync step and upload "
                "intrinsic.json there, or upload it directly here."
            )
        )
    if not extr_path.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                "Extrinsics not found. Re-run the Sync step and upload "
                "extrinsics.json there, or upload it directly here."
            )
        )

    job_id = manager.create_job()
    manager.update_job(job_id, status="queued", step="Starting…")

    threading.Thread(
        target=_run_overlay_job,
        args=(job_id, task_name, video_path, vicon_path,
              intr_path, extr_path, sync_json, task_dir, rotate, export_plain,
              blur_faces_rgb, export_2d_csv),
        daemon=True,
    ).start()

    return JobStarted(job_id=job_id,
                      message=f"Overlay started for task '{task_name}'")


@router.get("/{job_id}/stream")
async def stream_overlay(job_id: str):
    """SSE progress stream for an overlay job."""
    if manager.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return StreamingResponse(
        manager.stream_job(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{job_id}/result")
async def get_overlay_result(job_id: str):
    """Return the completed overlay result dict."""
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == "failed":
        raise HTTPException(status_code=422, detail=job.error or "Overlay failed")
    if job.status != "completed":
        raise HTTPException(status_code=202, detail="Job not yet complete")
    return job.result


@router.get("/{task_name}/info")
async def get_task_info(task_name: str):
    """
    Return sync metadata and video stats for the preview panel.

    Used by the frontend to populate the frame slider range and read the
    current saved manual_frame_offset without starting a job.
    """
    task_dir  = resolve_task_dir(app_config.get_outputs_dir(), task_name)
    sync_json = task_dir / Artifacts.SYNC_JSON
    if not sync_json.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No sync data for task '{task_name}'. Run Sync first."
        )

    with sync_json.open(encoding="utf-8") as f:
        sync = json.load(f)

    video_path = Path(sync["video_path"])
    if not video_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Video file missing — re-run Sync to re-upload."
        )

    cap          = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = float(cap.get(cv2.CAP_PROP_FPS)) or float(sync.get("video_fps", 60.0))
    cap.release()

    return {
        "task_name":           task_name,
        "total_frames":        total_frames,
        "video_fps":           round(video_fps, 3),
        "vicon_fps":           float(sync.get("vicon_fps", 100.0)),
        "manual_frame_offset": int(sync.get("manual_frame_offset", 0)),
        "sync_offset_s":       float(sync.get("sync_offset_s", 0.0)),
        "sync_method":         sync.get("sync_method", "unknown"),
    }


@router.post("/preview")
async def preview_frame(
    task_name:                    str           = Form(...),
    video_frame:                  int           = Form(0),
    manual_frame_offset_override: Optional[int] = Form(default=None),
    rotate:                       Optional[str] = Form(default=None),
    dx:     float = Form(0.0),
    dy:     float = Form(0.0),
    dz:     float = Form(0.0),
    droll:  float = Form(0.0),
    dpitch: float = Form(0.0),
    dyaw:   float = Form(0.0),
):
    """
    Render one video frame with skeleton overlay; return base64 JPEG.

    Used by the fine-tune panel so the user can adjust manual_frame_offset
    (temporal) and dx/dy/dz/droll/dpitch/dyaw (spatial — not yet saved, see
    core/extrinsic_correction.py) and see the result instantly before running
    a full overlay job or committing a spatial correction.

    Returns:
        {
          image_b64:    str    — base64-encoded JPEG,
          in_range:     bool   — whether the frame maps to valid Vicon data,
          vicon_time_s: float  — Vicon timestamp for this frame,
          total_frames: int,
          video_fps:    float,
        }
    """
    task_dir  = resolve_task_dir(app_config.get_outputs_dir(), task_name)
    sync_json = task_dir / Artifacts.SYNC_JSON
    if not sync_json.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No sync data for task '{task_name}'. Run Sync first."
        )

    with sync_json.open(encoding="utf-8") as f:
        sync = json.load(f)

    video_path = Path(sync["video_path"])
    vicon_path = Path(sync["vicon_csv_path"])

    if not video_path.exists():
        raise HTTPException(status_code=400, detail="Video file missing — re-run Sync.")
    if not vicon_path.exists():
        raise HTTPException(status_code=400, detail="Vicon CSV missing — re-run Sync.")

    task_temp = app_config.get_temp_dir() / task_name
    intr_path = task_temp / "intrinsic.json"
    extr_path = task_temp / "extrinsics.json"

    # Allow overlay-step uploads to substitute
    if (task_temp / "intrinsic_overlay.json").exists():
        intr_path = task_temp / "intrinsic_overlay.json"
    if (task_temp / "extrinsics_overlay.json").exists():
        extr_path = task_temp / "extrinsics_overlay.json"

    if not intr_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Intrinsics not found. Re-run Sync or upload via Overlay → Start."
        )
    if not extr_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Extrinsics not found. Re-run Sync or upload via Overlay → Start."
        )

    try:
        result = preview_single_frame(
            video_path=video_path,
            vicon_csv=vicon_path,
            intrinsics_path=intr_path,
            extrinsics_path=extr_path,
            sync=sync,
            video_frame=video_frame,
            manual_frame_offset_override=manual_frame_offset_override,
            rotate=rotate,
            extrinsic_delta={"dx": dx, "dy": dy, "dz": dz,
                              "droll": droll, "dpitch": dpitch, "dyaw": dyaw},
        )
    except Exception as exc:
        logger.error(
            f"Preview failed for task '{task_name}' frame {video_frame}: {exc}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "image_b64":    base64.b64encode(result["jpeg"]).decode("ascii"),
        "in_range":     result["in_range"],
        "vicon_time_s": result["vicon_time_s"],
        "total_frames": result["total_frames"],
        "video_fps":    result["video_fps"],
    }
