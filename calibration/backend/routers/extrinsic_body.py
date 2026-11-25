"""
extrinsic_body.py  (router)
===========================
Body-mode extrinsic calibration API endpoints.

Routes
------
  POST /api/extrinsic-body/frame    — extract a single video frame as JPEG
  GET  /api/extrinsic-body/joints   — return joint definitions (6 / 8 mode)
  POST /api/extrinsic-body/start    — start a body calibration job
  GET  /api/extrinsic-body/{id}/stream  — SSE progress stream
  GET  /api/extrinsic-body/{id}/result  — fetch completed result
  GET  /api/extrinsic-body/{id}/file/{filename} — download output file
"""

from __future__ import annotations
import base64
import json
import logging
import threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from core.extrinsic_body import (
    extract_video_frame, run_body_extrinsic,
    JOINTS_6, JOINTS_8, JOINT_DESCRIPTIONS,
)
from core.timestamp_sync import (
    parse_xcp_start_time,
    probe_video_start_time,
    compute_timestamp_sync,
)
from utils import app_config
from utils.job_manager import manager

logger    = logging.getLogger(__name__)
router    = APIRouter()

_TEMP     = Path(__file__).parent.parent.parent / "temp"


# ── Frame extraction ──────────────────────────────────────────────────────────

@router.post("/frame")
async def get_frame(
    video:     UploadFile = File(...),
    frame_idx: int        = Form(0),
    rotate:    str        = Form("none"),
):
    """
    Extract a single frame from a video.

    Returns JSON with:
      frame_b64    — JPEG as base64 string
      total_frames — total frame count in the video
      fps          — video frame rate
    """
    _TEMP.mkdir(exist_ok=True)
    tmp = _TEMP / f"frame_{video.filename}"
    try:
        tmp.write_bytes(await video.read())
        jpeg, total, fps = extract_video_frame(str(tmp), frame_idx, rotate)
        return {
            "frame_b64":    base64.b64encode(jpeg).decode("ascii"),
            "total_frames": total,
            "fps":          fps,
        }
    finally:
        tmp.unlink(missing_ok=True)


# ── Joint definitions ─────────────────────────────────────────────────────────

@router.get("/joints")
def get_joints():
    """Return the joint label lists and descriptions for both modes."""
    return {
        "joints_6": JOINTS_6,
        "joints_8": JOINTS_8,
        "descriptions": JOINT_DESCRIPTIONS,
    }


# ── Calibration job ───────────────────────────────────────────────────────────

@router.post("/start")
async def start_body(
    video:       UploadFile           = File(...),
    vicon:       UploadFile           = File(...),
    intrinsic:   UploadFile           = File(...),
    xcp:         Optional[UploadFile] = File(default=None),
    clicks:      str                  = Form("{}"),
    joint_count: int                  = Form(6),
    vicon_tz:    float                = Form(0.0),
    rotate:      str                  = Form("none"),
    subject_id:  str                  = Form(""),
):
    """
    Start a body-marker extrinsic calibration job.

    clicks    — JSON string: {"frame_idx": [[x, y], ...], ...}
    xcp       — optional Vicon .xcp file for wall-clock timestamp sync
    vicon_tz  — Vicon PC UTC offset in hours (e.g. 9.0 for UTC+9)

    Returns:
        {"job_id": str, "message": str}
    """
    _TEMP.mkdir(exist_ok=True)
    job_id = manager.create_job()
    manager.update_job(job_id, status="queued", step="Uploading files…")

    # Route into exports/<SUBJECT_ID>/ when a subject is given, so per-subject
    # runs land in one place instead of all mixed at the exports root.
    subj_id = subject_id.strip()
    exports_dir = app_config.get_exports_dir()
    output_dir = (exports_dir / subj_id) if subj_id else exports_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    suf_v = Path(video.filename or "video.mp4").suffix
    suf_c = Path(vicon.filename or "vicon.csv").suffix
    suf_i = Path(intrinsic.filename or "intrinsic.json").suffix

    vid_path  = _TEMP / f"{job_id}_video{suf_v}"
    vic_path  = _TEMP / f"{job_id}_vicon{suf_c}"
    int_path  = _TEMP / f"{job_id}_intrinsic{suf_i}"
    xcp_path  = None

    vid_path.write_bytes(await video.read())
    vic_path.write_bytes(await vicon.read())
    int_path.write_bytes(await intrinsic.read())
    if xcp is not None:
        xcp_bytes = await xcp.read()
        if xcp_bytes:
            xcp_path = _TEMP / f"{job_id}_session.xcp"
            xcp_path.write_bytes(xcp_bytes)

    clicks_dict = json.loads(clicks)

    def _run():
        # Compute XCP wall-clock sync hint (non-fatal if fails)
        sync_hint: Optional[float] = None
        if xcp_path and xcp_path.exists():
            try:
                manager.update_job(job_id, status="running",
                                   step="Reading wall-clock timestamps…", progress=2)
                vid_start, vid_fps, vid_tz, vid_total = probe_video_start_time(str(vid_path))
                tz_h     = vicon_tz if vicon_tz != 0.0 else vid_tz
                vic_start = parse_xcp_start_time(xcp_path, tz_offset_hours=tz_h)
                ts_sync  = compute_timestamp_sync(
                    vid_start, vic_start, vid_fps, 100.0, vid_total, 999_999)
                # pnp_solver offset convention: t_vicon = t_video + offset_s
                # compute_timestamp_sync delta_s sign is opposite → negate
                sync_hint = -ts_sync["offset_seconds"]
                logger.info(
                    f"XCP sync hint: delta={ts_sync['offset_seconds']:+.3f} s "
                    f"→ pnp offset hint={sync_hint:+.3f} s"
                )
            except Exception as exc:
                logger.warning(f"XCP sync failed (non-fatal): {exc} — using full grid-search")

        try:
            manager.update_job(job_id, status="running", step="Starting…", progress=3)
            result = run_body_extrinsic(
                str(vid_path), str(vic_path), str(int_path),
                clicks_dict, joint_count, str(output_dir),
                progress_cb=lambda pct, step: manager.update_job(
                    job_id, progress=pct, step=step),
                sync_offset_hint_s=sync_hint,
                rotate=rotate,
            )
            manager.update_job(
                job_id, status="completed", progress=100,
                step="Body calibration complete", result=result,
            )
            logger.info(
                f"Body job {job_id} done — RMS={result['rms_px']:.4f} px  "
                f"quality={result['quality']}"
            )
        except Exception as exc:
            logger.error(f"Body job {job_id} failed: {exc}", exc_info=True)
            manager.update_job(
                job_id, status="failed", step="Failed", error=str(exc))
        finally:
            for p in (vid_path, vic_path, int_path):
                p.unlink(missing_ok=True)
            if xcp_path and xcp_path.exists():
                xcp_path.unlink(missing_ok=True)

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "message": "Body extrinsic calibration started"}


# ── Streaming / results ───────────────────────────────────────────────────────

@router.get("/{job_id}/stream")
async def stream_body(job_id: str):
    if manager.get_job(job_id) is None:
        raise HTTPException(404, "Job not found")
    return StreamingResponse(
        manager.stream_job(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{job_id}/status")
async def get_status(job_id: str):
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return {"job_id": job_id, "status": job.status,
            "progress": job.progress, "step": job.step}


@router.get("/{job_id}/result")
async def get_result(job_id: str):
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status == "failed":
        raise HTTPException(422, job.error or "Calibration failed")
    if job.status != "completed":
        raise HTTPException(202, "Job not yet complete")
    return job.result


@router.get("/{job_id}/file/{filename}")
async def download_file(job_id: str, filename: str):
    job = manager.get_job(job_id)
    if job is None or job.result is None:
        raise HTTPException(404, "Job or result not found")
    for path in (job.result.get("files") or {}).values():
        p = Path(path)
        if p.name == filename or path == filename:
            if p.exists():
                return FileResponse(str(p), filename=p.name)
    raise HTTPException(404, f"File '{filename}' not found")
