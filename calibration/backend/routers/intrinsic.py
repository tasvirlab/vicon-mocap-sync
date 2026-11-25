"""Intrinsic calibration API endpoints."""

from __future__ import annotations
import json
import logging
import threading
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from models.schemas import JobStarted, JobStatus
from utils import app_config
from utils.job_manager import manager
from core.intrinsic_calibration import (
    run_intrinsic_calibration,
    run_intrinsic_calibration_from_images,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_TEMP_DIR    = Path(__file__).parent.parent.parent / "temp"

_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


@router.post("/start", response_model=JobStarted)
async def start_intrinsic(
    video:  Optional[UploadFile] = File(default=None),
    images: List[UploadFile]     = File(default=[]),
    config: str                  = Form("{}"),
):
    """
    Upload a calibration video OR a set of calibration images.

    - ``video``  — single video file (mp4/mov/avi/mkv)
    - ``images`` — one or more image files (jpg/png/etc.)

    Exactly one of the two must be provided.
    """
    cfg = json.loads(config)

    # Validate: need video OR images
    has_video  = video  is not None
    has_images = len(images) > 0
    if not has_video and not has_images:
        raise HTTPException(status_code=422, detail="Provide either 'video' or 'images'.")
    if has_video and has_images:
        raise HTTPException(status_code=422, detail="Provide 'video' OR 'images', not both.")

    job_id = manager.create_job()
    _TEMP_DIR.mkdir(exist_ok=True)

    # Route into exports/<SUBJECT_ID>/ when a subject is given, so per-subject
    # runs land in one place instead of all mixed at the exports root.
    subject_id = str(cfg.get("subject_id", "")).strip()
    exports_dir = app_config.get_exports_dir()
    output_dir = (exports_dir / subject_id) if subject_id else exports_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if has_video:
        # ── Single video path ─────────────────────────────────────────
        manager.update_job(job_id, status="queued", step="Uploading video…")
        video_path = _TEMP_DIR / f"{job_id}_{video.filename}"
        video_path.write_bytes(await video.read())

        def _run_video():
            try:
                manager.update_job(job_id, status="running", step="Starting…", progress=1)

                def _cb(pct, step):
                    manager.update_job(job_id, progress=pct, step=step)

                result = run_intrinsic_calibration(
                    str(video_path), str(output_dir), cfg, progress_cb=_cb
                )
                manager.update_job(
                    job_id, status="completed", progress=100,
                    step="Calibration complete", result=result,
                )
                logger.info(f"Job {job_id} completed — RMS={result['rms_px']:.4f} px")
            except Exception as exc:
                logger.error(f"Job {job_id} failed: {exc}", exc_info=True)
                manager.update_job(job_id, status="failed", step="Failed", error=str(exc))
            finally:
                video_path.unlink(missing_ok=True)

        threading.Thread(target=_run_video, daemon=True).start()

    else:
        # ── Multi-image path ──────────────────────────────────────────
        manager.update_job(job_id, status="queued", step=f"Uploading {len(images)} image(s)…")

        img_dir = _TEMP_DIR / job_id
        img_dir.mkdir(exist_ok=True)
        image_paths: List[Path] = []

        for upload in images:
            ext = Path(upload.filename or "img.jpg").suffix.lower()
            if ext not in _IMAGE_EXTS:
                continue
            dest = img_dir / f"{len(image_paths):04d}{ext}"
            dest.write_bytes(await upload.read())
            image_paths.append(dest)

        if not image_paths:
            raise HTTPException(
                status_code=422,
                detail="No valid image files received (expected jpg/png/bmp/tiff)."
            )

        def _run_images():
            try:
                manager.update_job(job_id, status="running", step="Starting…", progress=1)

                def _cb(pct, step):
                    manager.update_job(job_id, progress=pct, step=step)

                result = run_intrinsic_calibration_from_images(
                    [str(p) for p in image_paths], str(output_dir), cfg, progress_cb=_cb
                )
                manager.update_job(
                    job_id, status="completed", progress=100,
                    step="Calibration complete", result=result,
                )
                logger.info(f"Job {job_id} completed — RMS={result['rms_px']:.4f} px")
            except Exception as exc:
                logger.error(f"Job {job_id} failed: {exc}", exc_info=True)
                manager.update_job(job_id, status="failed", step="Failed", error=str(exc))
            finally:
                import shutil
                shutil.rmtree(str(img_dir), ignore_errors=True)

        threading.Thread(target=_run_images, daemon=True).start()

    return JobStarted(job_id=job_id, message="Intrinsic calibration started")


@router.get("/{job_id}/stream")
async def stream_intrinsic(job_id: str):
    if manager.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return StreamingResponse(
        manager.stream_job(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{job_id}/status", response_model=JobStatus)
async def get_status(job_id: str):
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(
        job_id=job_id, status=job.status,
        progress=job.progress, step=job.step,
        message=job.message or "",
    )


@router.get("/{job_id}/result")
async def get_result(job_id: str):
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == "failed":
        raise HTTPException(status_code=422, detail=job.error or "Calibration failed")
    if job.status != "completed":
        raise HTTPException(status_code=202, detail="Job not yet complete")
    return job.result


@router.get("/{job_id}/file/{filename}")
async def download_file(job_id: str, filename: str):
    job = manager.get_job(job_id)
    if job is None or job.result is None:
        raise HTTPException(status_code=404, detail="Job or result not found")
    files: dict = job.result.get("files", {})
    for key, path in files.items():
        if Path(path).name == filename or key == filename:
            p = Path(path)
            if p.exists():
                return FileResponse(str(p), filename=p.name)
    raise HTTPException(status_code=404, detail=f"File {filename!r} not found")
