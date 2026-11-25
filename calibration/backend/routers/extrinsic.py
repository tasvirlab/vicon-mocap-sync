"""Extrinsic calibration API endpoints."""

from __future__ import annotations
import json
import logging
import threading
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from typing import Optional

from models.schemas import JobStarted, JobStatus
from utils import app_config
from utils.job_manager import manager
from core.extrinsic_calibration import run_extrinsic_calibration
from core.timestamp_sync import (
    parse_xcp_start_time,
    probe_video_start_time,
    compute_timestamp_sync,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_TEMP_DIR    = Path(__file__).parent.parent.parent / "temp"


@router.post("/start", response_model=JobStarted)
async def start_extrinsic(
    video:     UploadFile = File(...),
    vicon:     Optional[UploadFile] = File(None),
    intrinsic: UploadFile = File(...),
    xcp:       Optional[UploadFile] = File(None),
    config:    str = Form("{}"),
):
    """Upload video + Vicon file + intrinsic params + optional .xcp, start background job.

    When an .xcp file is provided, the app extracts the Vicon recording START_TIME
    and pairs it with the video's creation timestamp to compute the exact Vicon frame
    index that corresponds to each video frame.  This removes any temporal uncertainty
    in the video_frame → vicon_frame mapping used during solvePnP.
    """
    cfg = json.loads(config)

    subject_id = str(cfg.get("subject_id", "")).strip()

    job_id = manager.create_job()
    manager.update_job(job_id, status="queued", step="Uploading files…")

    _TEMP_DIR.mkdir(exist_ok=True)

    video_bytes     = await video.read()
    vicon_bytes     = await vicon.read() if vicon else None
    intrinsic_bytes = await intrinsic.read()
    xcp_bytes       = await xcp.read() if xcp else None

    video_path     = _TEMP_DIR / f"{job_id}_video{Path(video.filename).suffix}"
    vicon_path     = _TEMP_DIR / f"{job_id}_vicon{Path(vicon.filename).suffix}" if vicon else None
    intrinsic_path = _TEMP_DIR / f"{job_id}_intrinsic{Path(intrinsic.filename).suffix}"

    video_path.write_bytes(video_bytes)
    if vicon_path is not None and vicon_bytes is not None:
        vicon_path.write_bytes(vicon_bytes)
    intrinsic_path.write_bytes(intrinsic_bytes)

    xcp_path = None
    if xcp_bytes and xcp:
        xcp_path = _TEMP_DIR / f"{job_id}_session.xcp"
        xcp_path.write_bytes(xcp_bytes)

    # Route into exports/<SUBJECT_ID>/ when a subject is given, so per-subject
    # runs land in one place instead of all mixed at the exports root.
    exports_dir = app_config.get_exports_dir()
    output_dir_path = (exports_dir / subject_id) if subject_id else exports_dir
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_dir = str(output_dir_path)

    def _run():
        try:
            manager.update_job(job_id, status="running", step="Starting…", progress=1)

            def _cb(pct: int, step: str):
                manager.update_job(job_id, progress=pct, step=step)

            # ── Compute timestamp-based sync offset (if .xcp provided) ──────────
            ts_sync_info: Optional[dict] = None
            if xcp_path and xcp_path.exists():
                _cb(3, "Reading wall-clock timestamps from .xcp and video…")
                try:
                    vid_start, vid_fps, vid_tz, vid_total_frames = probe_video_start_time(str(video_path))
                    tz_h = cfg.get("vicon_tz_offset_hours", vid_tz)
                    vic_start = parse_xcp_start_time(xcp_path, tz_offset_hours=tz_h)
                    ts_sync_info = compute_timestamp_sync(
                        video_start_utc=vid_start,
                        vicon_start_utc=vic_start,
                        video_fps=vid_fps,
                        vicon_fps=cfg.get("vicon_fps", 100.0),
                        video_total_frames=vid_total_frames,
                        vicon_total_frames=999999,   # Vicon frames not yet counted here
                    )
                    logger.info(
                        f"XCP timestamp sync: Δ={ts_sync_info['offset_ms']:+.1f}ms, "
                        f"video starts at frame {ts_sync_info['sync_start_video_frame']}, "
                        f"vicon starts at frame {ts_sync_info['sync_start_vicon_frame']}"
                    )
                    # Inject computed vicon_frame into sync config so extrinsic pipeline uses it
                    cfg.setdefault("sync", {})
                    if cfg["sync"].get("mode", "manual") != "manual" or cfg["sync"].get("vicon_frame", 0) == 0:
                        cfg["sync"]["vicon_frame"] = ts_sync_info["sync_start_vicon_frame"]
                        cfg["sync"]["mode"] = "timecode"
                        logger.info(
                            f"Injecting XCP vicon_frame={ts_sync_info['sync_start_vicon_frame']} "
                            f"into extrinsic sync config"
                        )
                except Exception as e:
                    logger.warning(f"XCP timestamp sync failed (non-fatal): {e} — using manual sync config")

            result = run_extrinsic_calibration(
                str(video_path), str(vicon_path) if vicon_path else None, str(intrinsic_path),
                output_dir, cfg, progress_cb=_cb,
            )

            # Attach timestamp sync metadata to result
            if ts_sync_info:
                result["timestamp_sync"] = {
                    "video_start_utc":    ts_sync_info["video_start_iso"],
                    "vicon_start_utc":    ts_sync_info["vicon_start_iso"],
                    "offset_ms":          ts_sync_info["offset_ms"],
                    "sync_start_vicon_frame": ts_sync_info["sync_start_vicon_frame"],
                }

            manager.update_job(
                job_id, status="completed", progress=100,
                step="Extrinsic calibration complete", result=result,
            )
            logger.info(f"Job {job_id} completed — RMS={result['rms_px']:.4f} px")
        except Exception as exc:
            logger.error(f"Job {job_id} failed: {exc}", exc_info=True)
            manager.update_job(
                job_id, status="failed", step="Failed", error=str(exc),
            )
        finally:
            for p in [video_path, intrinsic_path]:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
            if vicon_path is not None and vicon_path.exists():
                try:
                    vicon_path.unlink(missing_ok=True)
                except Exception:
                    pass
            if xcp_path and xcp_path.exists():
                try:
                    xcp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    threading.Thread(target=_run, daemon=True).start()
    return JobStarted(job_id=job_id, message="Extrinsic calibration started")


@router.get("/{job_id}/stream")
async def stream_extrinsic(job_id: str):
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
        progress=job.progress, step=job.step, message=job.message or "",
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
