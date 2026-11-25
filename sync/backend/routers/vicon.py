"""
vicon.py — /api/vicon endpoints
=================================
Process a Vicon C3D file → clean 21-joint CSV ready for the Sync step.

Detects and uses Plug-in Gait anatomical joint centres when available:
  RHIP/LHIP ← RFEO/LFEO (true HJC), RKNE/LKNE ← RTIO/LTIO (true KJC),
  RANK/LANK ← RFOO/LFOO (true AJC), shoulders/elbows/wrists from PiG too.
"""

from __future__ import annotations
import logging
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from models.schemas import JobStarted
from utils import app_config
from utils.job_manager import manager
from utils.task_paths import task_dir, init_metadata, update_stage, split_task_id
from core.c3d_pipeline import c3d_to_clean_csv

logger = logging.getLogger(__name__)
router = APIRouter()


def _run_c3d_job(
    job_id: str, c3d_path: Path, out_dir: Path,
    max_gap: int, raw_nexus_csv: Optional[Path],
    out_stem: Optional[str] = None,
    raw_nexus_csv_filename: Optional[str] = None,
    fallback_csv_auto_note: Optional[str] = None,
):
    t0 = time.time()
    try:
        def _cb(pct, step):
            manager.update_job(job_id, progress=pct, step=step)

        manager.update_job(job_id, status="running", progress=3,
                            step="Reading C3D…")

        report = c3d_to_clean_csv(
            c3d_path, out_dir, max_gap,
            raw_nexus_csv=raw_nexus_csv,
            progress_cb=_cb,
            out_stem=out_stem,
            raw_nexus_csv_filename=raw_nexus_csv_filename,
        )
        report["processing_time_s"] = round(time.time() - t0, 2)
        report["job_id"] = job_id
        if fallback_csv_auto_note:
            report["fallback_csv_auto_note"] = fallback_csv_auto_note

        # metadata.json is the trial's provenance record and the marker
        # discover_trials() looks for — until now C3D never wrote it at all,
        # so a trial's "c3d" stage stayed invisible (not even "pending") if
        # Sync hadn't run yet. Seed it here; Sync/Overlay/REBA only top it up.
        if out_stem:
            outputs_root = app_config.get_outputs_dir()
            init_metadata(outputs_root, out_stem, vicon_fps=report.get("frame_rate_hz"))
            update_stage(outputs_root, out_stem, "c3d", "completed",
                         c3d_pig_detected=report.get("pig_detected"),
                         c3d_fallback_csv_used=report.get("fallback_csv_used"))

        manager.update_job(job_id, status="completed", progress=100,
                            step="Complete", result=report)
        logger.info(
            f"C3D job {job_id} done — {report['output_frames']} frames, "
            f"PiG={'YES' if report['pig_detected'] else 'NO'}, "
            f"fallback={'YES' if raw_nexus_csv else 'NO'}, "
            f"{report['processing_time_s']:.1f}s"
        )
    except Exception as exc:
        logger.error(f"C3D job {job_id} failed: {exc}", exc_info=True)
        manager.update_job(job_id, status="failed", step="Failed", error=str(exc))


@router.post("/start", response_model=JobStarted)
async def start_vicon(
    c3d_file:      UploadFile          = File(...),
    raw_nexus_csv: Optional[UploadFile] = File(default=None),
    task_name:    str                  = Form(""),
    max_gap:      int                  = Form(10),
):
    """
    Upload a Vicon C3D file and optionally a vicon_clean.py _clean.csv fallback.

    When a fallback CSV is provided, any joint frames that are NaN in the C3D
    output are filled frame-by-frame from the fallback CSV.
    Output format is identical to vicon_clean.py.
    """
    name = task_name.strip() or Path(c3d_file.filename or "trial").stem.replace(" ", "_")
    out_dir = task_dir(app_config.get_outputs_dir(), name)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_temp = app_config.get_temp_dir() / name
    task_temp.mkdir(parents=True, exist_ok=True)

    # Save C3D
    c3d_ext  = Path(c3d_file.filename or "trial.c3d").suffix or ".c3d"
    c3d_path = task_temp / f"trial{c3d_ext}"
    c3d_path.write_bytes(await c3d_file.read())

    # Save optional raw Nexus CSV — or, when not uploaded, try to auto-find the
    # same-stem sibling CSV next to the raw C3D (both come from the same Nexus
    # export, so "Task1_Lifting 01.c3d" + ".csv" always sit side by side).
    fb_path: Optional[Path] = None
    fb_original_filename: Optional[str] = None
    fb_auto_note: Optional[str] = None
    if raw_nexus_csv is not None:
        fb_path = task_temp / "nexus_raw.csv"
        fb_path.write_bytes(await raw_nexus_csv.read())
        fb_original_filename = raw_nexus_csv.filename
    elif c3d_file.filename:
        try:
            subject, _, _ = split_task_id(name)
        except ValueError:
            subject = None
        found = app_config.find_raw_sibling(subject, c3d_file.filename, "csv") if subject else None
        if found is not None:
            fb_path = task_temp / "nexus_raw.csv"
            fb_path.write_bytes(found.read_bytes())
            fb_original_filename = found.name
            fb_auto_note = f"Fallback CSV auto-matched: {found.name}"
        else:
            fb_auto_note = (
                "No fallback CSV uploaded and none auto-matched next to the raw C3D "
                f"— proceeding without one. Check {subject or '(subject)'}'s raw Vicon "
                "folder if a fallback was expected."
            )

    job_id = manager.create_job()
    manager.update_job(job_id, status="queued", step="Starting…")

    threading.Thread(
        target=_run_c3d_job,
        args=(job_id, c3d_path, out_dir, max_gap, fb_path, name, fb_original_filename, fb_auto_note),
        daemon=True,
    ).start()

    return JobStarted(job_id=job_id,
                      message=f"C3D processing started for '{name}'")


@router.get("/check-sibling")
async def check_sibling_csv(subject: str, filename: str):
    """
    Read-only pre-check: does a same-stem CSV exist next to this C3D's
    original filename in the raw Vicon folder (<subject>/)?

    Called live from the frontend as soon as a C3D is picked (before Process
    is clicked), so the Fallback CSV dropzone can show what will actually be
    auto-attached instead of looking empty. Never writes anything.
    """
    found = app_config.find_raw_sibling(subject, filename, "csv")
    return {"found": found is not None, "filename": found.name if found else None}


@router.get("/fetch-sibling")
async def fetch_sibling_csv(subject: str, filename: str):
    """
    Serve the raw bytes of the same-stem sibling CSV.

    Lets the frontend turn a confirmed check-sibling match into a real
    browser File object and drop it straight into the Fallback CSV dropzone
    — so it displays exactly like a manual upload (name, size, remove
    button) instead of a separate banner next to an empty-looking dropzone.
    """
    found = app_config.find_raw_sibling(subject, filename, "csv")
    if found is None:
        raise HTTPException(status_code=404, detail="No matching sibling CSV found")
    return FileResponse(str(found), filename=found.name, media_type="text/csv")


@router.get("/{job_id}/stream")
async def stream_vicon(job_id: str):
    """SSE progress stream."""
    if manager.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return StreamingResponse(
        manager.stream_job(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{job_id}/result")
async def get_vicon_result(job_id: str):
    """Return the completed C3D processing result."""
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == "failed":
        raise HTTPException(status_code=422, detail=job.error or "Processing failed")
    if job.status != "completed":
        raise HTTPException(status_code=202, detail="Job not yet complete")
    return job.result


@router.get("/{job_id}/download")
async def download_csv(job_id: str):
    """Download the generated clean CSV file."""
    job = manager.get_job(job_id)
    if job is None or job.result is None:
        raise HTTPException(status_code=404, detail="Result not found")
    p = Path(job.result["output_path"])
    if not p.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(str(p), filename=p.name,
                        media_type="text/csv")
