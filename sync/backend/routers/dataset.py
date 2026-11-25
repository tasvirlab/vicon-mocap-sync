"""
dataset.py — /api/dataset endpoints
======================================
Run the windowed pose-lifting dataset export (dataset_export.py) over a chosen set
of trials, producing construction_vicon_train.npz / _test.npz.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from utils import app_config
from utils.job_manager import manager
from utils.task_paths import discover_task_dirs, Artifacts

logger = logging.getLogger(__name__)
router = APIRouter()


def _exports_dir() -> Path:
    return app_config.get_outputs_dir() / "_dataset_exports"


# ── Background worker ────────────────────────────────────────────────────────────

def _run_dataset_job(
    job_id: str,
    task_names: list[str] | None,
    window_size: int,
    step_size: int,
    test_subjects: list[str] | None,
    test_fraction: float,
    val_subjects: list[str] | None = None,
    val_fraction: float = 0.2,
):
    t0 = time.time()
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from core import dataset_export as de

        manager.update_job(job_id, status="running", progress=1, step="Starting…")

        def _cb(pct: int, step: str):
            manager.update_job(job_id, progress=pct, step=step)

        result = de.run_export(
            app_config.get_outputs_dir(),
            task_names=task_names,
            window_size=window_size,
            step_size=step_size,
            test_subjects=test_subjects,
            val_subjects=val_subjects,
            test_fraction=test_fraction,
            val_fraction=val_fraction,
            progress_cb=_cb,
        )
        result["job_id"] = job_id
        result["processing_time_s"] = round(time.time() - t0, 2)

        manager.update_job(job_id, status="completed", progress=100,
                            step="Export complete", result=result)
        logger.info(
            f"dataset export {job_id}: {result['n_train_windows']} train / "
            f"{result['n_val_windows']} val / {result['n_test_windows']} test "
            f"windows -> {result['export_dir']}"
        )

    except Exception as exc:
        logger.error(f"dataset export job {job_id} failed: {exc}", exc_info=True)
        manager.update_job(job_id, status="failed", step="Failed", error=str(exc))


# ── Endpoints ────────────────────────────────────────────────────────────────────

@router.get("/tasks")
def list_dataset_tasks():
    """List trial folders that have a <task>_pose2d.csv (eligible for export),
    with subject ID and synced.csv presence."""
    import re
    subid_re = re.compile(r"^(SUB\d+)_")

    tasks = []
    for task_id, trial_dir in sorted(discover_task_dirs(app_config.get_outputs_dir())):
        pose2d = trial_dir / Artifacts.POSE2D_RTMPOSE
        synced = trial_dir / Artifacts.SYNCED_60HZ
        if not pose2d.exists():
            continue
        m = subid_re.match(task_id)
        tasks.append({
            "task_name": task_id,
            "subject_id": m.group(1) if m else None,
            "has_synced_csv": synced.exists(),
        })
    return {"tasks": tasks}


@router.get("/exports")
def list_exports():
    """List previously generated dataset exports (newest first)."""
    if not _exports_dir().exists():
        return {"exports": []}

    exports = []
    for folder in sorted(_exports_dir().iterdir(), reverse=True):
        if not folder.is_dir():
            continue
        train_npz = folder / "construction_vicon_train.npz"
        test_npz = folder / "construction_vicon_test.npz"
        if train_npz.exists() and test_npz.exists():
            exports.append({
                "timestamp": folder.name,
                "train_npz": str(train_npz),
                "test_npz": str(test_npz),
                "has_visualize": (folder / "visualize_check.png").exists(),
            })
    return {"exports": exports}


@router.post("/start")
async def start_dataset_export(
    task_names: str = Form(default="null"),    # JSON-encoded list[str] or "null" for all
    window_size: int = Form(default=243),
    step_size: int = Form(default=81),
    test_subjects: str = Form(default="null"),  # JSON-encoded list[str] or "null" for auto-split
    test_fraction: float = Form(default=0.2),
    val_subjects: str = Form(default="null"),   # JSON-encoded list[str] or "null" for auto-split
    val_fraction: float = Form(default=0.2),
):
    """Start a dataset export job (subject-wise 3-way train/val/test split)."""
    try:
        names = json.loads(task_names)
        subjects = json.loads(test_subjects)
        val_subj = json.loads(val_subjects)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="task_names/test_subjects/val_subjects must be JSON-encoded")

    job_id = manager.create_job()
    manager.update_job(job_id, status="queued", step="Starting…")

    threading.Thread(
        target=_run_dataset_job,
        args=(job_id, names, window_size, step_size, subjects, test_fraction,
              val_subj, val_fraction),
        daemon=True,
    ).start()

    return {"job_id": job_id, "message": "Dataset export started"}


@router.get("/{job_id}/stream")
async def stream_dataset(job_id: str):
    if manager.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return StreamingResponse(
        manager.stream_job(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{job_id}/result")
async def get_dataset_result(job_id: str):
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == "failed":
        raise HTTPException(status_code=422, detail=job.error or "Dataset export failed")
    if job.status != "completed":
        raise HTTPException(status_code=202, detail="Job not yet complete")
    return job.result


@router.get("/download/{timestamp}/{filename}")
async def download_export_file(timestamp: str, filename: str):
    """Download a file from a specific export folder (construction_vicon_train.npz, etc.)."""
    file_path = _exports_dir() / timestamp / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path), filename=filename)
