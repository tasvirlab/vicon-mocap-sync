"""
projects.py — /api/projects endpoints
========================================
List, inspect, and delete saved task sessions from the outputs/ directory.
Each subdirectory of outputs/ corresponds to one task (task_name).
"""

from __future__ import annotations
import json
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from models.schemas import ProjectEntry, ProjectList
from utils import app_config
from utils.task_paths import (discover_task_dirs, split_task_id,
                              task_dir as resolve_task_dir, Artifacts,
                              CANONICAL_TASKS)

logger = logging.getLogger(__name__)
router = APIRouter()


def _load_project(task_id: str, folder: Path) -> ProjectEntry | None:
    """Build a ProjectEntry from a trial folder.

    Args:
        task_id: composite id, e.g. "SUB01_lifting_T01" — NOT folder.name,
            which may now just be "T01" if folder is nested under outputs/SUB01/lifting/.
        folder:  the resolved trial folder (any nesting depth).
    """
    sync_json = folder / Artifacts.SYNC_JSON
    if not sync_json.exists():
        return None

    try:
        with open(sync_json, encoding="utf-8") as f:
            d = json.load(f)
        stat = sync_json.stat()

        subject_id = task = trial = None
        try:
            subject_id, task, trial = split_task_id(task_id)
        except ValueError:
            pass   # legacy/unassigned id — leave subject_id/task/trial as None

        return ProjectEntry(
            id=task_id,
            name=task_id,
            created_at=datetime.fromtimestamp(stat.st_ctime).isoformat(),
            sync_method=d.get("sync_method"),
            sync_offset_s=float(d.get("sync_offset_s", 0.0)),
            video_rotation=d.get("video_rotation"),
            video_fps=d.get("video_fps"),
            has_overlay=(folder / Artifacts.QA_DIR / Artifacts.QA_OVERLAY).exists(),
            has_synced_csv=(folder / Artifacts.SYNCED_60HZ).exists(),
            has_diagnostic_png=(folder / Artifacts.QA_DIR / Artifacts.QA_SYNC_DIAG).exists(),
            output_dir=str(folder),
            subject_id=subject_id,
            task=task,
            trial=trial,
        )
    except Exception as e:
        logger.debug(f"Could not parse project {task_id}: {e}")
        return None


@router.get("", response_model=ProjectList)
def list_projects():
    """List all task sessions from the outputs directory, at any nesting depth."""
    app_config.get_outputs_dir().mkdir(exist_ok=True)
    entries = []
    for task_id, folder in discover_task_dirs(app_config.get_outputs_dir()):
        entry = _load_project(task_id, folder)
        if entry:
            entries.append(entry)
    entries.sort(key=lambda e: e.created_at, reverse=True)
    return ProjectList(projects=entries, total=len(entries))


@router.get("/task-names")
def list_task_names():
    """Distinct task-name segments already used, for autocomplete in the
    upload forms (Vicon/Sync pages) — e.g. ["lifting", "overhead_work", ...]."""
    names = set()
    for task_id, _ in discover_task_dirs(app_config.get_outputs_dir()):
        try:
            _, task, _ = split_task_id(task_id)
            names.add(task)
        except ValueError:
            continue
    return {"task_names": sorted(names)}


@router.get("/canonical-tasks")
def list_canonical_tasks():
    """The fixed 8-task whitelist, in protocol order, for the Task DROPDOWN on
    the trial-creation pages (C3D Convert / Sync). Single source of truth =
    task_paths.CANONICAL_TASKS, so the UI can only ever submit a canonical task
    name — no free-text typo can create a mislocated flat folder.

    Returns: [{"value": "lifting", "label": "1. Lifting", "folder": "01_lifting"}, ...]
    """
    tasks = []
    for canonical, folder in CANONICAL_TASKS.items():
        num, _, rest = folder.partition("_")          # "01_lifting" -> "01","lifting"
        label = f"{int(num)}. {rest.replace('_', ' ').title()}"
        tasks.append({"value": canonical, "label": label, "folder": folder})
    return {"tasks": tasks}


@router.delete("/{task_name}")
def delete_project(task_name: str):
    """Delete one trial: its folder, and the task folder if it was the last trial.

    One trial = one folder, so this is a single rmtree. It used to have to glob
    for `{task_name}_*` because every trial of a task shared a folder — that
    prefix dance was both fragile and the reason unprefixed artifacts (reba/,
    diagnostic_sync.png) leaked between trials on delete as well as on write.
    """
    folder = resolve_task_dir(app_config.get_outputs_dir(), task_name)
    if not folder.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    # Guard: only ever remove something inside outputs/. resolve_task_dir()
    # returns outputs/<task_id> verbatim for unrecognised ids, so a hostile
    # task_name must not be able to walk out of the tree before rmtree runs.
    outputs_root = app_config.get_outputs_dir().resolve()
    try:
        folder.resolve().relative_to(outputs_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task name")
    if folder.resolve() == outputs_root:
        raise HTTPException(status_code=400, detail="Refusing to delete the outputs root")

    shutil.rmtree(folder, ignore_errors=True)

    # Drop the task folder (e.g. SUB01/01_lifting/) if that was its last trial,
    # then the subject folder if that was their last task. Never touch outputs/.
    for parent in (folder.parent, folder.parent.parent):
        try:
            if parent.resolve() == outputs_root:
                break
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
            else:
                break
        except OSError:
            break

    # Clean up temp uploads for this task
    task_temp = app_config.get_temp_dir() / task_name
    if task_temp.exists():
        shutil.rmtree(task_temp, ignore_errors=True)

    return {"message": f"Task '{task_name}' deleted"}


@router.post("/{task_name}/open")
def open_project_folder(task_name: str):
    """Open the task output folder in Windows Explorer."""
    folder = resolve_task_dir(app_config.get_outputs_dir(), task_name)
    if not folder.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    subprocess.Popen(["explorer", str(folder)], shell=False)
    return {"message": "Opened"}


@router.get("/{task_name}/file/{filename:path}")
def get_project_file(task_name: str, filename: str):
    """Download a specific file from a trial's output folder.

    `filename` is a `:path` so it can address the subfolders the layout now uses
    (e.g. "qa/overlay.mp4", "reba/scores.csv"). That means it is attacker-shaped
    input — "../../../etc/passwd" is a valid path too — so the resolved target is
    checked to be inside the trial folder before anything is served.
    """
    folder = resolve_task_dir(app_config.get_outputs_dir(), task_name)
    if not folder.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    file_path = (folder / filename).resolve()
    try:
        file_path.relative_to(folder.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file path")

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")
    return FileResponse(str(file_path), filename=file_path.name)
