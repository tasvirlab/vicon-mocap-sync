"""Projects API — browse saved calibration outputs."""

from __future__ import annotations
import json
import logging
import os
import platform
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from models.schemas import ProjectEntry, ProjectList
from utils import app_config

logger = logging.getLogger(__name__)
router = APIRouter()



def _run_kind(folder: Path) -> tuple[str, Path] | None:
    """Detect a run folder's type from its files. None if `folder` isn't one
    (e.g. it's a subject folder containing run folders, not output files)."""
    if (folder / "intrinsic.json").exists():
        return "intrinsic", folder / "intrinsic.json"
    if (folder / "intrinsic_params.json").exists():
        return "intrinsic", folder / "intrinsic_params.json"
    if (folder / "extrinsics.json").exists():         # body-marker extrinsic
        return "extrinsic", folder / "extrinsics.json"
    if (folder / "extrinsic_params.json").exists():   # board extrinsic
        return "extrinsic", folder / "extrinsic_params.json"
    return None


def _build_entry(folder: Path, project_id: str) -> ProjectEntry | None:
    """Build a ProjectEntry for a single run folder, or None if unparseable."""
    detected = _run_kind(folder)
    if detected is None:
        return None
    kind, params_path = detected

    try:
        with open(params_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        logger.warning(f"Skipping {project_id}: failed to parse {params_path.name}")
        return None

    try:
        if kind == "intrinsic":
            # Current format: rms_reprojection_px at top level
            if "rms_reprojection_px" in data:
                rms     = float(data["rms_reprojection_px"])
                quality = "EXCELLENT" if rms < 0.5 else "GOOD" if rms < 1.0 else "POOR"
                ts      = data.get("date", folder.name)
            else:
                # Older format: calibration_info sub-dict
                info    = data.get("calibration_info", {})
                rms     = float(info.get("rms_px", 0))
                quality = info.get("quality", "Unknown")
                ts      = info.get("timestamp", folder.name)
        else:
            # extrinsics.json (body mode): rms_reprojection_px at top level
            # extrinsic_params.json (board mode): reprojection.rms_px or rms_px
            if "rms_reprojection_px" in data:
                rms     = float(data["rms_reprojection_px"])
                quality = "EXCELLENT" if rms < 1.0 else "GOOD" if rms < 3.0 else "ACCEPTABLE" if rms < 5.0 else "POOR"
                ts      = data.get("date", folder.name)
            else:
                rep     = data.get("reprojection", {})
                rms     = float(rep.get("rms_px", data.get("rms_px", 0)))
                quality = data.get("quality", "Unknown")
                ts      = data.get("timestamp", data.get("date", folder.name))

        # Thumbnail — check root files first, then reprojection_check subfolder
        thumbnail = None
        for candidate in ["reprojection_error_plot.png", "undistorted_preview.jpg",
                           "overlay_frame.jpg", "reprojection_plot.png"]:
            if (folder / candidate).exists():
                thumbnail = candidate
                break
        if thumbnail is None:
            # Body extrinsic: use first check image from reprojection_check/
            rp_dir = folder / "reprojection_check"
            if rp_dir.is_dir():
                imgs = sorted(rp_dir.glob("check_*.jpg"))
                if imgs:
                    thumbnail = f"reprojection_check/{imgs[0].name}"

        return ProjectEntry(
            id=project_id,
            name=project_id,
            type=kind,
            created_at=ts,
            rms_px=rms,
            quality=quality,
            output_dir=str(folder),
            thumbnail=thumbnail,
        )
    except Exception:
        logger.warning(f"Skipping {project_id}: unexpected error", exc_info=True)
        return None


def _scan_projects() -> list[ProjectEntry]:
    """Scan exports/ for run folders, one level deep.

    Two layouts coexist: a run folder directly under exports/ (the original,
    flat layout — id is just the folder name), or a run folder nested under a
    subject folder like exports/SUB01/intrinsic_.../ (id is "SUB01/intrinsic_
    ..."), used when a subject ID was given at calibration time so per-subject
    runs land in one place instead of all mixed at the exports root.
    """
    projects = []
    exports_dir = app_config.get_exports_dir()
    if not exports_dir.exists():
        return projects

    for folder in sorted(exports_dir.iterdir(), reverse=True):
        if not folder.is_dir():
            continue

        entry = _build_entry(folder, folder.name)
        if entry is not None:
            projects.append(entry)
            continue

        # Not a run folder itself — treat as a subject folder and scan one
        # level deeper.
        for sub in sorted(folder.iterdir(), reverse=True):
            if not sub.is_dir():
                continue
            sub_entry = _build_entry(sub, f"{folder.name}/{sub.name}")
            if sub_entry is not None:
                projects.append(sub_entry)

    return projects


def _resolve_project_dir(project_id: str) -> Path:
    """Resolve a project id (possibly "SUBJECT/run_folder") to its exports/
    subfolder, rejecting anything that would escape exports/ (e.g. "../..").
    project_id now accepts slashes (see the :path routes below) to address
    subject-nested runs, so this containment check is load-bearing, not
    decorative — without it a crafted project_id could read/open/delete
    arbitrary paths on disk.
    """
    exports_dir = app_config.get_exports_dir()
    folder = (exports_dir / project_id).resolve()
    exports_root = exports_dir.resolve()
    if exports_root not in folder.parents and folder != exports_root:
        raise HTTPException(status_code=400, detail="Invalid project id")
    if not folder.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    return folder


@router.get("", response_model=ProjectList)
async def list_projects():
    projects = _scan_projects()
    return ProjectList(projects=projects, total=len(projects))


@router.get("/{project_id:path}/thumbnail")
async def get_thumbnail(project_id: str):
    folder = _resolve_project_dir(project_id)
    # Root-level candidates
    for name in ["reprojection_error_plot.png", "undistorted_preview.jpg",
                 "overlay_frame.jpg", "reprojection_plot.png"]:
        p = folder / name
        if p.exists():
            return FileResponse(str(p))
    # Body extrinsic: first image in reprojection_check/
    rp_dir = folder / "reprojection_check"
    if rp_dir.is_dir():
        imgs = sorted(rp_dir.glob("check_*.jpg"))
        if imgs:
            return FileResponse(str(imgs[0]))
    raise HTTPException(status_code=404, detail="No thumbnail available")


@router.get("/{project_id:path}/file/{filename}")
async def download_project_file(project_id: str, filename: str):
    folder = _resolve_project_dir(project_id)
    # `filename` is client-supplied; on Windows a backslash is a path separator, so
    # "..\..\x" is a single URL segment that still walks out of the project folder.
    p = (folder / filename).resolve()
    if folder not in p.parents:
        raise HTTPException(status_code=400, detail="Invalid file name")
    if not p.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(p), filename=p.name)


# Bare catch-all — must stay after the more specific /thumbnail and /file/...
# GET routes above, since {project_id:path} is greedy and would otherwise
# swallow those suffixes too (Starlette matches routes in declaration order).
@router.get("/{project_id:path}")
async def get_project(project_id: str):
    folder = _resolve_project_dir(project_id)

    for name in ("intrinsic.json", "intrinsic_params.json",
                  "extrinsics.json", "extrinsic_params.json"):
        p = folder / name
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)

    raise HTTPException(status_code=404, detail="No result file found")


@router.post("/{project_id:path}/open")
async def open_project_folder(project_id: str):
    """Open the project output directory in the OS file manager."""
    folder = _resolve_project_dir(project_id)
    path = str(folder.resolve())
    try:
        if platform.system() == "Windows":
            # os.startfile opens the folder itself (shows contents).
            # explorer /select,"path" only selects the folder in its parent — wrong for dirs.
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not open folder: {e}")
    return {"opened": path}


@router.delete("/{project_id:path}")
async def delete_project(project_id: str):
    import shutil
    folder = _resolve_project_dir(project_id)
    shutil.rmtree(str(folder))
    return {"deleted": project_id}
