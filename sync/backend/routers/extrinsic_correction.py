"""
extrinsic_correction.py — /api/extrinsic-correction endpoints
================================================================
Persist a manual spatial correction (see core/extrinsic_correction.py) into
a task's extrinsics.json. Kept as its own small router rather than folding
into routers/overlay.py (already over this project's file-size ceiling).
"""

from __future__ import annotations
import logging

from fastapi import APIRouter, Form, HTTPException

from utils import app_config
from core.extrinsic_correction import save_correction

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/{task_name}/save")
async def save_spatial_correction(
    task_name: str,
    dx:     float = Form(0.0),
    dy:     float = Form(0.0),
    dz:     float = Form(0.0),
    droll:  float = Form(0.0),
    dpitch: float = Form(0.0),
    dyaw:   float = Form(0.0),
):
    """
    Permanently compose the given spatial correction into this task's
    extrinsics.json. Every task/trial that shares that same file (same
    subject/session) picks up the correction automatically on its next
    preview or full overlay render — nothing else needs to be re-run.
    """
    # Same resolution order as routers/overlay.py's /preview endpoint.
    task_temp = app_config.get_temp_dir() / task_name
    extr_path = task_temp / "extrinsics.json"
    if (task_temp / "extrinsics_overlay.json").exists():
        extr_path = task_temp / "extrinsics_overlay.json"

    if not extr_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Extrinsics not found. Re-run Sync or upload via Overlay → Start."
        )

    delta = {"dx": dx, "dy": dy, "dz": dz,
              "droll": droll, "dpitch": dpitch, "dyaw": dyaw}
    try:
        updated = save_correction(extr_path, delta)
    except Exception as exc:
        logger.error(f"Failed to save spatial correction for '{task_name}': {exc}",
                     exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "saved": True,
        "rvec": updated["rvec"],
        "tvec_m": updated["tvec_m"],
    }
