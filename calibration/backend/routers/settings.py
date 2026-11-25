"""/api/settings: choose where calibration results are saved."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Form, HTTPException

from utils import app_config

router = APIRouter()


def _snapshot() -> dict:
    return {"exports_dir": str(app_config.get_exports_dir())}


@router.get("")
def get_settings():
    return _snapshot()


@router.post("")
def update_settings(exports_dir: str = Form(...)):
    try:
        app_config.set_exports_dir(Path(exports_dir))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return _snapshot()
