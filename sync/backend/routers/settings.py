"""
settings.py — /api/settings endpoints
=========================================
Lets the user choose where outputs/ and temp/ live on disk instead of being
fixed next to the app install — needed so this app works unmodified for
anyone it's shared with.

Since this backend runs locally on the researcher's own machine (not a
remote/multi-tenant service), the folder picker opens the OS's real native
dialog via tkinter — it appears on the same desktop session as the browser,
which a plain web page can never do on its own.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Form, HTTPException

from utils import app_config

router = APIRouter()


def _snapshot() -> dict:
    return {
        "data_dir": str(app_config.get_data_dir()),
        "outputs_dir": str(app_config.get_outputs_dir()),
        "temp_dir": str(app_config.get_temp_dir()),
    }


def _ask_directory_native() -> str | None:
    """Blocking — must be run off the asyncio event loop (see /browse-native)."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)   # bring the dialog in front of the browser
    path = filedialog.askdirectory(title="Choose sync data folder")
    root.destroy()
    return path or None   # askdirectory() returns "" on cancel


@router.get("")
def get_settings():
    return _snapshot()


@router.post("")
def update_settings(data_dir: str = Form(...)):
    from pathlib import Path
    try:
        app_config.set_data_dir(Path(data_dir))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return _snapshot()


@router.post("/browse-native")
async def browse_native():
    """Open the OS's native folder-picker dialog on this machine and return the
    chosen absolute path (or null if the user cancelled). Runs in a worker
    thread so the dialog being open doesn't block other requests."""
    loop = asyncio.get_event_loop()
    try:
        path = await loop.run_in_executor(None, _ask_directory_native)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not open the native folder dialog: {e}. "
                   f"Is a display/desktop session available to the backend process?",
        )
    return {"path": path}
