"""
app_config.py — persisted app-level configuration (data directory location).

Lets outputs/ and temp/ live anywhere on disk instead of being fixed next to
the app install — required so this app works unmodified when shared with
another researcher on a different machine/drive layout.

config.json itself stays next to the app (the one path that must remain
fixed as a pointer); the folders it points to can be anywhere.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_APP_ROOT = Path(__file__).parent.parent.parent   # sync/
_CONFIG_PATH = _APP_ROOT / "config.json"


def _load() -> dict:
    if _CONFIG_PATH.exists():
        try:
            return json.loads(_CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save(cfg: dict) -> None:
    _CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def get_data_dir() -> Path:
    """Base folder containing outputs/ and temp/. Defaults to the app's own folder."""
    raw = _load().get("data_dir")
    return Path(raw) if raw else _APP_ROOT


def set_data_dir(path: Path) -> None:
    """Validate `path` is a writable directory, then persist it as the new data dir.

    Takes effect immediately for new work (jobs started after this call) — existing
    outputs/temp files in the previous location are left untouched, not migrated.
    """
    if not path.is_dir():
        raise ValueError(f"Not a directory: {path}")
    probe = path / ".write_test"
    try:
        probe.write_text("ok")
        probe.unlink()
    except OSError as e:
        raise ValueError(f"Folder is not writable: {path} ({e})")

    cfg = _load()
    cfg["data_dir"] = str(path)
    _save(cfg)

    get_outputs_dir()  # ensure subfolders exist immediately
    get_temp_dir()


def get_outputs_dir() -> Path:
    d = get_data_dir() / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_temp_dir() -> Path:
    d = get_data_dir() / "temp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_raw_vicon_dir() -> Path | None:
    """Optional read-only folder with raw Vicon exports (one subfolder per subject), used to auto-match a trial's XCP file.

    Set the environment variable VICONSYNC_RAW_VICON_DIR to enable it. None when unset or missing; callers must treat
    that as "auto-lookup unavailable", not an error.
    """
    raw = os.environ.get("VICONSYNC_RAW_VICON_DIR", "").strip()
    d = Path(raw) if raw else None
    return d if d is not None and d.is_dir() else None


def find_raw_sibling(subject: str, filename: str, ext: str) -> Path | None:
    """Case-insensitive same-stem sidecar lookup in <raw Vicon folder>/<subject>/.

    The raw Nexus export keeps a C3D, its cleaned CSV, and its XCP together as
    same-stem, different-extension files (e.g. "Trial 01.c3d" +
    ".csv" + ".xcp"). Given the ORIGINAL filename of one (as uploaded — not a
    renamed/processed copy), finds the sibling with the requested extension.
    Case-insensitive because raw filenames are inconsistently cased across
    subjects (e.g. "Trial" vs "trial").

    Returns None (never raises) when the raw mount isn't present, the subject
    folder doesn't exist, or nothing matches — callers surface that as a
    warning to the user rather than silently guessing or hard-failing.
    """
    raw_dir = get_raw_vicon_dir()
    if raw_dir is None:
        return None
    sub_dir = raw_dir / subject
    if not sub_dir.is_dir():
        return None
    stem = Path(filename).stem.strip().lower()
    ext = ext.lower().lstrip(".")
    for p in sub_dir.glob(f"*.{ext}"):
        if p.stem.strip().lower() == stem:
            return p
    return None
