"""Persisted application settings: where calibration results are stored.

The folder can live anywhere on disk. The pointer itself is kept in
``calibration/config.json`` next to the backend, so it survives restarts.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_APP_ROOT = Path(__file__).parent.parent.parent
_CONFIG_PATH = _APP_ROOT / "config.json"
_DEFAULT_EXPORTS = _APP_ROOT / "exports"


def _load() -> dict:
    if not _CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Ignoring unreadable %s: %s", _CONFIG_PATH, e)
        return {}


def get_exports_dir() -> Path:
    """Folder that receives intrinsic/extrinsic results.

    Uses the configured folder when it is available, otherwise falls back to
    ``calibration/exports`` so the app (and the Settings page) keeps working
    when, for example, an external drive is unplugged.
    """
    raw = _load().get("exports_dir")
    if raw:
        configured = Path(raw)
        if configured.is_dir():
            return configured
        logger.warning("Configured results folder %s is unavailable; using %s", configured, _DEFAULT_EXPORTS)
    _DEFAULT_EXPORTS.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_EXPORTS


def set_exports_dir(path: Path) -> None:
    """Validate that ``path`` is an absolute, writable directory, then persist it."""
    if not path.is_absolute():
        raise ValueError(f"Folder path must be absolute: {path}")
    path = path.resolve()
    if not path.is_dir():
        raise ValueError(f"Not a directory: {path}")
    probe = path / ".write_test"
    try:
        probe.write_text("ok")
        probe.unlink()
    except OSError as e:
        raise ValueError(f"Folder is not writable: {path} ({e})")

    cfg = _load()
    cfg["exports_dir"] = str(path)
    tmp = _CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    os.replace(tmp, _CONFIG_PATH)
