"""
main.py — ViconSync Studio API Server (v2)
==========================================
FastAPI backend for the synchronisation pipeline.
Starts on http://localhost:8766

Pipeline:
  sync    → wall-clock timestamps (XCP + ffprobe) + optional cross-correlation
  overlay → project Vicon skeleton onto video via calibrated camera
  projects → manage saved task sessions in outputs/

Usage:
    python main.py
"""

import logging
import os
import shutil
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles

# Load sync/.env (if present) before anything reads os.environ — resolved
# relative to this file so it works regardless of the caller's cwd. No .env is
# required: every setting below has a working default (see .env.example).
load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))

from routers import (sync, overlay, projects, vicon, reba, pose2d, dataset, settings,
                     extrinsic_correction, analyse)
from utils import app_config
from utils.job_manager import manager

LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("viconsync")


class _AccessLogThrottle(logging.Filter):
    """Suppress high-frequency uvicorn access-log lines (health + stream)."""
    _INTERVAL_S: float = 300.0
    _PATTERNS:   tuple = ("/api/health", "/stream")

    def __init__(self) -> None:
        super().__init__()
        self._last: dict = {}

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for pattern in self._PATTERNS:
            if pattern in msg:
                now = time.monotonic()
                if now - self._last.get(pattern, 0.0) >= self._INTERVAL_S:
                    self._last[pattern] = now
                    return True
                return False
        return True


logging.getLogger("uvicorn.access").addFilter(_AccessLogThrottle())


def _prune_old_temp(temp_dir: Path, max_age_days: int = 30) -> None:
    """Remove temp upload dirs that are older than max_age_days."""
    if not temp_dir.exists():
        return
    cutoff = time.time() - max_age_days * 86400
    for d in temp_dir.iterdir():
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)
            logger.info("Pruned stale temp: %s", d.name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ViconSync Studio backend ready on http://127.0.0.1:8766")
    logger.info("Data directory: %s", app_config.get_data_dir())
    (Path(__file__).parent.parent / "logs").mkdir(exist_ok=True)
    app_config.get_outputs_dir()
    _prune_old_temp(app_config.get_temp_dir())
    yield
    manager.cleanup_old_jobs()
    logger.info("Backend stopped")


app = FastAPI(
    title="ViconSync Studio",
    version="2.0.0",
    description=(
        "Vicon-video temporal synchronisation and skeleton overlay.  "
        "Synchronisation via wall-clock timestamps (XCP + ffprobe) "
        "with optional cross-correlation refinement."
    ),
    lifespan=lifespan,
)

# Local-only API: accept browser requests only from the app's own pages (Vite dev server / web mode).
# Extra origins, for example a packaged build, can be added with VICONSYNC_ALLOWED_ORIGINS (comma separated).
ALLOWED_ORIGINS = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:5174", "http://127.0.0.1:5174",
    *[o.strip() for o in os.environ.get("VICONSYNC_ALLOWED_ORIGINS", "").split(",") if o.strip()],
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Only loopback host names (blocks DNS rebinding).
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])


@app.middleware("http")
async def reject_foreign_origin(request: Request, call_next):
    """CORS only hides responses; a cross-site form POST would still run. Refuse any browser request from another origin."""
    origin = request.headers.get("origin")
    if origin is not None and origin not in ALLOWED_ORIGINS:
        return JSONResponse({"detail": "Origin not allowed"}, status_code=403)
    return await call_next(request)

app.include_router(vicon.router,    prefix="/api/vicon",    tags=["Vicon"])
app.include_router(sync.router,     prefix="/api/sync",     tags=["Sync"])
app.include_router(overlay.router,  prefix="/api/overlay",  tags=["Overlay"])
app.include_router(projects.router, prefix="/api/projects", tags=["Projects"])
app.include_router(reba.router,     prefix="/api/reba",     tags=["REBA"])
app.include_router(pose2d.router,   prefix="/api/pose2d",   tags=["Pose2D"])
app.include_router(dataset.router,  prefix="/api/dataset",  tags=["Dataset"])
app.include_router(settings.router, prefix="/api/settings", tags=["Settings"])
app.include_router(analyse.router,  prefix="/api/analyse",  tags=["Analyse"])
app.include_router(extrinsic_correction.router, prefix="/api/extrinsic-correction",
                   tags=["ExtrinsicCorrection"])


@app.get("/api/health", tags=["System"])
def health():
    return {"status": "ok", "version": "2.0.0", "app": "ViconSync Studio"}


@app.get("/api/jobs", tags=["System"])
def list_jobs():
    return {
        jid: {"status": j.status, "progress": j.progress, "step": j.step}
        for jid, j in manager.list_jobs()
    }


app.mount("/outputs", StaticFiles(directory=str(app_config.get_outputs_dir())), name="outputs")


def main():
    uvicorn.run(
        "main:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8766")),
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
