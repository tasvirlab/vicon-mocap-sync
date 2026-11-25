"""
main.py — calibration API server
=======================================
FastAPI backend exposing intrinsic and extrinsic calibration endpoints.
Starts on http://localhost:8765

Usage:
    python main.py
    uvicorn main:app --host 127.0.0.1 --port 8765 --reload
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

# Ensure backend/ is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from routers import intrinsic, extrinsic, extrinsic_body, projects, settings
from utils import app_config
from utils.job_manager import manager

LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT,
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("viconcalib")


class _AccessLogThrottle(logging.Filter):
    """Suppress high-frequency uvicorn access-log lines; allow one per 5 minutes.

    Patterns throttled:
      /api/health  — Electron frontend heartbeat poll
      /frame/      — Video frame scrubbing requests
      /stream      — Preview stream requests
    """

    _INTERVAL_S: float = 300.0   # 5 minutes
    _PATTERNS: tuple = ("/api/health", "/frame/", "/stream")

    def __init__(self) -> None:
        super().__init__()
        self._last: dict = {}   # pattern → last monotonic timestamp

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for pattern in self._PATTERNS:
            if pattern in msg:
                now = time.monotonic()
                if now - self._last.get(pattern, 0.0) >= self._INTERVAL_S:
                    self._last[pattern] = now
                    return True   # first hit in this window — let it through
                return False      # within cooldown — suppress
        return True               # unrelated line — pass through


# Install on uvicorn's access logger so it takes effect regardless of reload
logging.getLogger("uvicorn.access").addFilter(_AccessLogThrottle())


# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Report the ACTUAL bind host, not a hardcoded one — a stale "127.0.0.1"
    # here next to a real 0.0.0.0 bind is exactly the kind of mismatch that
    # hides the "server says it's up but nothing can connect" symptom.
    _host = os.environ.get("HOST", "127.0.0.1")
    logger.info(f"Calibration backend ready on http://{_host}:8765")
    # Ensure output dirs exist
    app_config.get_exports_dir()
    for d in ["temp", "logs"]:
        (Path(__file__).parent.parent / d).mkdir(exist_ok=True)
    yield
    manager.cleanup_old_jobs()
    logger.info("Backend stopped")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="ViconMoCap Sync Studio: calibration API",
    version="1.0.0",
    description="Automated Intrinsic & Extrinsic Calibration for Smartphone + Vicon MoCap",
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

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(intrinsic.router,      prefix="/api/intrinsic",      tags=["Intrinsic"])
app.include_router(extrinsic.router,      prefix="/api/extrinsic",      tags=["Extrinsic"])
app.include_router(extrinsic_body.router, prefix="/api/extrinsic-body", tags=["ExtrinsicBody"])
app.include_router(projects.router,       prefix="/api/projects",       tags=["Projects"])
app.include_router(settings.router,       prefix="/api/settings",       tags=["Settings"])


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["System"])
def health():
    return {"status": "ok", "version": "1.0.0", "app": "ViconMoCap Sync Studio (calibration)"}


@app.get("/api/jobs", tags=["System"])
def list_jobs():
    return {
        jid: {"status": j.status, "progress": j.progress, "step": j.step}
        for jid, j in manager._jobs.items()
    }



# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    # Bind to loopback only: the API must not be exposed to the local network.
    uvicorn.run(
        "main:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=8765,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
