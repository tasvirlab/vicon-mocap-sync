"""
job_manager.py
==============
Thread-safe in-memory job store with SSE-compatible progress streaming.

Each job has: status, progress (0-100), step label, result dict, error str.
Progress updates are written by worker threads; SSE endpoints poll the store.
"""

from __future__ import annotations
import asyncio
import uuid
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, AsyncIterator
import json

logger = logging.getLogger(__name__)

# Retain completed/failed jobs for this many seconds before cleanup
_JOB_TTL_SECONDS = 3600


@dataclass
class Job:
    job_id: str
    status: str = "queued"        # queued | running | completed | failed
    progress: int = 0             # 0 – 100
    step: str = "Waiting…"
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def update(self, *, status: str = None, progress: int = None,
               step: str = None, message: str = "", result=None, error=None):
        if status   is not None: self.status   = status
        if progress is not None: self.progress = progress
        if step     is not None: self.step     = step
        self.message   = message
        if result  is not None: self.result  = result
        if error   is not None: self.error   = error
        self.updated_at = time.time()

    def to_event(self) -> str:
        payload = {
            "job_id":   self.job_id,
            "status":   self.status,
            "progress": self.progress,
            "step":     self.step,
            "message":  self.message,
        }
        if self.error:
            payload["error"] = self.error
        return json.dumps(payload)


class JobManager:
    """Singleton store for all calibration jobs."""

    def __init__(self):
        self._jobs: Dict[str, Job] = {}

    # ------------------------------------------------------------------
    def create_job(self) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = Job(job_id=job_id)
        logger.debug(f"Job created: {job_id}")
        return job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs):
        job = self._jobs.get(job_id)
        if job:
            job.update(**kwargs)

    def cleanup_old_jobs(self):
        now = time.time()
        stale = [
            jid for jid, job in self._jobs.items()
            if job.status in ("completed", "failed")
            and (now - job.updated_at) > _JOB_TTL_SECONDS
        ]
        for jid in stale:
            del self._jobs[jid]

    # ------------------------------------------------------------------
    async def stream_job(self, job_id: str) -> AsyncIterator[str]:
        """
        Async generator that yields SSE-formatted events until the job
        is completed or failed.  Polls every 250 ms.
        """
        while True:
            job = self._jobs.get(job_id)
            if job is None:
                yield f"data: {json.dumps({'error': 'job not found'})}\n\n"
                return

            yield f"data: {job.to_event()}\n\n"

            if job.status in ("completed", "failed"):
                return

            await asyncio.sleep(0.25)


# Module-level singleton
manager = JobManager()
