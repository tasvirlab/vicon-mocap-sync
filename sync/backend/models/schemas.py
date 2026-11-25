"""Pydantic schemas for ViconSync Studio API (v2)."""

from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel


# ── Job lifecycle ──────────────────────────────────────────────────────────────

class JobStatus(BaseModel):
    job_id:   str
    status:   str    # queued | running | completed | failed
    progress: int
    step:     str
    message:  str = ""


class JobStarted(BaseModel):
    job_id:  str
    message: str = "Job started"


# ── Synchronisation ────────────────────────────────────────────────────────────

class SyncConfig(BaseModel):
    task_name:                   str
    timezone_hours:              float = 5.0
    video_time_correction_hours: float = 0.0
    vicon_fps:                   float = 100.0
    manual_frame_offset:         int   = 0
    auto_refine:                 bool  = False
    search_range_s:              float = 2.0
    video_rotation:              str   = "ccw90"
    manual_video_start_local:    Optional[str] = None


class SyncResult(BaseModel):
    job_id:                       str
    task_name:                    str
    sync_method:                  str
    accuracy_note:                str
    vicon_fps:                    float
    video_fps:                    float
    sync_offset_s:                float
    manual_frame_offset:          int
    first_valid_raw_vicon_frame:  int
    last_valid_raw_vicon_frame:   int
    first_valid_vicon_time_s:     float
    video_at_vicon_start_s:       float
    overlap:                      bool
    vicon_capture_start_local_iso: str
    video_creation_time_utc_iso:  str
    video_creation_time_source:   str
    timezone_offset_hours:        float
    video_rotation:               str
    output_dir:                   str
    sync_json_path:               str
    has_diagnostic_png:           bool
    processing_time_s:            float


# ── Overlay ────────────────────────────────────────────────────────────────────

class OverlayResult(BaseModel):
    job_id:              str
    task_name:           str
    overlay_video:       str
    frame_mapping_csv:   str
    synced_csv:          str
    synced2d_csv:        Optional[str] = None
    has_diagnostic_png:  bool
    total_video_frames:  int
    trimmed_frames:      int
    trimmed_duration_s:  float
    full_skeleton_pct:   float
    output_dir:          str
    processing_time_s:   float


# ── Projects ───────────────────────────────────────────────────────────────────

class ProjectEntry(BaseModel):
    id:            str
    name:          str
    created_at:    str
    sync_method:   Optional[str] = None
    sync_offset_s: Optional[float] = None
    video_rotation: Optional[str] = None
    video_fps:     Optional[float] = None   # real measured fps (sync.json) — e.g. SUB02/03
                                            # are ~30fps while most trials are ~60fps; the
                                            # UI shows this so "synced_60hz.csv" (a fixed
                                            # artifact name, not a literal rate) never gets
                                            # mistaken for confirmation the trial is 60Hz.
    has_overlay:   bool = False
    has_synced_csv: bool = False
    has_diagnostic_png: bool = False
    output_dir:    str
    subject_id:    Optional[str] = None   # parsed from id, e.g. "SUB01" — None for legacy/unassigned
    task:          Optional[str] = None   # e.g. "lifting"
    trial:         Optional[str] = None   # e.g. "T01"


class ProjectList(BaseModel):
    projects: List[ProjectEntry]
    total:    int
