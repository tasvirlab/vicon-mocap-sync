"""
sync.py — /api/sync endpoints
==============================
Vicon-video temporal synchronisation using wall-clock timestamps
(optionally refined by cross-correlation of projected motion signals).

Pipeline:
  1. Upload video, Vicon CSV, XCP file (+ optional intrinsics/extrinsics)
  2. compute_sync()  — wall-clock sync from XCP + ffprobe timestamps
  3. (optional) refine_sync_by_crosscorr()  — ±8 ms optical-flow refinement
  4. Save sync JSON to outputs/<task_name>/
  5. Copy uploaded files to temp/<task_name>/ for later overlay use
"""

from __future__ import annotations
import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse, FileResponse

from models.schemas import JobStarted
from utils import app_config
from utils.job_manager import manager
from utils.task_paths import (task_dir, Artifacts, init_metadata, update_stage,
                              canonicalise_task_id, split_task_id)
from core.sync_pipeline import compute_sync
from core.sync_crosscorr import refine_sync_by_crosscorr

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Background worker ──────────────────────────────────────────────────────────

def _run_sync(
    job_id: str,
    task_name: str,
    video_path: Path,
    vicon_csv: Path,
    xcp_path: Path,
    intrinsics_path: Optional[Path],
    extrinsics_path: Optional[Path],
    cfg: dict,
    xcp_auto_note: Optional[str] = None,
):
    t0 = time.time()
    try:
        manager.update_job(job_id, status="running", progress=5,
                            step="Parsing timestamps…")

        def _cb(pct, step):
            manager.update_job(job_id, progress=5 + int(pct * 0.2), step=step)

        meta = compute_sync(
            xcp_path=xcp_path,
            video_path=video_path,
            vicon_csv=vicon_csv,
            timezone_hours=cfg["timezone_hours"],
            manual_frame_offset=cfg["manual_frame_offset"],
            manual_video_start_local=cfg.get("manual_video_start_local"),
            vicon_fps=cfg["vicon_fps"],
            video_time_correction_hours=cfg["video_time_correction_hours"],
            video_rotation=cfg["video_rotation"],
            progress_cb=_cb,
        )
        if xcp_auto_note:
            meta["xcp_auto_note"] = xcp_auto_note

        output_dir = task_dir(app_config.get_outputs_dir(), task_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        has_diagnostic = False

        # Apply manual force override (bypasses automatic sync entirely)
        if cfg.get("force_sync_offset_s") is not None:
            meta["sync_offset_s"] = round(float(cfg["force_sync_offset_s"]), 6)
            meta["sync_method"]   = "manual_force_sync_offset"
            meta["accuracy_note"] = "sync_offset_s forced manually. Fine-tune with manual_frame_offset."
            meta["overlap"]       = True  # assume user knows what they're doing

        # Look for C3D saved by the Vicon processing step (no extra upload needed)
        c3d_path = app_config.get_temp_dir() / task_name / "trial.c3d"
        if not c3d_path.exists():
            c3d_path = None

        if cfg.get("auto_refine"):
            if not meta.get("overlap", True):
                meta["accuracy_note"] = (
                    "Cross-correlation skipped: no overlap detected between "
                    "video and Vicon. Fix timezone_hours / "
                    "video_time_correction_hours first, or use "
                    "force_sync_offset_s to override."
                )
                logger.warning(
                    f"Job {job_id}: skipping cross-correlation — overlap=False "
                    f"(sync_offset_s={meta['sync_offset_s']:+.1f}s)"
                )
            else:
                manager.update_job(job_id, progress=25,
                                    step="Starting multi-signal refinement…")

                def _refine_cb(pct, step):
                    manager.update_job(job_id, progress=25 + int(pct * 0.65),
                                       step=step)

                refine_result = refine_sync_by_crosscorr(
                    video_path=video_path,
                    vicon_csv=vicon_csv,
                    rough_sync_offset_s=meta["sync_offset_s"],
                    manual_frame_offset=meta["manual_frame_offset"],
                    vicon_fps=meta["vicon_fps"],
                    rotate=meta.get("video_rotation", "none"),
                    search_range_s=cfg.get("search_range_s", 4.0),
                    c3d_path=c3d_path,
                    intrinsics_json=intrinsics_path,
                    extrinsics_json=extrinsics_path,
                    output_dir=output_dir,
                    progress_cb=_refine_cb,
                )

                meta["sync_offset_s"]       = refine_result["refined_offset_s"]
                meta["refine_confidence"]   = refine_result["confidence"]
                meta["refine_conf_label"]   = refine_result["confidence_label"]
                meta["refine_signal_used"]  = refine_result["signal_used"]
                meta["refine_correction_applied"] = refine_result["correction_applied"]
                meta["refine_correction_ms"]      = refine_result.get("correction_ms", 0.0)
                meta["refine_all_signals"]  = refine_result.get("all_signals", {})
                # Signal-agreement audit trail. `needs_review` is the field to
                # filter on before using a trial: confidence alone only says one
                # signal matched something, agreement says two matched the SAME
                # event. Persisted so a bad sync is visible in sync.json rather
                # than only in the app log.
                meta["refine_confidence_metric"] = refine_result.get("confidence_metric", "pearson_r")
                meta["refine_signals_agree"]     = refine_result.get("signals_agree")
                meta["refine_lag_spread_ms"]     = refine_result.get("lag_spread_ms")
                meta["refine_lag_spread_frames"] = refine_result.get("lag_spread_frames")
                # Boundary-clipping flag: a winning lag near the edge of the
                # search window may be clipped rather than a genuine optimum —
                # neither confidence nor signal agreement catches this alone.
                meta["refine_near_search_boundary"] = refine_result.get("near_search_boundary")
                meta["refine_search_range_frames"]  = refine_result.get("search_range_frames")
                meta["needs_review"]             = refine_result.get("needs_review", True)

                if refine_result["correction_applied"]:
                    meta["sync_method"] = "cross_correlation_refined"
                    _agree = refine_result.get("signals_agree", True)
                    _near_boundary = refine_result.get("near_search_boundary", False)
                    meta["accuracy_note"] = (
                        f"Refined by {refine_result['signal_used']} cross-correlation "
                        f"(search ±{cfg.get('search_range_s', 4.0):.1f}s, "
                        f"r={refine_result['confidence']:.2f} "
                        f"{refine_result['confidence_label']}, "
                        f"correction={refine_result.get('correction_ms', 0.0):+.0f} ms)."
                        + ("" if _agree else
                           f" WARNING: signals disagree by "
                           f"{refine_result.get('lag_spread_ms', 0):+.0f} ms — "
                           f"review diagnostic_sync.png before using this trial.")
                        + ("" if not _near_boundary else
                           f" WARNING: winning lag sits at the edge of the "
                           f"±{cfg.get('search_range_s', 4.0):.1f}s search window — "
                           f"the true optimum may lie outside it. Re-run with a "
                           f"wider search_range_s and compare.")
                    )
                else:
                    meta["sync_method"] = "wall_clock_timestamps"
                    meta["accuracy_note"] = (
                        f"Cross-correlation confidence too low "
                        f"({refine_result['confidence_label']}, "
                        f"signal={refine_result['signal_used']}) — "
                        f"wall-clock offset kept. Use Fine-Tune Manual Offset "
                        f"or increase search window."
                    )
                has_diagnostic = (output_dir / Artifacts.QA_DIR / Artifacts.QA_SYNC_DIAG).exists()

        manager.update_job(job_id, progress=92, step="Saving results…")

        # Save sync JSON
        sync_json = output_dir / Artifacts.SYNC_JSON
        with sync_json.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Copy XCP into task folder (needed for timestamp reference)
        task_xcp = output_dir / Artifacts.XCP
        shutil.copy2(str(xcp_path), str(task_xcp))

        # metadata.json is the trial's identity + provenance record, and the
        # marker discover_trials() looks for — a trial folder without it is
        # invisible to every downstream stage. Sync is the first stage to create
        # the folder, so it seeds the file; later stages only top it up.
        outputs_root = app_config.get_outputs_dir()
        init_metadata(
            outputs_root, task_name,
            raw_task_name=cfg.get("raw_task_name"),
            vicon_fps=meta.get("vicon_fps"),
            video_fps=meta.get("video_fps"),
        )
        update_stage(outputs_root, task_name, "sync", "completed",
                     sync_offset_s=meta.get("sync_offset_s"),
                     sync_needs_review=meta.get("needs_review"))

        # Calibration files stay in temp/<task_name>/ — not copied to outputs/

        elapsed = round(time.time() - t0, 2)
        result = {
            **meta,
            "job_id":           job_id,
            "task_name":        task_name,
            "output_dir":       str(output_dir),
            "sync_json_path":   str(sync_json),
            "has_diagnostic_png": has_diagnostic,
            "processing_time_s":  elapsed,
        }

        manager.update_job(job_id, status="completed", progress=100,
                            step="Synchronisation complete", result=result)
        logger.info(f"Sync job {job_id} done in {elapsed:.1f}s  "
                    f"offset={meta['sync_offset_s']:+.3f}s  task={task_name}")

    except Exception as exc:
        logger.error(f"Sync job {job_id} failed: {exc}", exc_info=True)
        manager.update_job(job_id, status="failed", step="Failed", error=str(exc))


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/start", response_model=JobStarted)
async def start_sync(
    video:      UploadFile = File(...),
    vicon_csv:  UploadFile = File(...),
    xcp:        Optional[UploadFile] = File(default=None),
    intrinsics: Optional[UploadFile] = File(default=None),
    extrinsics: Optional[UploadFile] = File(default=None),
    config:     str = Form("{}"),
):
    """
    Start a synchronisation job.

    Required files: video (MP4/MOV), vicon_csv (cleaned CSV).
    xcp (Vicon XCP): upload it directly, or omit it and this endpoint will
    try to auto-find the same-stem sibling next to the raw C3D/CSV under
    the raw Vicon folder, <subject>/ (see app_config.find_raw_sibling) — only works
    when vicon_csv's ORIGINAL filename still matches the raw Nexus export
    name; a cleaned/renamed CSV (e.g. from the C3D→Sync handoff) won't
    match, and XCP must be uploaded manually in that case.
    Optional files: intrinsics JSON, extrinsics JSON (needed for auto_refine).

    Config JSON fields:
      task_name                   (default: video filename stem)
      timezone_hours              (default: 5.0  — UTC+5 Uzbekistan)
      video_time_correction_hours (default: 0.0)
      vicon_fps                   (default: 100.0)
      manual_frame_offset         (default: 0)
      auto_refine                 (default: true — requires intrinsics+extrinsics)
      search_range_s              (default: 4.0)
      video_rotation              (default: "ccw90" — for iPhone MOV)
      manual_video_start_local    (default: null — use ffprobe)
    """
    cfg = json.loads(config)
    task_name = cfg.get("task_name")
    if not task_name:
        raise HTTPException(status_code=422, detail="task_name is required in config")

    # Strict resolution for NEW writes: canonicalise the task component (mapping
    # shorthands like 'trunk_flexion' -> 'trunk_flexion_twisting') or FAIL LOUDLY.
    # Without this an unrecognised task silently fell through to a flat legacy
    # path and created an invisible project (the repeated trunk_flexion /
    # trunk_forward_bending bug). Read paths stay lenient; only writes are strict.
    try:
        task_name = canonicalise_task_id(task_name)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    job_id = manager.create_job()
    app_config.get_temp_dir().mkdir(exist_ok=True)

    # Save uploaded files to temp/<task_name>/
    task_temp = app_config.get_temp_dir() / task_name
    task_temp.mkdir(parents=True, exist_ok=True)

    video_ext  = Path(video.filename or "video.mp4").suffix or ".mp4"
    video_path = task_temp / f"video{video_ext}"
    video_path.write_bytes(await video.read())

    vicon_path = task_temp / "vicon.csv"
    vicon_path.write_bytes(await vicon_csv.read())

    xcp_auto_note: Optional[str] = None
    if xcp is not None:
        xcp_path = task_temp / (xcp.filename or f"{task_name}.xcp")
        xcp_path.write_bytes(await xcp.read())
    else:
        # No XCP uploaded — try the same-stem sibling next to the raw CSV.
        subject, _, _ = split_task_id(task_name)
        found = app_config.find_raw_sibling(subject, vicon_csv.filename or "", "xcp")
        if found is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No XCP uploaded and none auto-matched next to the raw Vicon CSV "
                    f"for {subject}. XCP is required for wall-clock sync — upload it "
                    "manually, or confirm vicon_csv's filename still matches its raw "
                    "Nexus export name (a cleaned/renamed CSV won't auto-match)."
                ),
            )
        xcp_path = task_temp / found.name
        xcp_path.write_bytes(found.read_bytes())
        xcp_auto_note = f"XCP auto-matched: {found.name}"

    intr_path = None
    if intrinsics:
        intr_path = task_temp / "intrinsic.json"
        intr_path.write_bytes(await intrinsics.read())

    extr_path = None
    if extrinsics:
        extr_path = task_temp / "extrinsics.json"
        extr_path.write_bytes(await extrinsics.read())

    fso = cfg.get("force_sync_offset_s")
    sync_cfg = {
        "timezone_hours":              float(cfg.get("timezone_hours", 5.0)),
        "video_time_correction_hours": float(cfg.get("video_time_correction_hours", 0.0)),
        "vicon_fps":                   float(cfg.get("vicon_fps", 100.0)),
        "manual_frame_offset":         int(cfg.get("manual_frame_offset", 0)),
        "auto_refine":                 bool(cfg.get("auto_refine", True)),
        "search_range_s":              float(cfg.get("search_range_s", 4.0)),
        "video_rotation":              str(cfg.get("video_rotation", "ccw90")),
        "manual_video_start_local":    cfg.get("manual_video_start_local"),
        "force_sync_offset_s":         float(fso) if fso is not None else None,
    }

    manager.update_job(job_id, status="queued", step="Starting…")
    threading.Thread(
        target=_run_sync,
        args=(job_id, task_name, video_path, vicon_path, xcp_path,
              intr_path, extr_path, sync_cfg, xcp_auto_note),
        daemon=True,
    ).start()

    return JobStarted(job_id=job_id, message=f"Sync started for task '{task_name}'")


@router.get("/check-sibling")
async def check_sibling_xcp(subject: str, filename: str):
    """
    Read-only pre-check: does a same-stem XCP exist next to this Vicon CSV's
    original filename in the raw Vicon folder (<subject>/)?

    Called live from the frontend as soon as a Vicon CSV is picked (before
    Sync is started), mirroring /api/vicon/check-sibling. Never writes
    anything.
    """
    found = app_config.find_raw_sibling(subject, filename, "xcp")
    return {"found": found is not None, "filename": found.name if found else None}


@router.get("/fetch-sibling")
async def fetch_sibling_xcp(subject: str, filename: str):
    """
    Serve the raw bytes of the same-stem sibling XCP — lets the frontend drop
    a confirmed check-sibling match straight into the XCP dropzone as a real
    File object, matching Vicon.vue's fallback-CSV auto-fill.
    """
    found = app_config.find_raw_sibling(subject, filename, "xcp")
    if found is None:
        raise HTTPException(status_code=404, detail="No matching sibling XCP found")
    return FileResponse(str(found), filename=found.name, media_type="application/xml")


@router.get("/{job_id}/stream")
async def stream_sync(job_id: str):
    """SSE progress stream for a sync job."""
    if manager.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return StreamingResponse(
        manager.stream_job(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{job_id}/result")
async def get_sync_result(job_id: str):
    """Return the completed sync result dict."""
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == "failed":
        raise HTTPException(status_code=422, detail=job.error or "Sync failed")
    if job.status != "completed":
        raise HTTPException(status_code=202, detail="Job not yet complete")
    return job.result


@router.post("/{task_name}/reapply")
async def reapply_offset(task_name: str, body: dict):
    """
    Update manual_frame_offset in the saved sync JSON without re-running the pipeline.

    Body: {"manual_frame_offset": int}
    """
    output_dir = task_dir(app_config.get_outputs_dir(), task_name)
    sync_json = output_dir / Artifacts.SYNC_JSON
    if not sync_json.exists():
        raise HTTPException(status_code=404, detail=f"No sync JSON for task '{task_name}'")

    with sync_json.open(encoding="utf-8") as f:
        meta = json.load(f)

    new_offset = int(body.get("manual_frame_offset", meta["manual_frame_offset"]))
    meta["manual_frame_offset"] = new_offset

    with sync_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {**meta, "task_name": task_name,
            "output_dir": str(output_dir),
            "sync_json_path": str(sync_json)}
