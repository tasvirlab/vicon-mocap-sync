"""
task_paths.py
==============
The output-layout core: canonical task names, per-trial folder resolution,
path-based discovery, artifact filenames, and per-trial metadata.

Layout
──────
    outputs/
    └── SUB01/
        ├── 01_lifting/
        │   ├── SUB01_lifting_T01/          <- one trial = one folder
        │   │   ├── metadata.json           <- identity + provenance + stage log
        │   │   ├── vicon_100hz.csv
        │   │   ├── sync.json
        │   │   ├── synced_60hz.csv
        │   │   ├── synced_vicon2d_60hz.csv   <- Vicon 2D PROJECTED (not RTMPose)
        │   │   ├── frame_mapping.csv
        │   │   ├── rgb.mp4
        │   │   ├── pose2d_rtmpose.csv
        │   │   ├── trial.xcp
        │   │   ├── reba/                   <- multi-file stage
        │   │   ├── qa/                     <- verification only
        │   │   └── reports/                <- created on demand
        │   ├── SUB01_lifting_T02/
        │   └── SUB01_lifting_T03/
        └── calibration/                    <- static trials, NOT a task
    ├── _dataset_exports/                   <- cross-subject
    └── _UNASSIGNED/                        <- legacy quarantine

Three naming layers, each with exactly one job:
  task_id      "SUB01_lifting_T01"  — the API identifier. Unchanged from the
                                      flat era: form fields, URLs, SSE streams
                                      and job ids all still pass this string.
  task folder  "01_lifting"          — protocol order + canonical spelling.
  trial folder "SUB01_lifting_T01"   — self-identifying when copied out, which
                                      matters because the FILENAMES no longer
                                      are (synced_60hz.csv is anonymous alone).

Why filenames carry no task_id
──────────────────────────────
The folder is the single source of truth for identity; filenames describe only
the artifact type. This is what makes the layout extensible — a new stage adds
`pose3d_lifted.csv` to an existing folder instead of inventing a new prefix
and touching path logic. The cost (a loose file is anonymous) is paid off by
metadata.json travelling in the same folder.

Why per-trial folders
─────────────────────
The previous layout put ALL trials of a task in one folder, disambiguated by a
filename prefix. Anything written WITHOUT that prefix — reba/, diagnostic_sync.png,
pose2d_mapping_check.jpg — silently overwrote the previous trial's copy (T03 ate
T01's REBA). That was not a bug to patch but a layout that permitted it. Here a
collision is unrepresentable.
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PIPELINE_VERSION = "2.0"          # bump when the layout or artifact set changes

_TASK_ID_RE = re.compile(r"^(SUB\d+)_(.+)_(T\d+)$")
_SUBJECT_RE = re.compile(r"^SUB\d+$")
_TASK_FOLDER_RE = re.compile(r"^\d{2}_")

# Folders under outputs/ that are not subjects.
RESERVED_ROOT_DIRS = ("_dataset_exports", "_UNASSIGNED")
# Folder under a subject that holds static/calibration trials — deliberately NOT
# one of the eight canonical tasks (static captures exist only for extrinsic
# calibration).
CALIBRATION_DIR = "calibration"


# ── Canonical tasks ───────────────────────────────────────────────────────────
# The eight-task protocol is fixed and does NOT depend on how the
# raw Vicon files happen to be named. Order here is protocol order.
CANONICAL_TASKS: dict[str, str] = {
    "lifting":                 "01_lifting",
    "overhead_work":           "02_overhead_work",
    "forward_trunk_bending":   "03_forward_trunk_bending",
    "squatting":               "04_squatting",
    "kneeling":                "05_kneeling",
    "handling":                "06_handling",
    "carrying":                "07_carrying",
    "trunk_flexion_twisting":  "08_trunk_flexion_twisting",
}

# Raw Vicon trial names -> canonical task.
#
# [WARNING] The raw task NUMBERS do not match the canonical numbers, because raw
# Task4 splits into two canonical tasks:
#     raw Task4-1 -> 04_squatting
#     raw Task4-2 -> 05_kneeling      <- split; everything after shifts by one
#     raw Task5   -> 06_handling
#     raw Task6   -> 07_carrying      <- and fixes the "carring" typo
#     raw Task7   -> 08_trunk_flexion_twisting
# Never infer the canonical number from the raw number.
#
# Keys are matched after _normalise_raw() (lowercased, punctuation/space -> "_",
# runs collapsed), so "Task5 handling", "Task5_handling" and "task5  handling"
# all hit the same entry.
RAW_TASK_ALIASES: dict[str, str] = {
    # canonical names map to themselves
    **{k: k for k in CANONICAL_TASKS},
    # raw Vicon file names (original capture naming)
    "task1_lifting":                                   "lifting",
    "task2_overhead_work":                             "overhead_work",
    "task3_forward_trunk_bending":                     "forward_trunk_bending",
    "task4_1_squatting":                               "squatting",
    "task4_2_kneeling":                                "kneeling",
    "task5_handling":                                  "handling",
    "task6_carring":                                   "carrying",   # raw typo
    "task6_carrying":                                  "carrying",
    "task7_reaching_with_trunk_flexion_axis_rotation": "trunk_flexion_twisting",
    # tolerated shorthands
    "overhead":                                        "overhead_work",
    "forward_bending":                                 "forward_trunk_bending",
    "trunk_forward_bending":                           "forward_trunk_bending",   # word-order variant seen in the wild
    "trunk_flexion_axial_twisting":                    "trunk_flexion_twisting",
    "trunk_flexion":                                   "trunk_flexion_twisting",   # shorthand for task 8 (raw Task7); NOT task 3 forward bending
    "trunk_flexion_twisting":                          "trunk_flexion_twisting",
}


def _normalise_raw(raw: str) -> str:
    """Lowercase, collapse any run of non-alphanumerics to a single underscore."""
    return re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")


def canonical_task(raw: str) -> str:
    """Map a raw/free-text task name to its canonical name.

    Args:
        raw: e.g. "Task6 carring", "Task4-2_kneeling", "lifting".

    Returns:
        Canonical task name, e.g. "carrying".

    Raises:
        ValueError: if `raw` is not a recognised task. This is deliberately a
            hard failure: silently accepting an unknown name is how "carrying",
            "carring" and "carry" become three folders across ten subjects. Add
            an explicit RAW_TASK_ALIASES entry instead.
    """
    key = _normalise_raw(raw)
    if key in RAW_TASK_ALIASES:
        return RAW_TASK_ALIASES[key]
    raise ValueError(
        f"Unrecognised task name {raw!r} (normalised: {key!r}). The eight-task "
        f"protocol is fixed: {', '.join(CANONICAL_TASKS)}. If this is a genuine "
        f"new alias for one of them, add it to RAW_TASK_ALIASES — do not rely on "
        f"free text, or a typo silently creates a ninth folder."
    )


def task_folder(task: str) -> str:
    """Canonical task name -> numbered folder name ("lifting" -> "01_lifting")."""
    canon = canonical_task(task)
    return CANONICAL_TASKS[canon]


# ── task_id ───────────────────────────────────────────────────────────────────

def build_task_id(subject_id: str, task: str, trial: str) -> str:
    """Compose the API task_id. The task part is canonicalised.

    Args:
        subject_id: "SUB01".
        task:       raw or canonical task name.
        trial:      "T01".

    Returns:
        "SUB01_lifting_T01" — canonical task name, NOT the numbered folder, so
        the id stays readable and stable if folder numbering ever changes.
    """
    return f"{subject_id}_{canonical_task(task)}_{trial}"


def split_task_id(task_id: str) -> tuple[str, str, str]:
    """Split "SUB01_lifting_T01" -> ("SUB01", "lifting", "T01").

    Raises:
        ValueError: if it doesn't match SUB<NN>_<task>_T<NN>.
    """
    m = _TASK_ID_RE.match(task_id)
    if not m:
        raise ValueError(f"'{task_id}' is not a SUB<NN>_<task>_T<NN> task id")
    return m.group(1), m.group(2), m.group(3)


def canonicalise_task_id(task_id: str) -> str:
    """Return `task_id` with its task component canonicalised, or RAISE.

    'SUB04_trunk_flexion_T01' -> 'SUB04_trunk_flexion_twisting_T01' (via aliases).

    Use this at WRITE time (the sync stage) so an unrecognised or shorthand task
    name is either mapped to its canonical form or fails LOUDLY with the list of
    valid tasks — instead of silently falling through trial_dir()'s legacy
    flat-path fallback and creating an invisible project (the repeated
    `trunk_forward_bending` / `trunk_flexion` bug). Read paths stay lenient;
    only new writes are strict.

    Raises:
        ValueError: if the id isn't SUB<NN>_<task>_T<NN>, or the task is unknown.
    """
    subject_id, task, trial = split_task_id(task_id)
    canon = canonical_task(task)           # raises (with valid-task list) on unknown
    return f"{subject_id}_{canon}_{trial}"


# ── Artifact filenames ────────────────────────────────────────────────────────

class Artifacts:
    """Every filename the pipeline writes, in one place.

    Filenames are descriptive of CONTENT, never of identity — identity lives in
    the folder path and metadata.json. Names state the sampling rate where two
    files hold the same quantity at different rates (vicon_100hz vs synced_60hz),
    and the MODEL where a stage can have competing implementations
    (pose2d_rtmpose vs a future pose2d_openpose), so benchmarking two models on
    the same trial never overwrites.

    Import these constants; do not hardcode the strings.
    """
    METADATA        = "metadata.json"

    # Stage 1 — C3D convert (native Vicon rate)
    VICON_100HZ     = "vicon_100hz.csv"
    XCP             = "trial.xcp"

    # Stage 2 — sync
    SYNC_JSON       = "sync.json"

    # Stage 3 — overlay (resampled to video frames)
    SYNCED_60HZ     = "synced_60hz.csv"
    # Named "vicon2d" (not just "2d") on purpose: this is the Vicon 3D
    # skeleton PROJECTED through the camera calibration, not RTMPose output.
    # pose2d_rtmpose.csv is a completely different source with the same
    # {joint}_X/_Y column shape — confusing the two silently mispairs
    # ground-truth-derived pixels with model-detected pixels.
    SYNCED2D_60HZ   = "synced_vicon2d_60hz.csv"
    FRAME_MAPPING   = "frame_mapping.csv"
    RGB_VIDEO       = "rgb.mp4"

    # Stage 5 — 2D pose (model-specific)
    POSE2D_RTMPOSE  = "pose2d_rtmpose.csv"

    # Stage 6 — 3D pose (model-specific; future)
    POSE3D_LIFTED = "pose3d_lifted.csv"

    # Subfolders
    REBA_DIR        = "reba"
    QA_DIR          = "qa"
    REPORTS_DIR     = "reports"

    # reba/ — the three ergonomic channels + shared angles
    REBA_ANGLES     = "angles.csv"            # all anatomical angles per frame
    REBA_SCORES     = "reba_scores.csv"       # Channel 1: per-frame REBA (matches writer)
    REBA_POSTURE_DURATION = "posture_duration.json"  # Channel 2: hysteresis bouts + provisional ISO
    REBA_TRUNK_DYNAMICS_CSV  = "trunk_dynamics.csv"  # Channel 3: per-frame trunk angle/vel/acc
    REBA_TRUNK_DYNAMICS_JSON = "trunk_dynamics.json" # Channel 3: Marras predictors + probability
    REBA_RESULT     = "result.json"           # combined summary of all three channels
    # NOTE: the former REBA_CUMULATIVE ("cumulative_scores.csv") / REBA_DOSE
    # ("dose.json") were removed: the output reports per-task posture, duration
    # and trunk-dynamics channels, not a cumulative exposure index.

    # qa/ — verification and debugging only, never an analysis input
    QA_OVERLAY      = "overlay.mp4"
    QA_SYNC_DIAG    = "diagnostic_sync.png"
    QA_POSE2D_CHECK = "pose2d_mapping_check.jpg"


# ── Path resolution ───────────────────────────────────────────────────────────

def subject_dir(outputs_dir: Path, subject_id: str) -> Path:
    return outputs_dir / subject_id


def calibration_dir(outputs_dir: Path, subject_id: str) -> Path:
    """Static/calibration captures for a subject — outside the task folders."""
    return outputs_dir / subject_id / CALIBRATION_DIR


def trial_dir(outputs_dir: Path, task_id: str) -> Path:
    """Resolve a task_id to its trial folder.

    "SUB01_lifting_T01" -> outputs/SUB01/01_lifting/SUB01_lifting_T01/

    Legacy ids that don't match SUB<NN>_<task>_T<NN>, or whose task is not in the
    whitelist, resolve to outputs/<task_id>/ unchanged so old flat folders and
    _UNASSIGNED entries stay reachable.
    """
    try:
        subject_id, task, _trial = split_task_id(task_id)
        return outputs_dir / subject_id / task_folder(task) / task_id
    except ValueError:
        return outputs_dir / task_id


# Kept so existing imports don't break; trial_dir() is the real name now.
task_dir = trial_dir


def reba_dir(outputs_dir: Path, task_id: str) -> Path:
    return trial_dir(outputs_dir, task_id) / Artifacts.REBA_DIR


def qa_dir(outputs_dir: Path, task_id: str) -> Path:
    return trial_dir(outputs_dir, task_id) / Artifacts.QA_DIR


def reports_dir(outputs_dir: Path, task_id: str) -> Path:
    """Reports folder. Created on demand by whatever writes a report — an empty
    folder in all 240 trials would be pure clutter."""
    return trial_dir(outputs_dir, task_id) / Artifacts.REPORTS_DIR


# ── Discovery (path-based) ────────────────────────────────────────────────────

def is_trial_dir(path: Path) -> bool:
    """True if `path` is a trial folder: named like a task_id and holding metadata."""
    return path.is_dir() and bool(_TASK_ID_RE.match(path.name)) \
        and (path / Artifacts.METADATA).is_file()


def discover_trials(outputs_dir: Path) -> list[tuple[str, Path]]:
    """Find every trial folder by PATH STRUCTURE.

    Walks outputs/<SUBnn>/<nn_task>/<task_id>/ and yields folders that carry a
    metadata.json. The task_id comes from the FOLDER NAME, not from any filename
    — which is what lets artifacts be named for their content alone.

    metadata.json is the marker because it is written when the trial folder is
    created and is the one file every trial has regardless of which stages have
    run. A folder without it is incomplete/hand-made and is skipped rather than
    half-reported.

    Returns:
        Sorted [(task_id, trial_dir)]. Excludes _dataset_exports/, _UNASSIGNED/
        and calibration/.
    """
    if not outputs_dir.exists():
        return []

    results: list[tuple[str, Path]] = []
    for subj in sorted(outputs_dir.iterdir()):
        if not subj.is_dir() or subj.name in RESERVED_ROOT_DIRS:
            continue
        if not _SUBJECT_RE.match(subj.name):
            continue
        for task_d in sorted(subj.iterdir()):
            if not task_d.is_dir() or not _TASK_FOLDER_RE.match(task_d.name):
                continue          # skips calibration/ and any stray folder
            for trial_d in sorted(task_d.iterdir()):
                if is_trial_dir(trial_d):
                    results.append((trial_d.name, trial_d))
    return sorted(results)


# Legacy alias. Note the semantics differ deliberately: the old version found
# trials by rglob("*_sync.json") at ANY depth, including _UNASSIGNED/ and flat
# folders. The new one reports only properly-placed trial folders — which is the
# point of the restructure.
discover_task_dirs = discover_trials


# ── Metadata ──────────────────────────────────────────────────────────────────

def _git_commit() -> Optional[str]:
    """Short git SHA of the code that produced this trial, or None.

    The GIT_COMMIT environment variable is read first (set it when the backend runs from a
    copy without a `.git` folder); otherwise `git rev-parse` is used when running from a checkout.
    Recording the commit makes results reproducible: the same trial can give different sync
    corrections across code versions, so each trial's metadata says which version produced it.
    """
    import os
    env = os.environ.get("GIT_COMMIT", "").strip()
    if env and env.lower() not in ("", "unknown", "none"):
        return env
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


STAGES = ("c3d", "sync", "overlay", "pose2d", "pose3d", "reba")


def init_metadata(outputs_dir: Path, task_id: str, *,
                  raw_task_name: Optional[str] = None,
                  vicon_fps: Optional[float] = None,
                  video_fps: Optional[float] = None,
                  video_resolution: Optional[str] = None) -> Path:
    """Create (or top up) a trial's metadata.json and return its path.

    Makes every trial self-describing: months later, or during a paper revision,
    the folder itself answers what produced it and with which code. git_commit is
    the reason this exists — the same trial gave -149.9 ms and then -217 ms of
    sync correction across two code versions, and nothing on disk recorded which
    was which.

    Existing values are preserved; only missing/None fields are filled, so
    re-running a stage never erases earlier provenance.
    """
    path = trial_dir(outputs_dir, task_id) / Artifacts.METADATA
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = read_metadata(outputs_dir, task_id) or {}
    try:
        subject_id, task, trial = split_task_id(task_id)
        folder = task_folder(task)
    except ValueError:
        subject_id, task, trial, folder = None, None, None, None

    defaults = {
        "task_id":          task_id,
        "participant":      subject_id,
        "task":             task,
        "task_folder":      folder,
        "trial":            trial,
        "raw_task_name":    raw_task_name,
        "video_fps":        video_fps,
        "video_resolution": video_resolution,
        "vicon_fps":        vicon_fps,
        "pipeline_version": PIPELINE_VERSION,
        "git_commit":       _git_commit(),
        "processing_date":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    for k, v in defaults.items():
        if meta.get(k) in (None, "") and v is not None:
            meta[k] = v
    meta.setdefault("stages", {s: "pending" for s in STAGES})

    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def read_metadata(outputs_dir: Path, task_id: str) -> Optional[dict]:
    """Return a trial's metadata dict, or None if absent/unreadable."""
    path = trial_dir(outputs_dir, task_id) / Artifacts.METADATA
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def update_stage(outputs_dir: Path, task_id: str, stage: str, status: str,
                 **fields) -> None:
    """Record a stage's status (and optionally top-level fields) in metadata.json.

    Args:
        stage:  one of STAGES.
        status: "pending" | "running" | "completed" | "failed".
        **fields: extra top-level keys to set/overwrite (e.g. video_fps once the
            overlay stage has actually measured it).

    No-ops if the trial folder doesn't exist — a processing stage should not
    crash because bookkeeping failed.
    """
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {STAGES}")
    td = trial_dir(outputs_dir, task_id)
    if not td.is_dir():
        return
    meta = read_metadata(outputs_dir, task_id)
    if not meta:
        init_metadata(outputs_dir, task_id)
        meta = read_metadata(outputs_dir, task_id) or {}
    meta.setdefault("stages", {s: "pending" for s in STAGES})
    meta["stages"][stage] = status
    meta["git_commit"] = _git_commit()
    meta["processing_date"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for k, v in fields.items():
        if v is not None:
            meta[k] = v
    (td / Artifacts.METADATA).write_text(json.dumps(meta, indent=2), encoding="utf-8")
