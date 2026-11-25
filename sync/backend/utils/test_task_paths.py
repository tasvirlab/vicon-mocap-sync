"""Regression tests for utils.task_paths — run with: python test_task_paths.py

This module decides where every artifact in the pipeline is written and how
every downstream stage finds it, so it is tested before anything depends on it.
Two properties matter most:

  1. The canonical mapping is the ONLY thing standing between inconsistent raw
     Vicon names ("Task6 carring", "Task5 handling") and a directory tree with
     nine task folders across ten subjects.
  2. Discovery is path-based now. If it silently returns nothing, every batch
     stage quietly processes zero trials and reports success.

No pytest dependency — plain asserts, runnable standalone.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # backend/
from utils import task_paths as tp   # noqa: E402


# ── Canonical task mapping ────────────────────────────────────────────────────

def test_raw_vicon_names_map_to_canonical():
    """Every original capture name must map."""
    cases = {
        "Task1_lifting":                                   "lifting",
        "Task2_overhead work":                             "overhead_work",
        "Task3_forward trunk bending":                     "forward_trunk_bending",
        "Task4-1_squatting":                               "squatting",
        "Task4-2_kneeling":                                "kneeling",
        "Task5 handling":                                  "handling",
        "Task6 carring":                                   "carrying",
        "Task7 reaching with trunk flexion axis rotation": "trunk_flexion_twisting",
    }
    for raw, expected in cases.items():
        assert tp.canonical_task(raw) == expected, f"{raw!r} -> {tp.canonical_task(raw)!r}"


def test_raw_task_numbers_are_not_canonical_numbers():
    """raw Task4 splits into 04+05, so every later raw number is off by one.

    This is the subtlety the whitelist exists for: inferring the folder number
    from the raw filename would put handling in 05, carrying in 06, etc.
    """
    assert tp.task_folder(tp.canonical_task("Task4-1_squatting")) == "04_squatting"
    assert tp.task_folder(tp.canonical_task("Task4-2_kneeling")) == "05_kneeling"
    assert tp.task_folder(tp.canonical_task("Task5 handling")) == "06_handling"
    assert tp.task_folder(tp.canonical_task("Task6 carring")) == "07_carrying"
    assert tp.task_folder(tp.canonical_task(
        "Task7 reaching with trunk flexion axis rotation")) == "08_trunk_flexion_twisting"


def test_raw_typo_is_corrected():
    """The raw files say 'carring'; the dataset must say 'carrying'."""
    assert tp.canonical_task("Task6 carring") == "carrying"
    assert tp.canonical_task("Task6 carrying") == "carrying"
    assert tp.task_folder("carring" if False else "carrying") == "07_carrying"


def test_normalisation_is_punctuation_and_case_insensitive():
    """'Task5 handling', 'Task5_handling', 'task5  HANDLING' are one task."""
    for variant in ("Task5 handling", "Task5_handling", "task5  HANDLING",
                    "TASK5-handling", "  Task5   handling  "):
        assert tp.canonical_task(variant) == "handling", variant


def test_unknown_task_fails_loudly():
    """An unrecognised name must raise, not create a ninth folder."""
    for bad in ("Task9_walking", "lifitng", "", "random"):
        try:
            tp.canonical_task(bad)
            raise AssertionError(f"expected ValueError for {bad!r}")
        except ValueError:
            pass


def test_all_eight_tasks_have_unique_ordered_folders():
    folders = list(tp.CANONICAL_TASKS.values())
    assert len(folders) == 8, folders
    assert len(set(folders)) == 8, "duplicate folder names"
    assert folders == sorted(folders), "folder numbering must sort in protocol order"
    for i, f in enumerate(folders, start=1):
        assert f.startswith(f"{i:02d}_"), f"{f} should start with {i:02d}_"


# ── task_id ───────────────────────────────────────────────────────────────────

def test_build_task_id_canonicalises_the_task():
    """A raw name typed in the UI must not leak into the id."""
    assert tp.build_task_id("SUB01", "Task6 carring", "T01") == "SUB01_carrying_T01"
    assert tp.build_task_id("SUB01", "lifting", "T01") == "SUB01_lifting_T01"


def test_split_task_id_roundtrip():
    tid = tp.build_task_id("SUB07", "Task4-2_kneeling", "T03")
    assert tid == "SUB07_kneeling_T03"
    assert tp.split_task_id(tid) == ("SUB07", "kneeling", "T03")


def test_split_task_id_rejects_legacy_names():
    for bad in ("Task1_lifting_01", "SUB01_Lifting", "random_folder"):
        try:
            tp.split_task_id(bad)
            raise AssertionError(f"expected ValueError for {bad!r}")
        except ValueError:
            pass


# ── Path resolution ───────────────────────────────────────────────────────────

def test_trial_dir_is_per_trial():
    out = Path("/out")
    d1 = tp.trial_dir(out, "SUB01_lifting_T01")
    d2 = tp.trial_dir(out, "SUB01_lifting_T02")
    assert d1 == out / "SUB01" / "01_lifting" / "SUB01_lifting_T01"
    assert d1 != d2, "trials of the same task must not share a folder"
    # the whole point: an unprefixed artifact cannot collide across trials
    assert (d1 / tp.Artifacts.SYNCED_60HZ) != (d2 / tp.Artifacts.SYNCED_60HZ)
    assert (d1 / tp.Artifacts.REBA_DIR) != (d2 / tp.Artifacts.REBA_DIR)


def test_trial_dir_legacy_ids_stay_flat():
    out = Path("/out")
    assert tp.trial_dir(out, "Task1_lifting_01") == out / "Task1_lifting_01"


def test_calibration_is_outside_the_task_folders():
    out = Path("/out")
    cal = tp.calibration_dir(out, "SUB01")
    assert cal == out / "SUB01" / "calibration"
    assert not tp._TASK_FOLDER_RE.match(cal.name), \
        "calibration must not look like a numbered task folder"


def test_subfolders_resolve_under_the_trial():
    out = Path("/out")
    td = tp.trial_dir(out, "SUB01_lifting_T01")
    assert tp.reba_dir(out, "SUB01_lifting_T01") == td / "reba"
    assert tp.qa_dir(out, "SUB01_lifting_T01") == td / "qa"
    assert tp.reports_dir(out, "SUB01_lifting_T01") == td / "reports"


# ── Discovery ─────────────────────────────────────────────────────────────────

def _make_tree(root: Path) -> None:
    """Build a synthetic outputs/ tree with the real edge cases in it."""
    for tid in ("SUB01_lifting_T01", "SUB01_lifting_T02", "SUB01_carrying_T01",
                "SUB02_kneeling_T01"):
        tp.init_metadata(root, tid)
    # a trial folder with NO metadata -> incomplete, must be skipped
    (root / "SUB03" / "01_lifting" / "SUB03_lifting_T01").mkdir(parents=True)
    # calibration must never be reported as a trial
    (root / "SUB01" / "calibration").mkdir(parents=True, exist_ok=True)
    (root / "SUB01" / "calibration" / "static01.c3d").write_text("x")
    # reserved roots must be ignored
    (root / "_dataset_exports" / "20260101").mkdir(parents=True)
    (root / "_UNASSIGNED" / "Task1_lifting_01").mkdir(parents=True)
    (root / "_UNASSIGNED" / "Task1_lifting_01" / "metadata.json").write_text("{}")


def test_discovery_is_path_based():
    root = Path(tempfile.mkdtemp())
    try:
        _make_tree(root)
        found = tp.discover_trials(root)
        ids = [t for t, _ in found]
        assert ids == ["SUB01_carrying_T01", "SUB01_lifting_T01",
                       "SUB01_lifting_T02", "SUB02_kneeling_T01"], ids
        # task_id comes from the FOLDER, not from any filename
        for tid, d in found:
            assert d.name == tid
        # and the path is fully resolved
        assert dict(found)["SUB01_carrying_T01"] == \
            root / "SUB01" / "07_carrying" / "SUB01_carrying_T01"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_discovery_skips_incomplete_calibration_and_reserved():
    root = Path(tempfile.mkdtemp())
    try:
        _make_tree(root)
        ids = [t for t, _ in tp.discover_trials(root)]
        assert "SUB03_lifting_T01" not in ids, "folder without metadata.json must be skipped"
        assert not any("calibration" in t for t in ids)
        assert not any(t == "Task1_lifting_01" for t in ids), \
            "_UNASSIGNED must not be reported as a trial"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_discovery_on_missing_dir_returns_empty():
    assert tp.discover_trials(Path(tempfile.mkdtemp()) / "nope") == []


def test_discovery_finds_no_files_by_name():
    """A stray *_sync.json outside a trial folder must not create a phantom trial.

    The OLD discovery globbed for *_sync.json at any depth and would have picked
    this up.
    """
    root = Path(tempfile.mkdtemp())
    try:
        tp.init_metadata(root, "SUB01_lifting_T01")
        stray = root / "SUB01" / "01_lifting" / "SUB01_lifting_T99_sync.json"
        stray.write_text("{}")
        ids = [t for t, _ in tp.discover_trials(root)]
        assert ids == ["SUB01_lifting_T01"], ids
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── Metadata ──────────────────────────────────────────────────────────────────

def test_metadata_is_self_describing():
    root = Path(tempfile.mkdtemp())
    try:
        tp.init_metadata(root, "SUB01_carrying_T02",
                         raw_task_name="Task6 carring",
                         vicon_fps=100.0, video_fps=60.0283,
                         video_resolution="1920x1080")
        m = tp.read_metadata(root, "SUB01_carrying_T02")
        assert m["participant"] == "SUB01"
        assert m["task"] == "carrying"
        assert m["task_folder"] == "07_carrying"
        assert m["trial"] == "T02"
        assert m["raw_task_name"] == "Task6 carring", "provenance back to the raw file"
        assert m["video_fps"] == 60.0283, "true rate, not the nominal 60"
        assert m["video_resolution"] == "1920x1080"
        assert m["vicon_fps"] == 100.0
        assert m["pipeline_version"] == tp.PIPELINE_VERSION
        assert set(m["stages"]) == set(tp.STAGES)
        assert all(v == "pending" for v in m["stages"].values())
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_update_stage_records_progress():
    root = Path(tempfile.mkdtemp())
    try:
        tp.init_metadata(root, "SUB01_lifting_T01")
        tp.update_stage(root, "SUB01_lifting_T01", "sync", "completed")
        tp.update_stage(root, "SUB01_lifting_T01", "overlay", "failed")
        m = tp.read_metadata(root, "SUB01_lifting_T01")
        assert m["stages"]["sync"] == "completed"
        assert m["stages"]["overlay"] == "failed"
        assert m["stages"]["pose3d"] == "pending"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_init_metadata_never_erases_earlier_provenance():
    """Re-running a stage must not wipe fields recorded by an earlier one."""
    root = Path(tempfile.mkdtemp())
    try:
        tp.init_metadata(root, "SUB01_lifting_T01", raw_task_name="Task1_lifting",
                         vicon_fps=100.0)
        tp.update_stage(root, "SUB01_lifting_T01", "c3d", "completed")
        # a later stage inits again with different/missing info
        tp.init_metadata(root, "SUB01_lifting_T01", video_fps=60.0283)
        m = tp.read_metadata(root, "SUB01_lifting_T01")
        assert m["raw_task_name"] == "Task1_lifting", "earlier provenance survived"
        assert m["vicon_fps"] == 100.0
        assert m["video_fps"] == 60.0283, "new field added"
        assert m["stages"]["c3d"] == "completed", "stage log survived"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_update_stage_rejects_unknown_stage():
    root = Path(tempfile.mkdtemp())
    try:
        tp.init_metadata(root, "SUB01_lifting_T01")
        try:
            tp.update_stage(root, "SUB01_lifting_T01", "magic", "completed")
            raise AssertionError("expected ValueError for unknown stage")
        except ValueError:
            pass
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_read_metadata_missing_returns_none():
    root = Path(tempfile.mkdtemp())
    try:
        assert tp.read_metadata(root, "SUB01_lifting_T01") is None
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_artifact_names_are_unique_and_unprefixed():
    """Filenames describe content only — no task_id may leak into them."""
    names = [v for k, v in vars(tp.Artifacts).items()
             if not k.startswith("_") and isinstance(v, str)]
    assert len(names) == len(set(names)), "duplicate artifact filename"
    for n in names:
        assert "SUB" not in n, f"{n} leaks identity into the filename"
        assert not n.startswith("T0"), f"{n} leaks a trial number"


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
