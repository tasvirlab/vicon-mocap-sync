"""Derive and FREEZE the low-motion thresholds (eps_omega, eps_theta) used by
the static-hold branch of the Within-Task High-Risk Posture-Duration Assessment
Layer.

Run once, from the training/validation participants only, BEFORE any test-set
evaluation. Until this has been run, every bout reports
`is_static_hold: None` ("not evaluated") — the layer never substitutes a default.

    python -m tools.derive_static_thresholds --train SUB01,SUB02,SUB03

Requires no video, no synchronisation and no camera calibration: the derivation
path is C3D -> 21-joint positions -> anatomical angles -> per-trial
median|omega| and SD(theta) -> percentile across trials.

LEAKAGE: --train is mandatory and has no default. A test-pool subject listed
here silently contaminates the frozen thresholds, and nothing downstream can
detect it. The chosen pool is recorded in the output file so the choice stays
auditable.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.c3d_pipeline import c3d_to_clean_csv                      # noqa: E402
from core.vicon_reba import angles as vr_angles                     # noqa: E402
from core.vicon_reba import load as vr_load                         # noqa: E402
from core.vicon_reba import neutral_offset as vr_off                # noqa: E402
from core.vicon_reba import posture_duration as vr_dur              # noqa: E402
from utils import app_config                                        # noqa: E402

# Segment -> the signed angle channel whose stillness defines that segment's eps.
# Mirrors channels._DURATION_SEGMENTS so the thresholds match what they gate.
SEGMENT_CHANNELS = {
    "trunk":     ["trunk_sagittal_deg"],
    "neck":      ["neck_flex_deg"],
    "upper_arm": ["upper_arm_r_deg", "upper_arm_l_deg"],
}

DEFAULT_RAW = app_config.get_raw_vicon_dir() or Path("raw/Vicon")
DEFAULT_OUT = app_config.get_outputs_dir() / "_calibration" / "static_thresholds.json"
DEFAULT_OFFSETS_OUT = app_config.get_outputs_dir() / "_calibration" / "neutral_offsets.json"


def find_static_c3d(raw_root: Path, subject: str) -> list[Path]:
    """Static calibration C3D files for one subject.

    Naming varies across subjects ("Task8 static callibration 01.c3d",
    "Task08_static calibration01.c3d", "Static01.c3d"), so match on tokens
    rather than an exact pattern. Files whose stem is only "static"/"staticNN"
    are the Plug-in Gait model-calibration captures; they are included because
    they are also genuine still-standing holds, but are tagged in the report so
    the two families can be told apart afterwards.
    """
    sub_dir = raw_root / subject
    if not sub_dir.is_dir():
        return []
    out = []
    for p in sorted(sub_dir.glob("*.c3d")):
        stem = p.stem.lower().replace("_", " ")
        if "static" in stem or "callibration" in stem:
            out.append(p)
    return out


def angles_from_c3d(c3d_path: Path, work_dir: Path) -> tuple[dict, float]:
    """C3D -> {angle_name: series}, plus the capture rate."""
    report = c3d_to_clean_csv(c3d_path, work_dir, out_stem=c3d_path.stem)
    csv_path = work_dir / f"{c3d_path.stem}.csv"
    if not csv_path.exists():                       # writer may pick its own name
        cands = sorted(work_dir.glob("*.csv"))
        if not cands:
            raise FileNotFoundError(f"no CSV produced for {c3d_path.name}")
        csv_path = cands[-1]
    joints, df = vr_load.load(csv_path)
    fps = float(report.get("frame_rate") or 100.0)
    return vr_angles.compute_reba_angles(joints), fps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True,
                    help="COMMA-SEPARATED training/validation subjects, e.g. "
                         "SUB01,SUB02,SUB03. Mandatory: no default, to keep the "
                         "leakage decision explicit and auditable.")
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW,
                    help=f"raw Vicon root containing SUBxx/ dirs (default {DEFAULT_RAW})")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"frozen threshold file to write (default {DEFAULT_OUT})")
    ap.add_argument("--percentile", type=float, default=95.0,
                    help="percentile of the static distribution kept (default 95)")
    ap.add_argument("--offsets-out", type=Path, default=DEFAULT_OFFSETS_OUT,
                    help=f"frozen neutral-offset file (default {DEFAULT_OFFSETS_OUT})")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing frozen file (refused otherwise — "
                         "re-freezing after test evaluation invalidates the split)")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print, write nothing")
    args = ap.parse_args()

    subjects = [s.strip().upper() for s in args.train.split(",") if s.strip()]
    if not subjects:
        print("ERROR: --train listed no subjects", file=sys.stderr)
        return 2

    if args.out.exists() and not args.force and not args.dry_run:
        print(f"REFUSED: {args.out} already exists.\n"
              f"  Thresholds are meant to be frozen ONCE, before test evaluation.\n"
              f"  Re-run with --force only if you are deliberately re-freezing and\n"
              f"  have not yet reported any test-set result.", file=sys.stderr)
        return 3

    print(f"Train/validation pool : {', '.join(subjects)}")
    print(f"Raw Vicon root        : {args.raw}")
    print(f"Percentile            : {args.percentile}")
    print()

    per_segment: dict[str, list[np.ndarray]] = {k: [] for k in SEGMENT_CHANNELS}
    # Offsets are PER SUBJECT (each participant has their own neutral pose),
    # unlike eps which is pooled across the whole train pool.
    per_subject_angles: dict[str, dict[str, list]] = {}
    sources: list[dict] = []
    fps_seen: list[float] = []

    with tempfile.TemporaryDirectory() as td:
        for subject in subjects:
            files = find_static_c3d(args.raw, subject)
            if not files:
                print(f"  {subject}: NO static calibration C3D found — skipped")
                continue
            for f in files:
                work = Path(td) / f"{subject}_{f.stem}".replace(" ", "_")
                work.mkdir(parents=True, exist_ok=True)
                try:
                    ang, fps = angles_from_c3d(f, work)
                except Exception as e:                    # noqa: BLE001
                    print(f"  {subject}: {f.name} FAILED ({type(e).__name__}: {e})")
                    continue
                fps_seen.append(fps)
                n_used = 0
                for seg, channels in SEGMENT_CHANNELS.items():
                    for ch in channels:
                        s = ang.get(ch)
                        if s is None:
                            continue
                        s = np.asarray(s, dtype=float)
                        if np.isfinite(s).sum() < 3:
                            continue
                        per_segment[seg].append(s)
                        n_used += 1
                # Offsets come ONLY from the PiG static family: those are the
                # genuinely still captures. The Task8 "static calibration" files
                # were measured to contain real movement (docs 4.1.2).
                is_pig = f.stem.lower().replace("_", " ").strip().startswith("static")
                if is_pig:
                    bucket = per_subject_angles.setdefault(subject, {})
                    for ch in vr_off.OFFSETTABLE_CHANNELS:
                        if ch in ang:
                            bucket.setdefault(ch, []).append(np.asarray(ang[ch], float))
                pig = "pig-model" if is_pig \
                    else "protocol-static"
                sources.append({"subject": subject, "file": f.name,
                                "kind": pig, "fps": round(fps, 4),
                                "n_frames": int(len(next(iter(ang.values())))),
                                "channels_used": n_used})
                print(f"  {subject}: {f.name}  ({pig}, {fps:.1f} Hz, "
                      f"{len(next(iter(ang.values())))} frames)")

    if not any(per_segment.values()):
        print("\nERROR: no usable static trials — nothing to estimate.", file=sys.stderr)
        return 4

    fps = float(np.median(fps_seen)) if fps_seen else 100.0
    doc = vr_dur.estimate_static_thresholds(per_segment, fps,
                                            percentile=args.percentile,
                                            cutoff_hz=None)
    doc["train_pool"] = subjects
    doc["sources"] = sources
    doc["note"] = (
        "eps_omega / eps_theta for the static-hold branch. Estimated ONLY from the "
        "listed training/validation participants' static calibration trials, and "
        "frozen before test-set evaluation. cutoff_hz=None because c3d_pipeline "
        "already 6 Hz zero-phase filters the markers at the native rate. Vary the "
        "percentile in the sensitivity analysis and report it.")

    print("\nDerived thresholds:")
    for seg, v in doc["segments"].items():
        print(f"  {seg:10s} eps_omega = {v['eps_omega_deg_s']:7.3f} deg/s   "
              f"eps_theta = {v['eps_theta_deg']:6.3f} deg   "
              f"(n_trials={v['n_trials']}, n_frames={v['n_frames']})")

    # ── Per-subject neutral offsets ─────────────────────────────────────────
    offsets_doc = {"subjects": {}, "note": (
        "Per-subject neutral-reference offsets, measured as the MEDIAN angle in "
        "that participant's still PiG static capture, subtracted from task angles "
        "so 0 deg means anatomical neutral. Arms are deliberately EXCLUDED: the "
        "PiG static pose holds them abducted, so that reading is the pose, not an "
        "offset. See core/vicon_reba/neutral_offset.py.")}
    for subject, chans in per_subject_angles.items():
        est = vr_off.estimate_offsets(chans)
        if est:
            offsets_doc["subjects"][subject] = est

    if offsets_doc["subjects"]:
        print("\nNeutral offsets (subtracted so neutral reads 0 deg):")
        for subject, chans in sorted(offsets_doc["subjects"].items()):
            bits = "  ".join(f"{c.replace('_deg','')}={v['offset_deg']:+.1f}"
                             for c, v in sorted(chans.items()))
            print(f"  {subject}: {bits}")
    else:
        print("\nNo PiG static captures found — no neutral offsets derived.")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\nFrozen -> {args.out}")
    if offsets_doc["subjects"]:
        args.offsets_out.parent.mkdir(parents=True, exist_ok=True)
        args.offsets_out.write_text(json.dumps(offsets_doc, indent=2), encoding="utf-8")
        print(f"Frozen -> {args.offsets_out}")
    print("The static branch is now active. Re-run the Analyse stage to see "
          "is_static_hold populated instead of 'not evaluated'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
