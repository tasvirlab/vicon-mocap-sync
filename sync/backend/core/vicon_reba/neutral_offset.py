"""Per-subject neutral-reference offsets for the anatomical angle channels.

WHY THIS EXISTS
---------------
`neck_flex_deg` is built from the C7 -> HD vector, where HD is the centroid of
the four head markers. That centroid lies ANTERIOR to C7 by anatomy and marker
placement, so the vector is tilted forward even when the participant is standing
neutrally. Measured in each subject's still static-calibration trial; it is a stable
bias, not noise.

Zero in that channel is therefore NOT anatomical neutral. Both the REBA neck
category (20 deg) and the duration layer's neck entry threshold (20 deg) are
applied to a biased signal, so the bias propagates into Channel 1 severity and
Channel 2 exposure alike. Expressing each channel relative to the participant's
own measured neutral pose is the standard correction and is exactly what the
static calibration trial is for.

WHICH CHANNELS MAY BE CORRECTED — AND WHICH MUST NOT
----------------------------------------------------
A channel may be offset only if the static calibration pose IS that channel's
anatomical neutral. Standing upright is neutral for the trunk and neck, so those
are corrected. It is NOT neutral for the arms: in the Plug-in Gait static capture
the arms are deliberately held away from the body for marker visibility, which is
why they read 26-42 deg there. Subtracting that as an "offset" would make
genuinely hanging arms report about -30 deg. The arms are therefore excluded by
name, not by omission, so nobody re-adds them by accident.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

# Standing upright is the anatomical neutral for these, so the static pose is a
# valid zero reference.
OFFSETTABLE_CHANNELS = (
    "trunk_sagittal_deg",
    "trunk_lateral_deg",
    "trunk_twist_deg",
    "neck_flex_deg",
)

# Excluded BY NAME with the reason attached — see the module docstring.
EXCLUDED_CHANNELS = {
    "upper_arm_r_deg": "PiG static pose holds the arms abducted; that is the pose, not an offset",
    "upper_arm_l_deg": "PiG static pose holds the arms abducted; that is the pose, not an offset",
    "upper_arm_r_abd_deg": "abduction is defined from the same abducted static pose",
    "upper_arm_l_abd_deg": "abduction is defined from the same abducted static pose",
}

_SUBJECT_RE = re.compile(r"(SUB\d{2,})", re.IGNORECASE)


def parse_subject(task_id: str) -> str | None:
    """'SUB01_lifting_T01' -> 'SUB01'. None when no subject token is present."""
    m = _SUBJECT_RE.search(task_id or "")
    return m.group(1).upper() if m else None


def estimate_offsets(angle_series_by_channel: dict[str, list[np.ndarray]]) -> dict:
    """Neutral offset per channel = MEDIAN across the pooled static samples.

    Median, not mean: a static trial can contain a few seconds of the
    participant settling into position, and the median ignores that tail instead
    of being dragged by it.
    """
    out: dict[str, dict] = {}
    for ch, series_list in angle_series_by_channel.items():
        if ch in EXCLUDED_CHANNELS or ch not in OFFSETTABLE_CHANNELS:
            continue
        pooled = []
        for s in series_list:
            v = np.asarray(s, dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                pooled.append(v)
        if not pooled:
            continue
        allv = np.concatenate(pooled)
        out[ch] = {
            "offset_deg": round(float(np.median(allv)), 3),
            "sd_deg": round(float(np.std(allv)), 3),
            "n_trials": len(pooled),
            "n_frames": int(allv.size),
        }
    return out


def apply_offsets(angles: dict[str, np.ndarray],
                  offsets: dict[str, float]) -> tuple[dict[str, np.ndarray], dict]:
    """Return (corrected angles, record). Non-offsettable channels pass through.

    The record names every channel that was corrected and every one that was
    deliberately skipped, so a figure or report can state exactly what was
    done rather than leaving the reader to infer it.
    """
    corrected = dict(angles)
    applied: dict[str, float] = {}
    skipped: dict[str, str] = {}

    for ch in angles:
        if ch in EXCLUDED_CHANNELS:
            skipped[ch] = EXCLUDED_CHANNELS[ch]
            continue
        if ch not in OFFSETTABLE_CHANNELS:
            continue
        off = offsets.get(ch)
        if off is None:
            skipped[ch] = "no neutral offset available for this subject"
            continue
        corrected[ch] = np.asarray(angles[ch], dtype=float) - float(off)
        applied[ch] = round(float(off), 3)

    return corrected, {
        "applied": applied,
        "skipped": skipped,
        "any_applied": bool(applied),
    }


def load_offsets(path, subject: str | None) -> dict[str, float]:
    """Read a frozen offsets file -> {channel: offset_deg} for one subject.

    Returns {} when the file, or that subject's entry, is absent — the caller
    then reports angles as RAW rather than silently applying someone else's
    neutral pose.
    """
    p = Path(path)
    if not p.exists() or not subject:
        return {}
    doc = json.loads(p.read_text(encoding="utf-8"))
    subj = (doc.get("subjects") or {}).get(subject.upper())
    if not subj:
        return {}
    return {ch: v["offset_deg"] for ch, v in subj.items()
            if isinstance(v, dict) and "offset_deg" in v}


# ── Self-check ───────────────────────────────────────────────────────────────

def _demo() -> None:
    """Assert-based self-check."""
    assert parse_subject("SUB01_lifting_T01") == "SUB01"
    assert parse_subject("angles.csv") is None

    # A neck sitting at +27 deg neutral, plus a real 40 deg flexion event.
    neck = np.concatenate([np.full(100, 27.0), np.full(100, 67.0)])
    est = estimate_offsets({"neck_flex_deg": [np.full(300, 27.0)]})
    assert abs(est["neck_flex_deg"]["offset_deg"] - 27.0) < 1e-6, est

    ang = {"neck_flex_deg": neck,
           "upper_arm_r_deg": np.full(200, 30.0),
           "wrist_r_deg": np.full(200, 5.0)}
    corr, rec = apply_offsets(ang, {"neck_flex_deg": 27.0, "upper_arm_r_deg": 30.0})

    # Neutral now reads 0 and the real event keeps its true 40 deg magnitude.
    assert abs(corr["neck_flex_deg"][0]) < 1e-6
    assert abs(corr["neck_flex_deg"][-1] - 40.0) < 1e-6
    assert rec["applied"] == {"neck_flex_deg": 27.0}

    # THE ARM GUARD: an arm offset must be refused even when one is supplied,
    # because the static pose holds the arms abducted. Correcting it would make
    # hanging arms read about -30 deg.
    assert np.allclose(corr["upper_arm_r_deg"], 30.0), "arm must NOT be offset"
    assert "upper_arm_r_deg" in rec["skipped"]
    assert "abducted" in rec["skipped"]["upper_arm_r_deg"]

    # A channel outside the offsettable set passes through untouched and unlisted.
    assert np.allclose(corr["wrist_r_deg"], 5.0)
    assert "wrist_r_deg" not in rec["applied"]

    # estimate_offsets must never emit an arm offset in the first place.
    est2 = estimate_offsets({"upper_arm_r_deg": [np.full(50, 30.0)],
                             "neck_flex_deg": [np.full(50, 27.0)]})
    assert "upper_arm_r_deg" not in est2 and "neck_flex_deg" in est2

    # Missing subject / missing file -> {} so angles stay RAW, never someone
    # else's neutral pose.
    assert load_offsets("/nonexistent/offsets.json", "SUB01") == {}
    _, rec2 = apply_offsets(ang, {})
    assert rec2["any_applied"] is False
    assert "no neutral offset available" in rec2["skipped"]["neck_flex_deg"]

    print("neutral_offset self-check OK")


if __name__ == "__main__":
    _demo()
