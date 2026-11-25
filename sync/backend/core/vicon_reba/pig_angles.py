"""Plug-in Gait (PiG) modeled-output extraction from Vicon C3D.

Why this exists
---------------
Nexus, when a PiG model is run, writes *validated* joint angles and full
inverse dynamics into the C3D alongside the raw markers. The rest of this
package computes angles from marker geometry (see ``angles.py``); those are
best-effort. Where PiG angles exist they are the biomechanical gold standard
and should be used as (a) ground truth to validate the marker-derived angles,
and (b) the actual leg angles feeding REBA.

Scope / caveats
---------------
- The captured model is **PiG lower body only**: Hip / Knee / Ankle angles for
  both sides. There are NO modeled trunk / neck / upper-limb angles, so REBA's
  Group-A trunk/neck and all of Group-B still come from ``angles.py``.
- PiG angle points are 3-vectors ``[flexion, ab/adduction, rotation]`` in
  degrees (component 0 = sagittal flexion, the only one REBA legs needs).
- Load estimation uses the two force plates + the subject Bodymass stored in the
  C3D PROCESSING group. It is a coarse whole-body estimate, not a per-hand load.

Everything here is READ-ONLY on the C3D. It never writes to the input folder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# PiG angle point labels (component 0 = sagittal flexion, degrees)
_PIG_ANGLE_LABELS = {
    "hip_l":   "LHipAngles",   "hip_r":   "RHipAngles",
    "knee_l":  "LKneeAngles",  "knee_r":  "RKneeAngles",
    "ankle_l": "LAnkleAngles", "ankle_r": "RAnkleAngles",
}

G = 9.80665  # m/s^2


def _read_points(c3d_path: Path):
    import ezc3d
    c = ezc3d.c3d(str(c3d_path))
    pts = c["data"]["points"]                                   # (4, n, N)
    labels = [l.strip() for l in c["parameters"]["POINT"]["LABELS"]["value"]]
    rate = float(c["parameters"]["POINT"]["RATE"]["value"][0])
    try:
        start = int(c["parameters"]["TRIAL"]["ACTUAL_START_FIELD"]["value"][0])
    except Exception:
        start = 1
    return c, pts, labels, rate, start


def has_pig_angles(c3d_path: Path) -> bool:
    """True if at least the knee angle points are present (cheap header check)."""
    try:
        _, _, labels, _, _ = _read_points(Path(c3d_path))
    except Exception:
        return False
    return "LKneeAngles" in labels and "RKneeAngles" in labels


def extract_pig_angles(c3d_path: Path) -> dict:
    """Extract PiG lower-limb sagittal flexion angles from a C3D.

    Returns:
        dict with:
          "available": bool
          "frames":  (N,) int64 Vicon frame numbers
          "time_s":  (N,) float64 seconds from ACTUAL_START_FIELD
          "rate":    float Hz
          plus one (N,) float64 array per key in _PIG_ANGLE_LABELS
          (NaN where the angle point was occluded). Missing angle points are
          simply absent from the dict.
        If no PiG angles exist, {"available": False}.
    """
    c3d_path = Path(c3d_path)
    try:
        _, pts, labels, rate, start = _read_points(c3d_path)
    except Exception:
        return {"available": False}

    if "LKneeAngles" not in labels:
        return {"available": False}

    n_frames = pts.shape[2]
    frames = np.arange(start, start + n_frames, dtype=np.int64)
    out: dict = {
        "available": True,
        "frames": frames,
        "time_s": (frames - start) / rate,
        "rate": rate,
    }
    for key, label in _PIG_ANGLE_LABELS.items():
        if label not in labels:
            continue
        i = labels.index(label)
        flex = pts[0, i, :].astype(np.float64)      # component 0 = flexion (deg)
        residual = pts[3, i, :]
        flex[(residual < 0) | ~np.isfinite(flex)] = np.nan
        out[key] = flex
    return out


def knee_flexion_at_times(pig: dict, query_time_s: np.ndarray) -> np.ndarray | None:
    """Interpolate max(L,R) PiG knee flexion onto arbitrary query times.

    Used to align Vicon-rate PiG angles to the per-video-frame timestamps in a
    synced.csv (its ``vicon_time_s`` column). Returns None if PiG unavailable.

    NaN-safe: a query time landing inside an occluded run yields NaN for that
    side; the max of two NaNs is NaN, so the frame stays invalid downstream.
    """
    if not pig.get("available") or "knee_l" not in pig or "knee_r" not in pig:
        return None
    t = pig["time_s"]

    def _interp(vals):
        # np.interp ignores NaN structure, so mask first: interpolate only over
        # finite samples, then re-insert NaN where the nearest sample is NaN.
        finite = np.isfinite(vals)
        if finite.sum() < 2:
            return np.full_like(query_time_s, np.nan, dtype=np.float64)
        q = np.interp(query_time_s, t[finite], vals[finite],
                      left=np.nan, right=np.nan)
        # invalidate queries whose bracketing samples were NaN
        idx = np.clip(np.searchsorted(t, query_time_s), 1, len(t) - 1)
        bad = ~finite[idx] | ~finite[idx - 1]
        q[bad] = np.nan
        return q

    return np.fmax(_interp(pig["knee_l"]), _interp(pig["knee_r"]))


def crosscheck_marker_vs_pig(pig: dict,
                             marker_knee_flex_deg: np.ndarray,
                             marker_time_s: np.ndarray) -> dict:
    """Compare marker-derived knee flexion against PiG (the gold standard).

    Args:
        pig: output of extract_pig_angles().
        marker_knee_flex_deg: (M,) marker-derived max-knee flexion per frame.
        marker_time_s: (M,) matching timestamps (same clock as pig["time_s"]).

    Returns:
        {"available": bool, "mae_deg", "rmse_deg", "bias_deg", "n": int}
        over frames finite in both. bias = marker - pig (positive = marker reads
        higher, which is what the review measured, ~5 deg).
    """
    pig_at = knee_flexion_at_times(pig, np.asarray(marker_time_s, dtype=np.float64))
    if pig_at is None:
        return {"available": False}
    m = np.asarray(marker_knee_flex_deg, dtype=np.float64)
    valid = np.isfinite(pig_at) & np.isfinite(m)
    if valid.sum() == 0:
        return {"available": False}
    diff = m[valid] - pig_at[valid]
    return {
        "available": True,
        "n": int(valid.sum()),
        "mae_deg":  round(float(np.mean(np.abs(diff))), 2),
        "rmse_deg": round(float(np.sqrt(np.mean(diff ** 2))), 2),
        "bias_deg": round(float(np.mean(diff)), 2),
    }


def estimate_load_score(c3d_path: Path) -> dict:
    """Estimate a REBA load score (0/1/2) from force plates + Bodymass.

    REBA load thresholds: 0 = <5 kg, 1 = 5-10 kg, 2 = >10 kg.

    Method: total vertical support = sum of |Fz| across force plates. The
    subject's own weight (Bodymass * g) is subtracted from a high percentile of
    that signal; the surplus is attributed to a handled/carried load.

    Note: coarse whole-body heuristic — 95th-percentile surplus, not a
    per-hand load. Force plates only register while the subject stands on them,
    so a load lifted off-plate is invisible. Upgrade path: instrument the load
    or log its nominal mass at collection time (see load metadata in the REBA
    router). Returns available=False when force/mass data is missing so the
    caller can fall back to a logged/manual load.

    Returns:
        {"available": bool, "load_score": int, "load_kg_est": float,
         "bodymass_kg": float, "peak_grf_n": float}
    """
    c3d_path = Path(c3d_path)
    try:
        import ezc3d
        c = ezc3d.c3d(str(c3d_path))
    except Exception:
        return {"available": False}

    try:
        analog = c["data"]["analogs"][0]                        # (n_ch, M)
        alabels = [l.strip() for l in c["parameters"]["ANALOG"]["LABELS"]["value"]]
    except Exception:
        return {"available": False}

    fz_idx = [i for i, l in enumerate(alabels) if l.upper().replace(".", "").startswith("FORCEFZ")
              or l.upper().endswith("FZ") or ".FZ" in l.upper()]
    if not fz_idx:
        return {"available": False}

    total_vertical = np.abs(analog[fz_idx, :]).sum(axis=0)      # N, per analog sample
    total_vertical = total_vertical[np.isfinite(total_vertical)]
    if total_vertical.size == 0 or total_vertical.max() < 1.0:
        return {"available": False}

    try:
        bodymass = float(c["parameters"]["PROCESSING"]["Bodymass"]["value"][0])
    except Exception:
        return {"available": False}

    peak_grf = float(np.percentile(total_vertical, 95))
    surplus_n = max(0.0, peak_grf - bodymass * G)
    load_kg = surplus_n / G
    load_score = 0 if load_kg < 5.0 else (1 if load_kg <= 10.0 else 2)
    return {
        "available": True,
        "load_score": int(load_score),
        "load_kg_est": round(load_kg, 1),
        "bodymass_kg": round(bodymass, 1),
        "peak_grf_n": round(peak_grf, 1),
    }


def _demo() -> None:
    """Self-check against a real C3D if present, else a synthetic sanity check."""
    sample = Path("sample.c3d")  # point this at a local C3D to run the real-file check
    if sample.exists():
        pig = extract_pig_angles(sample)
        assert pig["available"], "PiG angles should be detected in the sample C3D"
        assert "knee_l" in pig and "knee_r" in pig
        assert np.isfinite(pig["knee_l"]).any()
        # interp at the trial's own times must round-trip within tolerance
        q = knee_flexion_at_times(pig, pig["time_s"][:50])
        assert q is not None and np.isfinite(q).any()
        load = estimate_load_score(sample)
        assert load["available"] and load["load_score"] in (0, 1, 2)
        print(f"[pig_angles demo] knee_l mean={np.nanmean(pig['knee_l']):.1f} deg  "
              f"load={load['load_kg_est']} kg -> score {load['load_score']}  "
              f"(bodymass {load['bodymass_kg']} kg)")
    else:
        # Synthetic: two finite knee series, interp identity within tolerance
        pig = {"available": True, "time_s": np.arange(10) / 100.0,
               "rate": 100.0, "knee_l": np.linspace(0, 90, 10),
               "knee_r": np.linspace(0, 45, 10)}
        q = knee_flexion_at_times(pig, np.array([0.0, 0.09]))
        assert np.isclose(q[0], 0.0) and np.isclose(q[1], 90.0), q
        print("[pig_angles demo] synthetic check passed")
    print("OK")


if __name__ == "__main__":
    _demo()
