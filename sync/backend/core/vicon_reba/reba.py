"""REBA (Rapid Entire Body Assessment) scoring from joint angles.

Reference: Hignett, S. & McAtamney, L. (2000). Rapid Entire Body Assessment
(REBA). Applied Ergonomics, 31(2), 201-205.

Scoring overview
────────────────
Group A  trunk + neck + legs → Table A → score_a  → + activity → Score A
Group B  upper arm + lower arm + wrist → Table B → score_b  → + coupling → Score B
Table C  Score A × Score B → final REBA score (1–15)

Risk categories
───────────────
  1        : Negligible risk
  2–3      : Low risk       (change may be needed)
  4–7      : Medium risk    (further investigation, change soon)
  8–10     : High risk      (investigation and change required)
  11–15    : Very high risk (immediate change required)

Automated pipeline assumptions
────────────────────────────────
  - Activity score (+1 by default) — construction tasks involve repeated postures.
  - Coupling score (0 by default = good grip) — override with --coupling 1 (fair) or 2 (poor).
  - Load score (0 by default = <5 kg) — override with --load 1 (5-10 kg) or 2 (>10 kg).
  - Bilateral weight bearing assumed unless `unilateral=True` (affects leg base score).
  - Trunk twist modifier: if |shoulder-hip bar angle| > TWIST_THRESH_DEG.
  - Trunk lateral modifier: if |lateral component| > LATERAL_THRESH_DEG.
    (twist/lateral are SIGNED — see angles.py; compared as abs() here.)
  - Neck lateral modifier: if neck lateral angle > NECK_LATERAL_THRESH_DEG.
  - Upper arm abduction modifier: if abduction > ABD_THRESH_DEG.
  - Wrist: base score ONLY (1-2). REBA's +1 wrist modifier is not measurable
    from this marker set and is deliberately not applied — see below.

Wrist sub-score is low-fidelity and currently saturated
──────────────────────────────────────────────────────
Two independent problems, one fixed here and one still open:

  1. FIXED: the +1 modifier was re-applying wrist_base()'s own 15 deg
     threshold, double-counting one measurement and making wrist_adj take only
     {1, 3} — the middle category was unreachable. REBA's modifier is for a
     different condition ("bent from midline or twisted"), which a single hand
     marker (RFIN) cannot separate from flexion. It is now omitted.

  2. OPEN: wrist deviation carries a systematic offset. On SUB01/lifting,
     wrist_r_deg never drops below 16.6 deg, so it never reaches category 1 even
     at the most neutral instant, and wrist_adj was a CONSTANT (std 0 across all
     822 frames) — a fixed inflation of Score B carrying zero information. The
     cause is marker geometry, not posture: RFIN sits on the finger and the
     wrist JC on the distal radius, so the elbow-wrist-hand chain is not
     collinear at anatomical neutral. Fixing this needs a per-subject neutral
     offset measured from the session's static standing trial and subtracted
     before scoring. That calibration is NOT implemented; until it is, treat the
     wrist sub-score as a constant offset rather than a measurement, and do not
     report it as a per-frame risk finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Thresholds for auto-detected modifiers ───────────────────────────────────
TWIST_THRESH_DEG     = 10.0   # trunk twist → +1 trunk
LATERAL_THRESH_DEG   = 10.0   # trunk lateral bend → +1 trunk
NECK_LATERAL_THRESH  = 15.0   # neck lateral → +1 neck
ABD_THRESH_DEG       = 20.0   # shoulder abduction → +1 upper arm
# No WRIST_DEV_THRESH_DEG: REBA's +1 wrist modifier ("bent from midline or
# twisted") is a different condition from the 15 deg flexion threshold already
# applied by wrist_base(), and it is not measurable from a single hand marker.
# It used to be re-applied at the same 15 deg, double-counting one measurement.
# See the wrist note in _group_b_side().

# ── REBA lookup tables (Hignett & McAtamney 2000) ───────────────────────────

# TABLE_A[trunk-1][neck-1][legs-1] → raw score_a (before activity modifier)
# Trunk: 1-5  |  Neck: 1-3 (base 1-2 + optional +1 modifier)  |  Legs: 1-4
TABLE_A: list[list[list[int]]] = [
    [[1, 2, 3, 4], [1, 2, 3, 4], [3, 3, 5, 6]],   # trunk=1
    [[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],   # trunk=2
    [[3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],   # trunk=3
    [[4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],   # trunk=4
    [[5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 9]],   # trunk=5
]

# TABLE_B[ua_adj-1][la-1][wr_adj-1] → raw score_b (before coupling modifier)
# Upper arm (adjusted): 1-6  |  Lower arm: 1-2  |  Wrist (adjusted): 1-3
TABLE_B: list[list[list[int]]] = [
    [[1, 2, 2], [1, 2, 3]],   # ua=1
    [[1, 2, 3], [2, 3, 4]],   # ua=2
    [[3, 4, 5], [4, 5, 5]],   # ua=3
    [[4, 5, 5], [5, 6, 7]],   # ua=4
    [[6, 7, 8], [7, 8, 8]],   # ua=5
    [[7, 8, 8], [8, 9, 9]],   # ua=6
]

# TABLE_C[score_a-1][score_b-1] → final REBA score (1–15)
# Score A: 1-12  |  Score B: 1-12
TABLE_C: list[list[int]] = [
    [ 1,  1,  1,  2,  3,  3,  4,  5,  6,  7,  7,  7],  # sa=1
    [ 1,  2,  2,  3,  4,  4,  5,  6,  6,  7,  7,  8],  # sa=2
    [ 2,  3,  3,  3,  4,  5,  6,  7,  7,  8,  8,  8],  # sa=3
    [ 3,  4,  4,  4,  5,  6,  7,  8,  8,  9,  9,  9],  # sa=4
    [ 4,  4,  4,  5,  6,  7,  8,  8,  9,  9,  9,  9],  # sa=5
    [ 6,  6,  6,  7,  8,  8,  9,  9, 10, 10, 10, 10],  # sa=6
    [ 7,  7,  7,  8,  9,  9,  9, 10, 10, 11, 11, 11],  # sa=7
    [ 8,  8,  8,  9, 10, 10, 10, 10, 10, 11, 11, 11],  # sa=8
    [ 9,  9,  9, 10, 10, 10, 11, 11, 11, 12, 12, 12],  # sa=9
    [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],  # sa=10
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],  # sa=11
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],  # sa=12
]

# ── Per-joint base scores ────────────────────────────────────────────────────

def trunk_base(deg: float) -> int | None:
    """REBA trunk position score (1-4) from SIGNED sagittal flexion/extension.

    Args:
        deg: signed degrees — positive = forward flexion, negative = extension.
             (Must be trunk_sagittal_deg, NOT trunk_total_deg — the latter
             includes lateral bend, which is scored separately as a modifier
             and must not also inflate the base position score.)

    Returns:
        1-4, or None if `deg` is NaN/non-finite (occluded marker — caller
        must treat the frame as invalid, not silently score it as severe).

    Hignett & McAtamney (2000) trunk table: extension has no >60° (score 4)
    category — any extension beyond 20° is capped at score 3.
    """
    if not np.isfinite(deg):
        return None
    if deg > 60:
        return 4            # severe flexion
    if deg > 20:
        return 3            # moderate flexion
    if deg > 0:
        return 2            # mild flexion
    if deg == 0:
        return 1            # absolutely erect
    if deg >= -20:
        return 2            # mild/moderate extension
    return 3                # severe extension (capped — standard defines no score-4 extension)


def neck_base(flex_deg: float) -> int | None:
    """REBA neck position score (1-2) from SIGNED flexion relative to trunk.

    Args:
        flex_deg: signed degrees — positive = flexion, negative = extension.

    Returns:
        1 or 2, or None if non-finite. Any extension (deg < 0) scores 2
        regardless of magnitude, per the standard.
    """
    if not np.isfinite(flex_deg):
        return None
    if flex_deg < 0:
        return 2            # any extension
    return 1 if flex_deg <= 20 else 2


def legs_base(knee_max_deg: float, unilateral: bool = False) -> int | None:
    """REBA legs position score (1-2 + knee modifier) from max knee flexion."""
    if not np.isfinite(knee_max_deg):
        return None
    score = 2 if unilateral else 1
    if knee_max_deg > 60:
        score += 2
    elif knee_max_deg > 30:
        score += 1
    return min(score, 4)


def upper_arm_base(elev_deg: float) -> int | None:
    """REBA upper arm position score (1-4) from 3D elevation from neutral hang."""
    if not np.isfinite(elev_deg):
        return None
    if elev_deg <= 20:
        return 1
    elif elev_deg <= 45:
        return 2
    elif elev_deg <= 90:
        return 3
    else:
        return 4


def lower_arm_base(flexion_deg: float) -> int | None:
    """REBA lower arm score (1-2) from elbow flexion."""
    if not np.isfinite(flexion_deg):
        return None
    return 1 if 60 <= flexion_deg <= 100 else 2


def wrist_base(deviation_deg: float) -> int | None:
    """REBA wrist score (1-2) from deviation from straight."""
    if not np.isfinite(deviation_deg):
        return None
    return 1 if deviation_deg <= 15 else 2


# ── Single-frame full REBA score ─────────────────────────────────────────────

def score_frame(
    trunk_sagittal: float,
    trunk_twist:    float,
    trunk_lateral:  float,
    neck_flex:      float,
    neck_lateral:   float,
    knee_max:       float,
    upper_arm_r:    float,
    upper_arm_l:    float,
    upper_arm_r_abd: float,
    upper_arm_l_abd: float,
    lower_arm_r:    float,
    lower_arm_l:    float,
    wrist_r:        float,
    wrist_l:        float,
    *,
    load:      int = 0,
    coupling:  int = 0,
    activity:  int = 1,
    unilateral: bool = False,
) -> dict:
    """Compute REBA score for a single frame and return all component scores.

    Args:
        trunk_sagittal: SIGNED trunk flexion/extension angle (degrees) —
            positive = forward flexion, negative = extension. Must be the
            sagittal-only angle, NOT the unsigned total 3D inclination —
            using the total would double-count lateral bend, which is
            already scored separately via the `trunk_lateral` modifier.
        trunk_twist:  SIGNED shoulder-hip bar angle in horizontal plane
            (degrees). Compared as abs() — REBA's modifier is severity-only.
        trunk_lateral: SIGNED lateral trunk bend (degrees), + = subject's
            right. Compared as abs() — see above.
        neck_flex:    SIGNED neck flexion relative to the trunk (degrees) —
            positive = flexion, negative = extension.
        neck_lateral: Neck lateral bending magnitude (degrees).
        knee_max:     Maximum knee flexion, both sides (degrees).
        upper_arm_r/l: 3D elevation from neutral hang (degrees), right/left.
        upper_arm_r/l_abd: Shoulder abduction (degrees), right/left.
        lower_arm_r/l: Elbow flexion (degrees), right/left.
        wrist_r/l:    Wrist deviation (degrees), right/left.
        load:         0=<5 kg, 1=5–10 kg, 2=>10 kg (or +1 for sudden force).
        coupling:     0=good, 1=fair, 2=poor/unacceptable.
        activity:     0=static, 1=dynamic/repeated (default=1 for construction).
        unilateral:   True if worker is bearing weight on one leg only.

    Returns:
        dict with keys: trunk_score, neck_score, legs_score, score_a,
        upper_arm_adj, lower_arm_score, wrist_adj, score_b, reba_total,
        risk_level, and all raw subscores.

        If ANY input is non-finite (occluded marker → NaN angle), the frame
        is invalid: all score fields are None, reba_total is NaN, and
        risk_level is "invalid". Callers MUST exclude these frames from
        statistics/peak-detection rather than letting them fall through to a
        scored value — occlusion must never be silently treated as a real
        (and especially not a worst-case) posture.
    """
    _inputs = (trunk_sagittal, trunk_twist, trunk_lateral, neck_flex, neck_lateral,
               knee_max, upper_arm_r, upper_arm_l, upper_arm_r_abd, upper_arm_l_abd,
               lower_arm_r, lower_arm_l, wrist_r, wrist_l)
    if not all(np.isfinite(v) for v in _inputs):
        return {
            "trunk_base": None, "trunk_modifier": None, "trunk_score": None,
            "neck_base": None, "neck_modifier": None, "neck_score": None,
            "legs_score": None, "score_a_raw": None, "score_a": None,
            "upper_arm_r_adj": None, "upper_arm_l_adj": None, "upper_arm_adj": None,
            "lower_arm_score": None, "wrist_adj": None, "dominant_arm": None,
            "score_b_raw": None, "score_b": None,
            "reba_total": float("nan"), "risk_level": "invalid",
        }

    # ── Group A ──────────────────────────────────────────────────────────────
    t_base   = trunk_base(trunk_sagittal)
    # abs(): trunk_twist and trunk_lateral are SIGNED (see angles.py sign
    # conventions). REBA's modifier is severity-only and direction-agnostic —
    # a 15 deg bend to the left is as awkward as 15 deg to the right — so
    # compare magnitude. Without abs() every left-side bend/twist would
    # silently fail the threshold and lose its +1.
    t_mod    = int(abs(trunk_twist) > TWIST_THRESH_DEG
                   or abs(trunk_lateral) > LATERAL_THRESH_DEG)
    t_score  = min(t_base + t_mod, 5)

    n_base   = neck_base(neck_flex)
    n_mod    = int(neck_lateral > NECK_LATERAL_THRESH)
    n_score  = min(n_base + n_mod, 3)

    l_score  = legs_base(knee_max, unilateral=unilateral)  # includes knee modifier

    raw_a    = TABLE_A[t_score - 1][n_score - 1][min(l_score, 4) - 1]
    score_a  = min(raw_a + load + activity, 12)

    # ── Group B (worst arm) ──────────────────────────────────────────────────
    def _group_b_side(ua_deg: float, ua_abd: float, la_deg: float, wr_deg: float) -> tuple[int, int, int, int]:
        ua_b   = upper_arm_base(ua_deg)
        ua_mod = int(ua_abd > ABD_THRESH_DEG)
        ua_adj = min(ua_b + ua_mod, 6)

        la_b   = lower_arm_base(la_deg)
        # Wrist: base score only (1-2). REBA's +1 wrist modifier is for a
        # SEPARATE condition — "wrist is bent from midline or twisted" — not a
        # second test of the same flexion magnitude. The previous code applied
        # `wr_mod = int(wr_deg > WRIST_DEV_THRESH_DEG)` with the SAME 15 deg
        # threshold as wrist_base(), so it double-counted one measurement and
        # made wrist_adj take only {1, 3}: the middle category was unreachable.
        #
        # The modifier is not recoverable from this marker set. A single hand
        # marker (RFIN) cannot separate wrist flexion/extension from radial/ulnar
        # deviation or pronation — that is what the module docstring means by
        # "wrist is low-fidelity by design". So the honest score is the base
        # alone; wrist_adj is 1-2 and never claims REBA's worst wrist category.
        # This is conservative in the right direction (it cannot invent risk).
        wr_b   = wrist_base(wr_deg)
        wr_adj = wr_b

        raw_b  = TABLE_B[ua_adj - 1][la_b - 1][wr_adj - 1]
        return ua_adj, la_b, wr_adj, raw_b

    ua_r_adj, la_r, wr_r, raw_b_r = _group_b_side(upper_arm_r, upper_arm_r_abd, lower_arm_r, wrist_r)
    ua_l_adj, la_l, wr_l, raw_b_l = _group_b_side(upper_arm_l, upper_arm_l_abd, lower_arm_l, wrist_l)

    # Use worst arm (higher raw B score)
    if raw_b_r >= raw_b_l:
        raw_b, dominant = raw_b_r, "right"
        ua_adj, la, wr = ua_r_adj, la_r, wr_r
    else:
        raw_b, dominant = raw_b_l, "left"
        ua_adj, la, wr = ua_l_adj, la_l, wr_l

    score_b = min(raw_b + coupling, 12)

    # ── Final REBA ───────────────────────────────────────────────────────────
    reba = TABLE_C[score_a - 1][score_b - 1]

    risk = (
        "negligible" if reba == 1 else
        "low"        if reba <= 3 else
        "medium"     if reba <= 7 else
        "high"       if reba <= 10 else
        "very_high"
    )

    return {
        # Group A
        "trunk_base":     t_base,
        "trunk_modifier": t_mod,
        "trunk_score":    t_score,
        "neck_base":      n_base,
        "neck_modifier":  n_mod,
        "neck_score":     n_score,
        "legs_score":     l_score,
        "score_a_raw":    raw_a,
        "score_a":        score_a,
        # Group B
        "upper_arm_r_adj": ua_r_adj,
        "upper_arm_l_adj": ua_l_adj,
        "upper_arm_adj":   ua_adj,
        "lower_arm_score": la,
        "wrist_adj":       wr,
        "dominant_arm":    dominant,
        "score_b_raw":     raw_b,
        "score_b":         score_b,
        # Final
        "reba_total":      reba,
        "risk_level":      risk,
    }


# ── Vectorised time-series ────────────────────────────────────────────────────

def score_timeseries(
    angles: dict[str, np.ndarray],
    *,
    load:      int = 0,
    coupling:  int = 0,
    activity:  int = 1,
    unilateral: bool = False,
) -> pd.DataFrame:
    """Compute REBA scores for every frame.

    Args:
        angles: Output of angles.compute_reba_angles() — dict of (F,) arrays.
        load, coupling, activity, unilateral: See score_frame().

    Returns:
        DataFrame with one row per frame and all component scores.
    """
    F = next(iter(angles.values())).shape[0]
    rows = []
    for i in range(F):
        row = score_frame(
            trunk_sagittal = float(angles["trunk_sagittal_deg"][i]),
            trunk_twist    = float(angles["trunk_twist_deg"][i]),
            trunk_lateral  = float(angles["trunk_lateral_deg"][i]),
            neck_flex      = float(angles["neck_flex_deg"][i]),
            neck_lateral   = float(angles["neck_lateral_deg"][i]),
            knee_max       = float(angles["knee_max_deg"][i]),
            upper_arm_r    = float(angles["upper_arm_r_deg"][i]),
            upper_arm_l    = float(angles["upper_arm_l_deg"][i]),
            upper_arm_r_abd = float(angles["upper_arm_r_abd_deg"][i]),
            upper_arm_l_abd = float(angles["upper_arm_l_abd_deg"][i]),
            lower_arm_r    = float(angles["lower_arm_r_deg"][i]),
            lower_arm_l    = float(angles["lower_arm_l_deg"][i]),
            wrist_r        = float(angles["wrist_r_deg"][i]),
            wrist_l        = float(angles["wrist_l_deg"][i]),
            load           = load,
            coupling       = coupling,
            activity       = activity,
            unilateral     = unilateral,
        )
        rows.append(row)
    return pd.DataFrame(rows)


# ── Summary statistics ────────────────────────────────────────────────────────

RISK_COLORS = {
    "negligible": "#2ecc71",
    "low":        "#f1c40f",
    "medium":     "#e67e22",
    "high":       "#e74c3c",
    "very_high":  "#8e44ad",
}


def summarise(reba_df: pd.DataFrame) -> dict:
    """Return trial-level REBA summary statistics.

    Args:
        reba_df: Output of score_timeseries().

    Returns:
        dict with mean, median, max, min, std, risk distribution, and
        n_valid_frames/n_dropped_frames. Frames where score_frame() flagged
        risk_level == "invalid" (occluded marker → NaN angle) are EXCLUDED
        from every statistic and from peak-frame detection — they must never
        be allowed to look like a real (let alone a worst-case) posture.

    Raises:
        ValueError: if every frame in the trial is invalid (nothing to summarise).
    """
    valid = reba_df[reba_df["risk_level"] != "invalid"]
    n_dropped = len(reba_df) - len(valid)
    if len(valid) == 0:
        raise ValueError(
            "All frames are invalid (occluded) — nothing to summarise. "
            "Check marker tracking quality for this trial."
        )

    scores = valid["reba_total"].to_numpy()
    risk_counts = valid["risk_level"].value_counts().to_dict()
    peak_idx = int(np.argmax(scores))            # local index into `valid`
    peak_row = valid.iloc[peak_idx]

    return {
        "n_frames":          len(reba_df),
        "n_valid_frames":    len(valid),
        "n_dropped_frames":  n_dropped,
        "mean":              float(np.mean(scores)),
        "median":            float(np.median(scores)),
        "max":               int(np.max(scores)),
        "min":               int(np.min(scores)),
        "std":               float(np.std(scores)),
        "peak_frame":        int(peak_row.name),   # original reba_df row index
        "risk_distribution": risk_counts,
        "peak_score_a":      int(peak_row["score_a"]),
        "peak_score_b":      int(peak_row["score_b"]),
        "peak_trunk":        int(peak_row["trunk_score"]),
        "peak_neck":         int(peak_row["neck_score"]),
        "peak_legs":         int(peak_row["legs_score"]),
        "peak_upper_arm":    int(peak_row["upper_arm_adj"]),
        "peak_lower_arm":    int(peak_row["lower_arm_score"]),
        "peak_wrist":        int(peak_row["wrist_adj"]),
    }
