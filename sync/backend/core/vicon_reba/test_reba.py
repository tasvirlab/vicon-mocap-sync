"""Regression tests for vicon_reba.reba — run with: python test_reba.py

Covers the two bugs fixed in this module:
  1. trunk_base()/neck_base() must score SIGNED flexion/extension per the
     Hignett & McAtamney (2000) REBA tables, not unsigned total inclination.
  2. score_frame()/summarise() must mark occluded (NaN) frames "invalid" and
     exclude them from statistics/peak-detection — never silently score them
     as a worst-case posture.

No pytest dependency — plain asserts, runnable standalone or via `python -m
unittest` discovery is not needed since this is intentionally a flat script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # backend/
from core.vicon_reba import reba


def test_trunk_base_boundaries():
    """Exact thresholds from the action plan / Hignett & McAtamney (2000)."""
    assert reba.trunk_base(0)      == 1   # absolutely erect
    assert reba.trunk_base(10)     == 2   # mild flexion
    assert reba.trunk_base(20)     == 2   # flexion boundary, inclusive
    assert reba.trunk_base(20.01)  == 3   # just past flexion boundary
    assert reba.trunk_base(60)     == 3   # flexion boundary, inclusive
    assert reba.trunk_base(60.01)  == 4   # severe flexion
    assert reba.trunk_base(-10)    == 2   # mild extension
    assert reba.trunk_base(-20)    == 2   # extension boundary, inclusive
    assert reba.trunk_base(-20.01) == 3   # extension beyond 20°
    assert reba.trunk_base(-90)    == 3   # severe extension — capped at 3, REBA has no extension score 4
    assert reba.trunk_base(float("nan")) is None


def test_neck_base_boundaries():
    assert reba.neck_base(0)      == 1   # neutral
    assert reba.neck_base(20)     == 1   # flexion boundary, inclusive
    assert reba.neck_base(20.01)  == 2   # past flexion boundary
    assert reba.neck_base(-0.01)  == 2   # ANY extension scores 2, regardless of magnitude
    assert reba.neck_base(-15)    == 2
    assert reba.neck_base(float("nan")) is None


def test_lateral_bend_does_not_inflate_trunk_base():
    """A purely lateral lean (zero sagittal flexion) must score trunk_base
    as upright — lateral bend is a separate +1 modifier, not part of the
    position score. This is the exact bug that was fixed: trunk_base() used
    to receive trunk_total_deg (sagittal+lateral combined)."""
    assert reba.trunk_base(0) == 1  # sagittal=0 regardless of how much lateral lean occurred


def _neutral_kwargs(**overrides) -> dict:
    base = dict(
        trunk_sagittal=0.0, trunk_twist=0.0, trunk_lateral=0.0,
        neck_flex=0.0, neck_lateral=0.0, knee_max=0.0,
        upper_arm_r=0.0, upper_arm_l=0.0, upper_arm_r_abd=0.0, upper_arm_l_abd=0.0,
        lower_arm_r=80.0, lower_arm_l=80.0, wrist_r=0.0, wrist_l=0.0,
    )
    base.update(overrides)
    return base


def test_neutral_posture_scores_low_not_worst_case():
    row = reba.score_frame(**_neutral_kwargs())
    assert row["risk_level"] != "invalid"
    assert row["reba_total"] == 1
    assert row["trunk_score"] == 1


def test_occluded_frame_is_invalid_not_worst_case():
    """The core regression for the NaN-handling bug: an occluded marker must
    produce an 'invalid' frame, never a fabricated worst-case (4/severe)."""
    row = reba.score_frame(**_neutral_kwargs(trunk_sagittal=float("nan")))
    assert row["risk_level"] == "invalid"
    assert row["trunk_score"] is None
    assert np.isnan(row["reba_total"])


def test_summarise_excludes_invalid_frames_and_finds_real_peak():
    F = 10
    angles = {
        "trunk_total_deg":     np.zeros(F),
        "trunk_sagittal_deg":  np.zeros(F),
        "trunk_twist_deg":     np.zeros(F),
        "trunk_lateral_deg":   np.zeros(F),
        "neck_flex_deg":       np.zeros(F),
        "neck_lateral_deg":    np.zeros(F),
        "knee_max_deg":        np.zeros(F),
        "upper_arm_r_deg":     np.zeros(F),
        "upper_arm_l_deg":     np.zeros(F),
        "upper_arm_r_abd_deg": np.zeros(F),
        "upper_arm_l_abd_deg": np.zeros(F),
        "lower_arm_r_deg":     np.full(F, 80.0),
        "lower_arm_l_deg":     np.full(F, 80.0),
        "wrist_r_deg":         np.zeros(F),
        "wrist_l_deg":         np.zeros(F),
    }
    angles["trunk_sagittal_deg"][5] = np.nan   # occluded frame
    angles["trunk_sagittal_deg"][8] = 70.0     # genuine high-risk frame

    reba_df = reba.score_timeseries(angles)
    summary = reba.summarise(reba_df)

    assert summary["n_valid_frames"]   == 9
    assert summary["n_dropped_frames"] == 1
    assert summary["peak_frame"] == 8          # real posture, not the occluded frame
    assert summary["max"] == 4


def test_summarise_raises_when_all_frames_invalid():
    F = 3
    angles = {k: np.full(F, np.nan) for k in (
        "trunk_sagittal_deg", "trunk_twist_deg", "trunk_lateral_deg",
        "neck_flex_deg", "neck_lateral_deg", "knee_max_deg",
        "upper_arm_r_deg", "upper_arm_l_deg", "upper_arm_r_abd_deg", "upper_arm_l_abd_deg",
        "lower_arm_r_deg", "lower_arm_l_deg", "wrist_r_deg", "wrist_l_deg",
    )}
    reba_df = reba.score_timeseries(angles)
    try:
        reba.summarise(reba_df)
        raise AssertionError("expected ValueError when every frame is invalid")
    except ValueError:
        pass


def _synthetic_skeleton(flex_deg: float, lateral_deg: float = 0.0) -> dict:
    """Build a 1-frame Z-up skeleton with a known trunk flexion / lateral bend.

    Subject faces +Y, right hand side is +X, so the pelvis frame resolves to
    right=+X, up=+Z, forward=+Y. The trunk is rotated by `flex_deg` in the
    sagittal plane and `lateral_deg` toward the subject's right, so the expected
    angles are known exactly by construction.
    """
    f, l = np.radians(flex_deg), np.radians(lateral_deg)
    pel = np.array([[0.0, 0.0, 900.0]])
    L = 500.0
    # Trunk direction: lateral tilt toward +X (right), flexion toward +Y (fwd).
    trunk_dir = np.array([[np.sin(l),
                           np.cos(l) * np.sin(f),
                           np.cos(l) * np.cos(f)]])
    sho_mid = pel + L * trunk_dir
    bar = np.array([[200.0, 0.0, 0.0]])          # shoulder/hip bars along +X
    j = {
        "PELCEN": pel,
        "RHIP": pel + np.array([[100.0, 0.0, 0.0]]),
        "LHIP": pel - np.array([[100.0, 0.0, 0.0]]),
        "RSHO": sho_mid + bar,
        "LSHO": sho_mid - bar,
        "C7":   sho_mid,
        "HD":   sho_mid + np.array([[0.0, 0.0, 200.0]]),
    }
    # Remaining joints just need to be finite — they do not affect trunk angles.
    for name, offset in (("RELB", [250, 0, -200]), ("LELB", [-250, 0, -200]),
                         ("RWR", [250, 0, -400]),  ("LWR", [-250, 0, -400]),
                         ("RHAND", [250, 0, -500]), ("LHAND", [-250, 0, -500])):
        j[name] = sho_mid + np.array([offset], dtype=float)
    for name, offset in (("RKNE", [100, 0, -450]), ("LKNE", [-100, 0, -450]),
                         ("RANK", [100, 0, -880]), ("LANK", [-100, 0, -880])):
        j[name] = pel + np.array([offset], dtype=float)
    return j


def test_lateral_bend_decoupled_from_deep_flexion():
    """Regression: deep forward flexion must not read as a huge lateral bend.

    trunk_lateral_deg was arctan2(|t_rt|, t_up). Once the trunk passes
    horizontal t_up goes negative and the angle swung toward 180 deg with zero
    actual lateral bend — measured 169 deg lateral at 112 deg flexion on
    SUB01/lifting, corr(|sagittal|, lateral) = 0.71. That inflated REBA's
    lateral +1 modifier and made lateral angular velocity (a Marras low-back
    predictor) read 2263 deg/s.
    """
    from core.vicon_reba.angles import compute_reba_angles

    for flex in (0.0, 45.0, 89.0, 112.0, 150.0):
        a = compute_reba_angles(_synthetic_skeleton(flex_deg=flex))
        assert abs(a["trunk_sagittal_deg"][0] - flex) < 1e-6, \
            f"sagittal wrong at {flex} deg: {a['trunk_sagittal_deg'][0]}"
        assert abs(a["trunk_lateral_deg"][0]) < 1e-6, (
            f"pure {flex} deg flexion reported {a['trunk_lateral_deg'][0]:.1f} deg "
            f"of lateral bend — flexion/lateral coupling has regressed"
        )

    # Lateral bend must still be measured, signed, and bounded to +/-90.
    for lat in (-30.0, -10.0, 10.0, 30.0):
        a = compute_reba_angles(_synthetic_skeleton(flex_deg=0.0, lateral_deg=lat))
        assert abs(a["trunk_lateral_deg"][0] - lat) < 1e-6, \
            f"lateral {lat} deg misread as {a['trunk_lateral_deg'][0]}"

    # Both at once: each plane must read its own value, deep past horizontal.
    a = compute_reba_angles(_synthetic_skeleton(flex_deg=100.0, lateral_deg=20.0))
    assert abs(a["trunk_lateral_deg"][0] - 20.0) < 1e-6, a["trunk_lateral_deg"][0]


def test_reba_modifier_uses_magnitude_of_signed_trunk_angles():
    """trunk_lateral/twist are SIGNED — REBA must abs() them at the threshold.

    Without abs(), a bend/twist to the subject's LEFT is negative, fails the
    `> 10 deg` test, and silently loses its +1 trunk modifier.
    """
    base = dict(
        trunk_sagittal=0.0, neck_flex=0.0, neck_lateral=0.0, knee_max=0.0,
        upper_arm_r=0.0, upper_arm_l=0.0, upper_arm_r_abd=0.0,
        upper_arm_l_abd=0.0, lower_arm_r=80.0, lower_arm_l=80.0,
        wrist_r=0.0, wrist_l=0.0,
    )
    for side in (+1, -1):
        lat = reba.score_frame(trunk_twist=0.0, trunk_lateral=15.0 * side, **base)
        tw  = reba.score_frame(trunk_twist=15.0 * side, trunk_lateral=0.0, **base)
        assert lat["trunk_modifier"] == 1, f"lateral {15*side} deg lost its +1"
        assert tw["trunk_modifier"] == 1, f"twist {15*side} deg lost its +1"

    neutral = reba.score_frame(trunk_twist=0.0, trunk_lateral=0.0, **base)
    assert neutral["trunk_modifier"] == 0, "neutral trunk must not get +1"


def test_wrist_middle_category_is_reachable():
    """Regression: wrist_adj must span 1-2, not jump {1, 3}.

    The +1 wrist modifier used to re-apply wrist_base()'s own 15 deg threshold,
    so wrist_adj was 1 below 15 deg and 2+1=3 above it — category 2 could never
    occur. REBA's modifier is a separate condition ("bent from midline or
    twisted") that a single hand marker cannot measure, so it is now omitted.
    """
    base = dict(
        trunk_sagittal=0.0, trunk_twist=0.0, trunk_lateral=0.0, neck_flex=0.0,
        neck_lateral=0.0, knee_max=0.0, upper_arm_r=0.0, upper_arm_l=0.0,
        upper_arm_r_abd=0.0, upper_arm_l_abd=0.0, lower_arm_r=80.0,
        lower_arm_l=80.0,
    )
    seen = set()
    for dev in (0.0, 5.0, 14.9, 15.0, 15.1, 30.0, 60.0):
        r = reba.score_frame(wrist_r=dev, wrist_l=dev, **base)
        seen.add(r["wrist_adj"])
    assert 3 not in seen, f"wrist_adj must not reach 3 (unmeasurable modifier): {seen}"
    assert seen == {1, 2}, f"wrist_adj should span exactly 1-2, got {seen}"

    # Boundary: <=15 -> 1, >15 -> 2 (wrist_base only, no double-count)
    assert reba.score_frame(wrist_r=15.0, wrist_l=15.0, **base)["wrist_adj"] == 1
    assert reba.score_frame(wrist_r=15.1, wrist_l=15.1, **base)["wrist_adj"] == 2


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
