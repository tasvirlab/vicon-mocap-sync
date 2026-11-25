"""Repetition counting for repetitive material-handling tasks.

Counts movement cycles from a signed angle signal (default: trunk sagittal
flexion) via prominence-gated peak detection, and derives the Marras model's
lift-rate input plus per-cycle timing for recovery/duty-cycle analysis.

Why peak detection and not a learned model: this project's trials are
single-subject, single-known-task, clean lab signals, so scipy find_peaks with a
prominence floor + a minimum inter-peak period is sufficient and fully
interpretable. Chen & Yu (2024, Autom. Constr.) need an SSM+transformer because
they count unstructured multi-worker *site* video with changing backgrounds —
a much harder problem than one known lifting trial.

METHODOLOGICAL CAVEAT (load-bearing — do NOT drop when reporting):
`lift_rate_per_hr` is the WITHIN-TRIAL instantaneous rate
(n_reps / trial_duration_s * 3600), NOT a validated shift-level lift rate. A
9-20 s lab clip cannot establish a sustained hourly cadence; the linear
extrapolation assumes the worker holds that pace for an hour, which is not
measured. Marras et al.'s own lift_rate was measured over a workplace
observation period. Always report this number as an extrapolation with its
assumption stated, and prefer a field-observed lift rate when one exists. The
output carries `lift_rate_is_within_trial_extrapolation: True` to force this.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, peak_prominences

# A genuine lift cycle swings the trunk by at least this much peak-to-neighbour;
# smaller oscillations are sway/noise, not a repetition.
DEFAULT_PROMINENCE_DEG = 15.0
# Two distinct lifts cannot be closer than this in time (fastest plausible MMH
# cadence); prevents a single lift's shoulder from being double-counted.
DEFAULT_MIN_PERIOD_S = 0.6

# ── Task awareness ───────────────────────────────────────────────────────────
#
# 1. WHAT the cycle is called. Only a repetitive material-handling task produces
#    a "lift"; every other task produces a movement cycle. This is not cosmetic:
#    `lift_rate_per_hr` is a Marras model INPUT, and feeding it a kneeling task's
#    trunk-flexion cadence would push a number into a lifting model that never
#    saw that kind of task. Non-lift tasks therefore return lift_rate_per_hr =
#    None, so the value cannot reach Marras even by accident.
LIFT_TASKS = ("lifting",)

# 2. WHICH PLANE defines the cycle. Counting on trunk sagittal flexion is right
#    for tasks whose cycle IS a bend, but a predominantly axial-twisting cycle
#    can complete with little flexion and would be undercounted. Task 8 is
#    literally "trunk flexion + axial twisting", so its cycle is counted on the
#    transverse channel. Substring-matched against task_id; default sagittal.
CYCLE_CHANNEL_BY_TASK = {
    "trunk_flexion_twisting": "trunk_twist_deg",
    "twisting":               "trunk_twist_deg",
}
DEFAULT_CYCLE_CHANNEL = "trunk_sagittal_deg"

# A different plane exceeding the counted plane's excursion by this factor means
# the cycle probably lives in that other plane — reported as a warning, never
# silently auto-switched (a silent switch would change counts between runs).
_DOMINANT_PLANE_FACTOR = 1.5


def cycle_kind(task_id: str) -> str:
    """'lift' for repetitive material handling, else 'movement_cycle'."""
    return "lift" if any(t in task_id for t in LIFT_TASKS) else "movement_cycle"


def cycle_channel(task_id: str) -> str:
    """Angle channel whose peaks define one cycle for this task."""
    for token, ch in CYCLE_CHANNEL_BY_TASK.items():
        if token in task_id:
            return ch
    return DEFAULT_CYCLE_CHANNEL


def count_task_repetitions(angles: dict, fps: float, task_id: str = "",
                           **kwargs) -> dict:
    """Task-aware wrapper: picks the counting plane, counts, and labels the rate.

    Adds to count_repetitions()'s output:
        cycle_kind          'lift' | 'movement_cycle'
        counting_channel    which angle channel the peaks were found in
        cycle_rate_per_hr   the within-trial rate, whatever the cycle is
        lift_rate_per_hr    the SAME number for a lift task, else None
        plane_warning       set when another plane's excursion dominates
    """
    ch = cycle_channel(task_id)
    series = angles.get(ch)
    if series is None:                      # requested plane absent -> fall back
        ch = DEFAULT_CYCLE_CHANNEL
        series = angles.get(ch, np.array([]))

    out = count_repetitions(np.asarray(series, dtype=float), fps, **kwargs)

    kind = cycle_kind(task_id)
    out["cycle_kind"] = kind
    out["counting_channel"] = ch
    out["cycle_rate_per_hr"] = out["lift_rate_per_hr"]
    # A movement-cycle cadence is NOT a lift rate and must not reach Marras.
    if kind != "lift":
        out["lift_rate_per_hr"] = None

    # Excursion check across the three trunk planes, reported not acted upon.
    ranges = {}
    for name in ("trunk_sagittal_deg", "trunk_lateral_deg", "trunk_twist_deg"):
        v = np.asarray(angles.get(name, []), dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            ranges[name] = round(float(np.ptp(v)), 2)
    out["plane_ranges_deg"] = ranges
    counted = ranges.get(ch)
    if counted:
        bigger = {k: r for k, r in ranges.items()
                  if k != ch and r > counted * _DOMINANT_PLANE_FACTOR}
        out["plane_warning"] = (
            f"cycles counted on {ch} (range {counted}deg) but "
            + ", ".join(f"{k} spans {r}deg" for k, r in bigger.items())
            + " — check that the counted plane is the one that defines a cycle "
              "for this task." if bigger else None)
    else:
        out["plane_warning"] = None
    return out



# Movement-onset criterion: a cycle is considered to have STARTED once the angle
# has risen this fraction of the way from its local base to the peak, and to have
# COMPLETED once it has fallen back through the same level. A fractional
# (prominence-relative) level is used rather than an absolute angle so the
# criterion means the same thing for a 20 deg cycle and a 60 deg one.
#
# Pre-specified algorithm parameter, NOT a validated ergonomic threshold — vary
# it in the sensitivity analysis and report it, exactly as for the hysteresis
# band. 0.10 is the conventional movement-onset fraction; it is not claimed to
# be optimal for these tasks.
DEFAULT_ONSET_FRACTION = 0.10


def segment_cycles(x: np.ndarray, fps: float, peaks: np.ndarray,
                   prominences: np.ndarray, left_bases: np.ndarray,
                   right_bases: np.ndarray,
                   onset_fraction: float = DEFAULT_ONSET_FRACTION) -> list[dict]:
    """Resolve each detected peak into an onset -> peak -> completion cycle.

    For peak i with prominence p and local base b = x[peak] - p, the crossing
    level is  L = b + onset_fraction * p.  Onset is the LAST sample before the
    peak at which the signal was still below L (searching back no further than
    the previous peak); completion is the FIRST sample after the peak at which
    it falls below L again (searching forward no further than the next peak).
    Bounding the search by the neighbouring peaks is what stops one cycle's
    onset being placed inside the previous cycle when the trunk never returns to
    baseline between repetitions.

    Completion is None when the signal never returns below L before the trial
    ends — a cycle that is still in progress at the end of the observation must
    not be given a fabricated end time.
    """
    n = x.size
    cycles: list[dict] = []
    for i, (pk, prom, lb, rb) in enumerate(zip(peaks, prominences,
                                               left_bases, right_bases)):
        pk = int(pk)
        # Crossing level is built from the LOCAL valley on each side, not from
        # the global prominence base. For the tallest peak in a trial the
        # prominence base runs all the way to the trial minimum, putting the
        # level far below the valleys that actually bracket the movement, which
        # can leave the trial's biggest cycle with BOTH boundaries unresolved. A per-side
        # local valley gives each boundary its own level and segments the cycle
        # the way a reader would draw it.
        lo_b = int(peaks[i - 1]) if i > 0 else 0
        hi_b = int(peaks[i + 1]) if i + 1 < peaks.size else n - 1
        left_seg = x[lo_b:pk + 1]
        right_seg = x[pk:hi_b + 1]
        left_valley = float(np.nanmin(left_seg)) if np.isfinite(left_seg).any() else float(x[pk])
        right_valley = float(np.nanmin(right_seg)) if np.isfinite(right_seg).any() else float(x[pk])
        level_left = left_valley + onset_fraction * (float(x[pk]) - left_valley)
        level_right = right_valley + onset_fraction * (float(x[pk]) - right_valley)
        level = float(x[pk] - prom + onset_fraction * prom)   # reported reference

        # Search back only as far as the previous peak, forward only as far as
        # the next: bounding by neighbours stops one cycle's onset being placed
        # inside the previous cycle when the trunk never returns to baseline.
        onset = None
        for j in range(pk, lo_b - 1, -1):
            if not np.isfinite(x[j]) or x[j] < level_left:
                onset = j
                break

        completion = None
        for j in range(pk, hi_b + 1):
            if np.isfinite(x[j]) and x[j] < level_right:
                completion = j
                break

        # UNRESOLVED BOUNDARIES ARE REPORTED AS None, NOT SUBSTITUTED.
        # Falling back to the search bound looked tidier but manufactured
        # numbers: on SUB04_kneeling_T01 it placed cycle 4's onset exactly on
        # cycle 3's peak, yielding a 3.57 s "rise time" that no movement made and
        # a -0.90 s "recovery" that read as overlapping cycles. A cycle whose
        # start or end never crosses the level inside its own window simply has
        # no defensible boundary, and every duration derived from it is None.
        rise_s = None if onset is None else round((pk - onset) / fps, 3)
        ret_s = None if completion is None else round((completion - pk) / fps, 3)
        dur_s = (None if onset is None or completion is None
                 else round((completion - onset) / fps, 3))
        cyc = {
            "index": i + 1,
            "onset_frame": None if onset is None else int(onset),
            "peak_frame": pk,
            "completion_frame": None if completion is None else int(completion),
            "onset_s": None if onset is None else round(onset / fps, 3),
            "peak_s": round(pk / fps, 3),
            "completion_s": None if completion is None else round(completion / fps, 3),
            "rise_time_s": rise_s,
            "return_time_s": ret_s,
            "cycle_duration_s": dur_s,
            "peak_angle_deg": round(float(x[pk]), 2),
            "prominence_deg": round(float(prom), 2),
            "crossing_level_deg": round(level, 2),
            "onset_level_deg": round(level_left, 2),
            "completion_level_deg": round(level_right, 2),
            "onset_resolved": onset is not None,
            "completion_resolved": completion is not None,
            "complete": onset is not None and completion is not None,
        }
        cycles.append(cyc)

    # Recovery = the gap between one cycle completing and the next starting.
    # Negative means the next cycle began before the previous one finished, i.e.
    # the movements overlap and there is no recovery interval at all.
    for a, b in zip(cycles[:-1], cycles[1:]):
        if a["completion_s"] is None or b["onset_s"] is None:
            a["recovery_to_next_s"] = None
        else:
            a["recovery_to_next_s"] = round(b["onset_s"] - a["completion_s"], 3)
    if cycles:
        cycles[-1]["recovery_to_next_s"] = None
    return cycles


def summarise_cycles(cycles: list[dict], trial_s: float) -> dict:
    """Trial-level timing summary over the segmented cycles."""
    done = [c for c in cycles if c["complete"]]
    durs = [c["cycle_duration_s"] for c in done]
    # Rise and return are usable whenever their own boundary resolved, even if
    # the opposite one did not — a resolved rise is still a measured rise.
    rises = [c["rise_time_s"] for c in cycles if c["rise_time_s"] is not None]
    rets = [c["return_time_s"] for c in cycles if c["return_time_s"] is not None]
    recs = [c["recovery_to_next_s"] for c in cycles
            if c.get("recovery_to_next_s") is not None]

    def _m(v):
        return round(float(np.mean(v)), 3) if v else None

    total = float(np.sum(durs)) if durs else 0.0
    return {
        "n_cycles_segmented": len(cycles),
        "n_cycles_complete": len(done),
        "n_onset_unresolved": sum(1 for c in cycles if not c["onset_resolved"]),
        "n_completion_unresolved": sum(1 for c in cycles if not c["completion_resolved"]),
        "mean_cycle_duration_s": _m(durs),
        "mean_rise_time_s": _m(rises),
        "mean_return_time_s": _m(rets),
        "mean_recovery_s": _m(recs),
        "min_recovery_s": round(float(np.min(recs)), 3) if recs else None,
        "total_cycle_time_s": round(total, 3),
        # Fraction of the OBSERVED interval spent inside a movement cycle. A
        # within-trial descriptor only — not a shift-level duty cycle.
        "duty_cycle_percent": (round(100.0 * total / trial_s, 2)
                               if trial_s and np.isfinite(trial_s) and trial_s > 0
                               else None),
    }


def _empty(trial_s: float, params: dict) -> dict:
    return {
        "n_reps": 0, "peak_frames": [], "peak_times_s": [],
        "cycles": [], "cycle_timing": summarise_cycles([], trial_s),
        "trial_duration_s": round(trial_s, 3) if np.isfinite(trial_s) else None,
        "peak_to_peak_periods_s": [], "mean_period_s": None, "period_cv": None,
        "lift_rate_per_hr": 0.0 if np.isfinite(trial_s) else None,
        "lift_rate_is_within_trial_extrapolation": True,
        "caveat": _CAVEAT, "params": params,
    }


_CAVEAT = (
    "lift_rate_per_hr is the within-trial instantaneous rate "
    "(n_reps / trial_duration * 3600), NOT a validated shift-level lift rate. "
    "Report as an extrapolation and state the assumption; prefer a "
    "field-observed rate when available."
)


def count_repetitions(signal: np.ndarray, fps: float,
                      *, prominence_deg: float = DEFAULT_PROMINENCE_DEG,
                      min_period_s: float = DEFAULT_MIN_PERIOD_S,
                      min_height_deg: float | None = None,
                      onset_fraction: float = DEFAULT_ONSET_FRACTION) -> dict:
    """Count movement cycles as prominent peaks in `signal`.

    Args:
        signal:         (F,) angle series (e.g. trunk_sagittal_deg). May contain
                        NaN — interior gaps are interpolated for peak-finding
                        only, not for any reported statistic.
        fps:            sample rate (Hz).
        prominence_deg: minimum peak prominence to count (rejects sway/noise).
        min_period_s:   minimum time between counted peaks.
        min_height_deg: optional absolute-height floor for a peak.

    Returns:
        dict: n_reps, peak_frames/peak_times_s, trial_duration_s,
        peak_to_peak_periods_s, mean_period_s, period_cv (cadence regularity),
        lift_rate_per_hr (+ extrapolation flag/caveat), params.
    """
    params = {"prominence_deg": prominence_deg, "min_period_s": min_period_s,
              "min_height_deg": min_height_deg, "onset_fraction": onset_fraction}
    x = np.asarray(signal, dtype=float)
    n = x.size
    trial_s = n / fps if fps and fps > 0 else float("nan")

    finite = np.isfinite(x)
    if finite.sum() < 3 or not np.isfinite(trial_s):
        return _empty(trial_s, params)

    # find_peaks cannot handle NaN — fill interior gaps only for detection.
    xf = x.copy()
    if not finite.all():
        idx = np.arange(n)
        xf[~finite] = np.interp(idx[~finite], idx[finite], x[finite])

    distance = max(1, int(round(min_period_s * fps)))
    kwargs: dict = {"prominence": prominence_deg, "distance": distance}
    if min_height_deg is not None:
        kwargs["height"] = min_height_deg
    peaks, props = find_peaks(xf, **kwargs)

    # Resolve every peak into onset -> peak -> completion. Segmentation runs on
    # the same gap-filled series the detection used, so a cycle boundary can
    # never land on a NaN the detector never saw.
    if peaks.size:
        prom, lbase, rbase = peak_prominences(xf, peaks)
        cycles = segment_cycles(xf, fps, peaks, prom, lbase, rbase, onset_fraction)
    else:
        cycles = []

    n_reps = int(peaks.size)
    periods = np.diff(peaks) / fps if peaks.size >= 2 else np.array([])
    mean_period = float(np.mean(periods)) if periods.size else float("nan")
    period_cv = (float(np.std(periods) / mean_period)
                 if periods.size and mean_period > 0 else float("nan"))
    lift_rate = n_reps / trial_s * 3600.0

    return {
        "n_reps": n_reps,
        "cycles": cycles,
        "cycle_timing": summarise_cycles(cycles, trial_s),
        "peak_frames": peaks.tolist(),
        "peak_times_s": [round(float(p) / fps, 3) for p in peaks],
        "trial_duration_s": round(trial_s, 3),
        "peak_to_peak_periods_s": [round(float(p), 3) for p in periods],
        "mean_period_s": round(mean_period, 3) if np.isfinite(mean_period) else None,
        "period_cv": round(period_cv, 3) if np.isfinite(period_cv) else None,
        "lift_rate_per_hr": round(lift_rate, 1),
        "lift_rate_is_within_trial_extrapolation": True,
        "caveat": _CAVEAT,
        "params": params,
    }


# ── Self-check ───────────────────────────────────────────────────────────────

def _demo() -> None:
    """Assert-based self-check with known cycle counts."""
    fps = 60.0

    # 1. Pure sine with exactly 5 full periods over 10 s -> 5 maxima.
    t = np.arange(0, 10, 1 / fps)
    x = 40 * np.sin(2 * np.pi * 0.5 * t)           # ampl 40 -> prominence ~80
    r = count_repetitions(x, fps)
    assert r["n_reps"] == 5, r["n_reps"]
    # lift_rate = 5 reps / 10 s * 3600 = 1800/hr
    assert abs(r["lift_rate_per_hr"] - 1800.0) < 1.0, r["lift_rate_per_hr"]
    assert r["lift_rate_is_within_trial_extrapolation"] is True
    # regular cadence -> low CV
    assert r["period_cv"] is not None and r["period_cv"] < 0.05, r["period_cv"]

    # 2. Small sway below the prominence floor must count as 0 reps.
    sway = 5 * np.sin(2 * np.pi * 0.5 * t)          # prominence ~10 < 15 floor
    assert count_repetitions(sway, fps)["n_reps"] == 0

    # 3. Minimum-period gate: two peaks closer than min_period_s collapse to one.
    two_fast = np.zeros(int(1.0 * fps))
    two_fast[20] = 50.0
    two_fast[25] = 50.0                             # 5 frames = 0.083 s apart
    assert count_repetitions(two_fast, fps, min_period_s=0.6)["n_reps"] == 1

    # 4. NaN gap does not crash and interior fill lets detection proceed.
    gapped = x.copy()
    gapped[100:110] = np.nan
    assert count_repetitions(gapped, fps)["n_reps"] == 5

    # 5. Too-short / all-NaN signal returns 0 cleanly.
    assert count_repetitions(np.full(2, np.nan), fps)["n_reps"] == 0

    # 6. CYCLE SEGMENTATION — onset -> peak -> completion.
    #    Symmetric triangular bumps on a flat baseline: what a repeated bend
    #    actually looks like, and the case where rise and return SHOULD match.
    #    (A full sine is deliberately not used here: its first cycle's left
    #    valley is the trial start rather than a real trough, so its onset and
    #    completion levels differ and rise != return by construction.)
    bump = np.concatenate([np.linspace(0, 45, 45), np.linspace(45, 0, 45)])
    flat = np.zeros(45)
    bumps = np.concatenate([flat] + [np.concatenate([bump, flat]) for _ in range(4)])
    seg = count_repetitions(bumps, fps)
    cyc = seg["cycles"]
    assert len(cyc) == 4, len(cyc)
    for c in cyc:
        assert c["onset_s"] < c["peak_s"], c          # ordering must hold
        if c["completion_s"] is not None:
            assert c["peak_s"] < c["completion_s"], c
    done = [c for c in cyc if c["complete"]]
    assert len(done) == 4, done
    for c in done:
        assert abs(c["rise_time_s"] - c["return_time_s"]) < 0.10, c   # symmetric
        assert abs(c["cycle_duration_s"]
                   - (c["rise_time_s"] + c["return_time_s"])) < 1e-6, c
    tim = seg["cycle_timing"]
    assert tim["n_cycles_segmented"] == 4
    assert 0.0 < tim["duty_cycle_percent"] <= 100.0, tim["duty_cycle_percent"]
    assert abs(tim["total_cycle_time_s"]
               - sum(c["cycle_duration_s"] for c in done)) < 1e-6

    #    A cycle still in progress at the end of the trial gets NO fabricated
    #    completion — the boundary is reported unresolved instead.
    rising = np.concatenate([np.zeros(60), np.linspace(0, 60, 240)])
    tail = count_repetitions(rising, fps, prominence_deg=15.0)
    for c in tail["cycles"]:
        if not c["completion_resolved"]:
            assert c["completion_s"] is None and c["cycle_duration_s"] is None, c

    #    Per-side local valleys: the TALLEST peak must still segment. Its
    #    prominence base runs to the global minimum, so a single global level
    #    left both its boundaries unresolved on real data.
    tall = np.concatenate([
        np.full(30, 10.0), np.linspace(10, 40, 60), np.linspace(40, 22, 60),
        np.linspace(22, 80, 90), np.linspace(80, 15, 90), np.full(60, 12.0)])
    rt2 = count_repetitions(tall, fps, prominence_deg=15.0)
    biggest = max(rt2["cycles"], key=lambda c: c["peak_angle_deg"])
    assert biggest["onset_resolved"] and biggest["completion_resolved"], biggest
    assert biggest["onset_level_deg"] > biggest["crossing_level_deg"],         "local-valley level must sit above the global prominence level here"

    #    Recovery between cycles is measured completion -> next onset.
    recs = [c["recovery_to_next_s"] for c in seg["cycles"]
            if c.get("recovery_to_next_s") is not None]
    assert recs and all(r >= 0 for r in recs), recs

    # 7. TASK AWARENESS — the cycle's NAME and PLANE follow the task.
    assert cycle_kind("SUB04_lifting_T01") == "lift"
    assert cycle_kind("SUB04_kneeling_T01") == "movement_cycle"
    assert cycle_channel("SUB04_lifting_T01") == "trunk_sagittal_deg"
    assert cycle_channel("SUB01_trunk_flexion_twisting_T02") == "trunk_twist_deg"

    twist = 35 * np.sin(2 * np.pi * 0.5 * t)          # cycles in the transverse plane
    flat  = 2 * np.sin(2 * np.pi * 0.5 * t)           # almost nothing sagittal
    angles = {"trunk_sagittal_deg": flat, "trunk_twist_deg": twist,
              "trunk_lateral_deg": flat}

    # A twisting task counts on the transverse plane and finds the cycles that
    # the sagittal channel would have missed entirely.
    rt = count_task_repetitions(angles, fps, "SUB01_trunk_flexion_twisting_T01")
    assert rt["counting_channel"] == "trunk_twist_deg", rt["counting_channel"]
    assert rt["n_reps"] == 5, rt["n_reps"]
    assert count_task_repetitions(angles, fps, "SUB01_kneeling_T01")["n_reps"] == 0

    # THE SAFETY PROPERTY: a non-lift task must never hand a lift rate onward,
    # or a kneeling cadence would be fed to a model built from lifting data.
    rk = count_task_repetitions({"trunk_sagittal_deg": x}, fps, "SUB04_kneeling_T01")
    assert rk["cycle_kind"] == "movement_cycle"
    assert rk["n_reps"] == 5 and rk["cycle_rate_per_hr"] == 1800.0
    assert rk["lift_rate_per_hr"] is None, "a movement-cycle rate is not a lift rate"

    rl = count_task_repetitions({"trunk_sagittal_deg": x}, fps, "SUB04_lifting_T01")
    assert rl["cycle_kind"] == "lift"
    assert rl["lift_rate_per_hr"] == rl["cycle_rate_per_hr"] == 1800.0

    # 8. Counting on a plane that another plane dwarfs raises a warning rather
    #    than silently switching (a silent switch would change counts per run).
    warned = count_task_repetitions(angles, fps, "SUB04_kneeling_T01")
    assert warned["counting_channel"] == "trunk_sagittal_deg"
    assert warned["plane_warning"] and "trunk_twist_deg" in warned["plane_warning"]
    assert count_task_repetitions({"trunk_sagittal_deg": x}, fps,
                                  "SUB04_lifting_T01")["plane_warning"] is None

    print("repetition self-check OK")


if __name__ == "__main__":
    _demo()
