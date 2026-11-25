"""Angular velocity / acceleration from signed joint-angle time series.

This is the measurement layer for the trunk-motion-dynamics channel: Marras et
al. (1993, Spine 18(5):617-628) showed across 400+ industrial jobs that trunk
LATERAL VELOCITY and TWISTING VELOCITY are independent predictors of low-back
disorder risk, alongside sagittal trunk ANGLE, lifting frequency and load
moment. Position alone does not determine risk, so the pipeline needs
derivatives, not just postures.

Three rules make these numbers trustworthy; all three are load-bearing.

1. DIFFERENTIATE THE SIGNED ANGLE.
   d|theta|/dt != |d(theta)/dt|. Taking the magnitude first puts a kink at
   every neutral crossing; differentiating that kink fabricates a velocity
   spike. angles.py therefore emits SIGNED trunk sagittal/lateral/twist, and
   any magnitude is taken AFTER differentiation. See angles.py sign conventions.

2. FILTER BOTH SIDES TO THE SAME BANDWIDTH.
   Differentiation amplifies noise by ~f per derivative, so the low-pass cutoff
   silently defines what a "velocity" even means. Vicon marker data arriving via
   c3d_pipeline is ALREADY zero-phase Butterworth filtered at
   BUTTERWORTH_CUTOFF_HZ (6 Hz) at the native 100 Hz capture rate, so the Vicon
   path must pass cutoff_hz=None here — filtering twice over-smooths and biases
   velocity low. An RGB/model-derived angle series has NOT been filtered and
   must pass the SAME cutoff the Vicon side used (6.0 Hz), or the RGB-vs-Vicon
   comparison comes out rigged: the noisier stream is penalised for bandwidth it
   was never allowed to keep.

3. NEVER BRIDGE A GAP.
   Occlusion is NaN. Filtering runs per contiguous finite segment and
   differentiation lets NaN propagate, so a marker dropout can never manufacture
   a velocity across the hole.

All angles in degrees, velocities in deg/s, accelerations in deg/s^2.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

# Trunk channels that feed the Marras trunk-motion model. These must be the
# signed angles from angles.compute_reba_angles().
TRUNK_CHANNELS = ("trunk_sagittal_deg", "trunk_lateral_deg", "trunk_twist_deg")

DEFAULT_ORDER = 4


# ── Internals ────────────────────────────────────────────────────────────────

def _finite_runs(x: np.ndarray):
    """Yield (start, stop) index pairs of contiguous finite runs in x."""
    n = len(x)
    is_finite = np.isfinite(x)
    i = 0
    while i < n:
        if not is_finite[i]:
            i += 1
            continue
        j = i
        while j < n and is_finite[j]:
            j += 1
        yield i, j
        i = j


def unwrap_signed(x: np.ndarray, period: float = 360.0) -> np.ndarray:
    """Remove +/-period wrap discontinuities, per contiguous finite run.

    atan2-derived angles live on (-180, 180]. A wrap would inject a ~360 deg
    step between two frames, i.e. a ~21600 deg/s artifact at 60 Hz — the same
    class of physically-impossible velocity this module exists to avoid. Real
    trunk angles should never wrap (a shoulder bar cannot rotate past the hip
    bar by 180 deg), so this is a guard, not a transform: on well-formed data it
    is a no-op.

    Unwrapping is done per finite run so a NaN gap never gets a phase jump
    carried across it.
    """
    out = x.copy()
    for i, j in _finite_runs(x):
        if (j - i) >= 2:
            out[i:j] = np.unwrap(x[i:j], period=period)
    return out


def lowpass_series(
    x: np.ndarray,
    fs: float,
    cutoff_hz: float,
    order: int = DEFAULT_ORDER,
) -> np.ndarray:
    """Zero-phase Butterworth low-pass of a 1-D series, NaN-gap safe.

    Mirrors c3d_pipeline._filter_1d_segments: only contiguous finite runs long
    enough for a stable filtfilt are filtered; shorter runs are returned raw
    (real data, just unsmoothed) and NaN runs stay NaN. Never lets one run's
    edge artifacts bleed across a gap into another.

    Args:
        x:         (F,) angle series, may contain NaN.
        fs:        sample rate in Hz.
        cutoff_hz: low-pass cutoff, must be below Nyquist.
        order:     Butterworth order.

    Raises:
        ValueError: if cutoff_hz is at or above Nyquist.
    """
    nyquist = fs / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError(
            f"Cutoff {cutoff_hz} Hz must be below the Nyquist frequency "
            f"({nyquist} Hz at {fs} Hz)."
        )
    b, a = butter(order, cutoff_hz / nyquist, btype="low")
    min_len = 3 * max(len(a), len(b)) + 1   # scipy filtfilt padlen + 1

    out = x.copy()
    for i, j in _finite_runs(x):
        if (j - i) >= min_len:
            out[i:j] = filtfilt(b, a, x[i:j])
    return out


def derivative(x: np.ndarray, fs: float) -> np.ndarray:
    """d/dt of a 1-D series via central differences, NaN-propagating.

    np.gradient is 2nd-order accurate in the interior and one-sided at the
    edges. NaN is deliberately allowed to propagate to the neighbouring frames:
    a value adjacent to a dropout has no defined derivative, and emitting a
    number there would invent motion across the gap.

    Args:
        x:  (F,) series, may contain NaN.
        fs: sample rate in Hz.

    Returns:
        (F,) derivative per second. NaN where undefined.
    """
    if len(x) < 2:
        return np.full_like(x, np.nan, dtype=float)
    return np.gradient(x, 1.0 / fs)


# ── Public API ───────────────────────────────────────────────────────────────

def compute_kinematics(
    angles: dict[str, np.ndarray],
    fs: float,
    *,
    cutoff_hz: float | None = None,
    order: int = DEFAULT_ORDER,
    channels: tuple[str, ...] = TRUNK_CHANNELS,
) -> dict[str, np.ndarray]:
    """Angular velocity and acceleration for the requested angle channels.

    Args:
        angles:    dict {name: (F,) degrees} from angles.compute_reba_angles().
                   Channels MUST be signed — see module docstring rule 1.
        fs:        sample rate of the angle series in Hz (60.0 for
                   video-frame-aligned data, 100.0 for native Vicon).
        cutoff_hz: low-pass cutoff applied BEFORE differentiating, or None to
                   differentiate as-is. Pass None for Vicon-derived angles
                   (c3d_pipeline already filtered the markers at 6 Hz); pass the
                   SAME cutoff the Vicon side used (6.0) for RGB/model-derived
                   angles. See module docstring rule 2 — this choice is not
                   cosmetic, it defines the comparison.
        order:     Butterworth order when cutoff_hz is given.
        channels:  which angle names to differentiate.

    Returns:
        dict with, per channel `c`:
          f"{c}"              — the (possibly filtered, unwrapped) angle
          f"{c[:-4]}_vel_deg_s"  — angular velocity
          f"{c[:-4]}_acc_deg_s2" — angular acceleration
        (the `_deg` suffix is stripped before appending _vel/_acc).

    Raises:
        KeyError: if a requested channel is absent from `angles`.
    """
    missing = [c for c in channels if c not in angles]
    if missing:
        raise KeyError(
            f"Angle channels {missing} not found. Available: {sorted(angles)}"
        )

    out: dict[str, np.ndarray] = {}
    for c in channels:
        x = np.asarray(angles[c], dtype=float)
        x = unwrap_signed(x)
        if cutoff_hz is not None:
            x = lowpass_series(x, fs, cutoff_hz, order)

        vel = derivative(x, fs)
        acc = derivative(vel, fs)

        stem = c[:-4] if c.endswith("_deg") else c
        out[c] = x
        out[f"{stem}_vel_deg_s"] = vel
        out[f"{stem}_acc_deg_s2"] = acc
    return out


def summarise_trunk_motion(kin: dict[str, np.ndarray]) -> dict:
    """Trial-level trunk-motion predictors from compute_kinematics() output.

    Returns the quantities Marras et al. (1993) identified as trunk-motion
    predictors of low-back-disorder risk: peak sagittal trunk angle, and peak /
    mean ABSOLUTE lateral and twisting velocity. Magnitude is taken here —
    AFTER differentiation — which is the only order that is correct (module
    docstring rule 1).

    NaN frames (occlusion) are excluded from every statistic rather than being
    treated as zero motion.

    NOTE: this returns the model's INPUTS, not an LBD risk probability. The
    Marras logistic model's coefficients must be transcribed from the paper and
    verified before any probability is reported here — they are deliberately not
    hardcoded from memory.

    Args:
        kin: output of compute_kinematics() covering TRUNK_CHANNELS.

    Returns:
        dict of predictors, plus n_valid/n_dropped frame counts.

    Raises:
        KeyError: if the trunk channels are absent from `kin`.
    """
    required = ("trunk_sagittal_deg", "trunk_lateral_vel_deg_s",
                "trunk_twist_vel_deg_s")
    missing = [k for k in required if k not in kin]
    if missing:
        raise KeyError(
            f"{missing} missing — run compute_kinematics() with "
            f"channels=TRUNK_CHANNELS first."
        )

    def _stats(x: np.ndarray) -> tuple[float, float, int]:
        v = x[np.isfinite(x)]
        if v.size == 0:
            return float("nan"), float("nan"), 0
        return float(np.max(np.abs(v))), float(np.mean(np.abs(v))), int(v.size)

    sag = kin["trunk_sagittal_deg"]
    sag_valid = sag[np.isfinite(sag)]
    lat_peak, lat_mean, n_valid = _stats(kin["trunk_lateral_vel_deg_s"])
    tw_peak, tw_mean, _ = _stats(kin["trunk_twist_vel_deg_s"])

    return {
        # Position predictor
        "peak_trunk_sagittal_deg": (float(np.max(sag_valid))
                                    if sag_valid.size else float("nan")),
        # Velocity predictors (magnitude taken after differentiation)
        "peak_lateral_vel_deg_s": lat_peak,
        "mean_lateral_vel_deg_s": lat_mean,
        "peak_twist_vel_deg_s":   tw_peak,
        "mean_twist_vel_deg_s":   tw_mean,
        "n_valid_frames":         n_valid,
        "n_dropped_frames":       int(sag.size - sag_valid.size),
    }


# ── Self-check ───────────────────────────────────────────────────────────────

def _demo() -> None:
    """Assert-based self-check on synthetic signals with known derivatives."""
    fs = 60.0
    t = np.arange(0, 4, 1 / fs)

    # 1. Known sinusoid: theta = A sin(2 pi f t) -> vel peak = A*2*pi*f
    A, f = 30.0, 0.5
    theta = A * np.sin(2 * np.pi * f * t)
    kin = compute_kinematics(
        {"trunk_lateral_deg": theta, "trunk_sagittal_deg": theta,
         "trunk_twist_deg": theta}, fs)
    expect_peak = A * 2 * np.pi * f
    got = np.nanmax(np.abs(kin["trunk_lateral_vel_deg_s"]))
    assert abs(got - expect_peak) / expect_peak < 0.01, (got, expect_peak)

    # 2. The bug this module exists to prevent (rule 1). Note the PEAK velocity
    #    magnitude is NOT where the damage shows: |d|theta|/dt| == |d(theta)/dt|
    #    away from the kink, so peak |vel| is identical either way. The harm is
    #    (a) acceleration, which sees the kink as a corner and fabricates a
    #    ~19x spike, and (b) the velocity AT the neutral crossing, which is
    #    exactly where trunk velocity peaks and which abs-first zeroes out.
    vel_signed    = derivative(theta, fs)
    vel_abs_first = derivative(np.abs(theta), fs)
    acc_signed    = derivative(vel_signed, fs)
    acc_abs_first = derivative(vel_abs_first, fs)

    assert np.isclose(np.nanmax(np.abs(vel_abs_first)),
                      np.nanmax(np.abs(vel_signed))), \
        "peak |vel| is expected to be identical — if not, this test is stale"
    assert np.nanmax(np.abs(acc_abs_first)) > 10 * np.nanmax(np.abs(acc_signed)), \
        "abs-then-differentiate must fabricate an acceleration spike"

    zero_cross = int(round(fs / (2 * f)))   # theta crosses 0 with max |slope|
    assert abs(vel_signed[zero_cross]) > 0.99 * expect_peak, \
        "signed velocity must be at its peak at the neutral crossing"
    assert abs(vel_abs_first[zero_cross]) < 0.01 * expect_peak, \
        "abs-first must zero out the velocity at the neutral crossing"

    # 3. NaN gaps must not be bridged.
    gapped = theta.copy()
    gapped[100:110] = np.nan
    v = derivative(gapped, fs)
    assert np.isnan(v[100:110]).all(), "gap interior must stay NaN"
    assert np.isnan(v[99]) and np.isnan(v[110]), "gap edges have no derivative"
    assert np.isfinite(v[:99]).all(), "data before the gap must survive"

    # 4. Filtering must not bridge a gap either.
    fl = lowpass_series(gapped, fs, cutoff_hz=6.0)
    assert np.isnan(fl[100:110]).all(), "filter must not fill a gap"
    assert np.isfinite(fl[:99]).all(), "filter must keep valid runs"

    # 5. Unwrap guard: a fabricated 360 deg wrap must not survive.
    wrapped = np.concatenate([np.full(60, 179.0), np.full(60, -179.0)])
    vel_wrapped = derivative(unwrap_signed(wrapped), fs)
    assert np.nanmax(np.abs(vel_wrapped)) < 200.0, \
        "unwrap should remove the 360 deg step artifact"

    # 6. Summary excludes NaN and takes magnitude after differentiation.
    kin_g = compute_kinematics(
        {"trunk_lateral_deg": gapped, "trunk_sagittal_deg": gapped,
         "trunk_twist_deg": gapped}, fs)
    s = summarise_trunk_motion(kin_g)
    assert s["n_dropped_frames"] == 10, s
    assert np.isfinite(s["peak_lateral_vel_deg_s"]), s

    print("kinematics self-check OK")


if __name__ == "__main__":
    _demo()
