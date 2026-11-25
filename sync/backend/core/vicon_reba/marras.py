"""Marras et al. (1993) low-back-disorder high-risk-group probability model.

Marras WS, Lavender SA, Leurgans SE, Rajulu SL, Allread WG, Fathallah FA,
Ferguson SA. The role of dynamic three-dimensional trunk motion in
occupationally-related low back disorders. Spine. 1993;18(5):617-628.
Table 4, p.624 (multiple logistic regression, N=403 industrial jobs:
111 high-risk / 124 low-risk). Coefficients transcribed directly from the
published table -- do not "round" or re-derive them from memory elsewhere.

Five predictors, in the paper's own units (Table 4 + Methods):
  lift_rate_per_hr           lifts/hour
  avg_twisting_vel_deg_s     AVERAGE twisting (axial) velocity, deg/s
  max_moment_nm              MAXIMUM external load moment, N.m
  max_sagittal_flexion_deg   MAXIMUM sagittal trunk flexion, degrees
  max_lateral_vel_deg_s      MAXIMUM lateral-bending velocity, deg/s

Note the mix: twisting is the AVERAGE, lateral bending is the MAXIMUM -- this
is not a typo, Table 4 lists them that way and it is the model that was
actually fitted (a stepwise selection over sagittal/lateral/twist x
mean/max/range/accel variants -- see Discussion, p.622: sagittal velocity was
tested and dropped because it added no discrimination once flexion angle was
in the model).

max_sagittal_flexion_deg MUST be zeroed the same way the source data was:
relative to each worker's own calibrated upright-standing posture, excluding
resting lumbar lordosis (Fig. 3 caption footnote: "Sagittal angle does not
include resting lordotic angle"). This is a per-subject baseline, not a fixed
anatomical reference -- reconcile against whatever zero convention angles.py
uses before treating this as a like-for-like comparison.

The output is the probability that a JOB (not an individual) is a member of
the paper's high-risk occupational-group classification -- the paper's own
Limitations (p.625) state this explicitly and note the model is not proof of
causation and applies only to repetitive manual-material-handling jobs.
"""
from __future__ import annotations

import math

# Table 4 (p.624). Constant + coefficients define the logit; SEs are kept for
# reference/reporting (e.g. a delta-method CI) but are not used in the point
# estimate below.
CONSTANT = -3.80
COEF_LIFT_RATE = 0.0014
COEF_AVG_TWISTING_VEL = 0.061
COEF_MAX_MOMENT = 0.024
COEF_MAX_SAGITTAL_FLEXION = 0.020
COEF_MAX_LATERAL_VEL = 0.036

STANDARD_ERRORS = {
    "constant": 0.67,
    "lift_rate": 0.0006,
    "avg_twisting_vel": 0.041,
    "max_moment": 0.004,
    "max_sagittal_flexion": 0.012,
    "max_lateral_vel": 0.014,
}

REQUIRED_FIELDS = (
    "lift_rate_per_hr",
    "avg_twisting_vel_deg_s",
    "max_moment_nm",
    "max_sagittal_flexion_deg",
    "max_lateral_vel_deg_s",
)


def lbd_high_risk_probability(
    lift_rate_per_hr: float,
    avg_twisting_vel_deg_s: float,
    max_moment_nm: float,
    max_sagittal_flexion_deg: float,
    max_lateral_vel_deg_s: float,
) -> float:
    """Probability of high-risk occupational LBD group membership.

    All five inputs are required. Individually each factor discriminates
    poorly between risk groups (Discussion, p.622) -- the model's predictive
    power depends on the full multivariate combination, so do not substitute
    a missing input with 0 or a population mean; that silently describes a
    different job. Use summarise_lbd_risk() when an input may be missing.
    """
    logit = (
        CONSTANT
        + COEF_LIFT_RATE * lift_rate_per_hr
        + COEF_AVG_TWISTING_VEL * avg_twisting_vel_deg_s
        + COEF_MAX_MOMENT * max_moment_nm
        + COEF_MAX_SAGITTAL_FLEXION * max_sagittal_flexion_deg
        + COEF_MAX_LATERAL_VEL * max_lateral_vel_deg_s
    )
    return 1.0 / (1.0 + math.exp(-logit))


def summarise_lbd_risk(
    lift_rate_per_hr: float | None,
    avg_twisting_vel_deg_s: float | None,
    max_moment_nm: float | None,
    max_sagittal_flexion_deg: float | None,
    max_lateral_vel_deg_s: float | None,
) -> dict:
    """lbd_high_risk_probability(), with explicit missing-input / applicability handling.

    Returns {"applicable": False, ...} instead of guessing when any of the
    five inputs is None/NaN -- following the model definition: the
    model applies only to trials that supply all five inputs and meet the
    repetitive-material-handling task condition, not to static-posture tasks
    (overhead holding, kneeling, isolated forward bending).
    """
    inputs = {
        "lift_rate_per_hr": lift_rate_per_hr,
        "avg_twisting_vel_deg_s": avg_twisting_vel_deg_s,
        "max_moment_nm": max_moment_nm,
        "max_sagittal_flexion_deg": max_sagittal_flexion_deg,
        "max_lateral_vel_deg_s": max_lateral_vel_deg_s,
    }
    missing = [k for k, v in inputs.items()
               if v is None or (isinstance(v, float) and math.isnan(v))]
    if missing:
        return {"applicable": False, "reason": f"missing inputs: {missing}",
                "probability": None}

    p = lbd_high_risk_probability(**inputs)
    return {"applicable": True, "reason": None, "probability": round(p, 4)}


# ── Self-check ───────────────────────────────────────────────────────────────

def _demo() -> None:
    """Assert-based self-check.

    Does NOT attempt to reproduce Figure 3's "34%" worked example -- that
    figure uses a different scaled-benchmark averaging method (proportional
    individual-factor logits divided by five), not this multivariate logistic
    formula. Instead this checks (a) monotonicity/sign of every coefficient
    matches the paper's Conclusion #3 ("as the magnitude of each of these
    variables increases, the risk increases"), and (b) directional validity
    against the paper's OWN published descriptive statistics: plugging in
    Table 2's High-Risk group means must give a materially higher probability
    than the Low-Risk group means.
    """
    # Table 2 (p.621) group means, matched to Table 4's five variables:
    #   Lift rate                              -> "Lift rate (lifts/hr)"
    #   Average twisting velocity               -> Twisting Plane "Average velocity"
    #   Maximum moment                          -> "Maximum moment (Nm)"
    #   Maximum sagittal flexion                -> Sagittal Plane "Maximum flexion position"
    #   Maximum lateral velocity                -> Lateral Plane "Maximum velocity"
    high_risk_means = dict(
        lift_rate_per_hr=175.89, avg_twisting_vel_deg_s=8.71,
        max_moment_nm=73.65, max_sagittal_flexion_deg=17.85,
        max_lateral_vel_deg_s=46.36,
    )
    low_risk_means = dict(
        lift_rate_per_hr=118.83, avg_twisting_vel_deg_s=5.44,
        max_moment_nm=23.64, max_sagittal_flexion_deg=10.37,
        max_lateral_vel_deg_s=35.45,
    )
    p_high = lbd_high_risk_probability(**high_risk_means)
    p_low = lbd_high_risk_probability(**low_risk_means)

    assert p_high > p_low, (p_high, p_low)
    assert p_high > 0.5, "high-risk group mean should sit above 50%"
    assert p_low < 0.5, "low-risk group mean should sit below 50%"

    # Monotonicity: bumping any single factor up must not decrease p.
    base = dict(low_risk_means)
    p_base = lbd_high_risk_probability(**base)
    for field in REQUIRED_FIELDS:
        bumped = dict(base)
        bumped[field] *= 1.5
        assert lbd_high_risk_probability(**bumped) > p_base, field

    # summarise_lbd_risk: missing-input handling.
    r = summarise_lbd_risk(None, 5.0, 20.0, 10.0, 30.0)
    assert r["applicable"] is False and r["probability"] is None, r
    r_ok = summarise_lbd_risk(**low_risk_means)
    assert r_ok["applicable"] is True
    assert abs(r_ok["probability"] - p_low) < 1e-4  # summarise_lbd_risk rounds to 4dp

    print(f"marras self-check OK (p_high={p_high:.3f}, p_low={p_low:.3f})")


if __name__ == "__main__":
    _demo()
