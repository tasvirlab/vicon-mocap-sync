"""Compute REBA-critical joint angles from Vicon 3D marker positions.

Coordinate system (Vicon Nexus, Z-up — confirmed from lab data, assumption A7
resolved: pelvis/HJC height ~880 mm, right-handed Z-up):
    Z = vertical (upward)

Sagittal / frontal decomposition is done in a PER-FRAME PELVIS FRAME, not the
world axes. The pelvis frame is built from the joints already present in the
21-joint set:
    right   = unit(RHIP - LHIP)          # mediolateral, subject's right
    up      = world +Z
    forward = unit(up x right)           # anterior, subject's facing direction
This makes trunk/neck flexion sign and lateral-bend/abduction magnitudes
INDEPENDENT of which way the subject faces in the room — required for the
axial-twisting and load-carrying tasks where the subject turns (previously the
sign was decided by world +Y and would invert when facing -Y).

Wrist caveat (REBA Group B): wrist "deviation" below is a single 3-point angle
(RELB-RWR-RHAND) with RHAND = the finger (RFIN) marker. A one-marker hand
cannot separate flexion/extension from radial/ulnar deviation, so the wrist
sub-score is low-fidelity by construction — treat it as an approximate/known
limitation, not a measured wrist posture.

All inputs are in mm (Vicon world frame). All outputs are in degrees.

The trunk direction is defined as (shoulder_mid - PELCEN), which avoids the
posterior-marker placement offset of T10.

SIGN CONVENTIONS (load-bearing — do not "simplify" to magnitudes)
────────────────────────────────────────────────────────────────
trunk_sagittal_deg, trunk_lateral_deg, trunk_twist_deg and neck_flex_deg are
SIGNED. Consumers that only need severity (REBA's threshold modifiers) must
take abs() at the point of use — see reba.py. The signs exist because trunk
angular VELOCITY is a low-back-disorder predictor (Marras et al. 1993, Spine
18(5):617-628) and d|theta|/dt != |d(theta)/dt|: taking the magnitude first puts
a kink at every neutral crossing, and differentiating that kink fabricates a
velocity spike. Any derivative must be taken on the signed angle, then reduced
to a magnitude if needed. See kinematics.py.
"""

from __future__ import annotations

import numpy as np


# ── Low-level geometry ───────────────────────────────────────────────────────

def _unit(v: np.ndarray) -> np.ndarray:
    """L2-normalize along last axis. Shape (..., 3) → (..., 3)."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < 1e-6, 1.0, n)


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Angle in degrees between two (F, 3) vector arrays. Returns (F,)."""
    cos = np.clip(np.einsum("...i,...i", _unit(v1), _unit(v2)), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _angle_at_joint(
    prox: np.ndarray,
    joint: np.ndarray,
    distal: np.ndarray,
) -> np.ndarray:
    """Interior angle at `joint` formed by prox-joint-distal. Returns (F,) degrees."""
    return _angle_deg(prox - joint, distal - joint)


# ── Main computation ─────────────────────────────────────────────────────────

def compute_reba_angles(joints: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute all REBA-critical joint angles for every frame.

    Args:
        joints: dict {name: (F, 3) ndarray} in mm, Vicon Z-up frame.

    Returns:
        dict {angle_name: (F,) ndarray in degrees}.
        All values are non-negative unless noted (sagittal sign encodes direction).
    """
    F = next(iter(joints.values())).shape[0]
    VERT = np.tile([[0.0, 0.0, 1.0]], (F, 1))  # vertical unit vector (Z-up)

    # ── Derived landmarks ────────────────────────────────────────────────────
    sho_mid = (joints["RSHO"] + joints["LSHO"]) / 2.0   # (F,3)

    # ── Per-frame pelvis frame (facing-independent) ──────────────────────────
    # right: mediolateral (subject's right), from the hip bar.
    # up:    world vertical (Z-up confirmed).
    # forward: anterior (subject's facing) = up x right.
    # Projecting onto {forward, up} gives sagittal angles that do NOT depend on
    # which way the subject faces in the room (fixes the old world-+Y assumption).
    up      = VERT
    right   = _unit(joints["RHIP"] - joints["LHIP"])          # (F,3)
    forward = _unit(np.cross(up, right))                       # (F,3)

    def _dot(a, b):
        return np.einsum("...i,...i", a, b)

    # ── Trunk ────────────────────────────────────────────────────────────────
    trunk_vec  = sho_mid - joints["PELCEN"]   # pelvis → shoulder midpoint (F,3)
    trunk_unit = _unit(trunk_vec)

    t_up  = _dot(trunk_vec, up)
    t_fwd = _dot(trunk_vec, forward)
    t_rt  = _dot(trunk_vec, right)

    # Sagittal flexion in the pelvis sagittal plane: + = forward flexion,
    # − = extension. Sign comes from the anterior (forward) component, so it is
    # correct regardless of the subject's world-facing direction.
    trunk_sagittal_deg = np.degrees(np.arctan2(t_fwd, t_up))
    # Lateral bending: SIGNED mediolateral deviation, + = bend toward subject's
    # right. The second arm is the in-sagittal-plane magnitude sqrt(up^2+fwd^2),
    # NOT t_up alone: with bare t_up this angle blows up toward 180 deg whenever
    # the trunk passes horizontal (t_up < 0), so a deep forward bend was being
    # reported as a huge "lateral" bend (measured: 169 deg lateral at 112 deg
    # flexion, corr(|sagittal|, lateral) = 0.71). That inflated REBA's lateral
    # modifier and made lateral angular velocity — a Marras LBD predictor —
    # physically impossible (2263 deg/s). This matches the decomposition already
    # used by neck_lateral_deg and the upper-arm abduction angles below.
    # Equivalent to arcsin(t_rt / |trunk_vec|); bounded to +/-90 deg by
    # construction and independent of sagittal flexion.
    trunk_lateral_deg  = np.degrees(np.arctan2(
        t_rt, np.sqrt(t_up ** 2 + t_fwd ** 2 + 1e-9)))
    # Total inclination from vertical (scalar).
    trunk_total_deg    = _angle_deg(trunk_unit, VERT)

    # Twist: shoulder bar vs hip bar in the horizontal plane. Already a relative
    # angle between two body bars, so inherently facing-independent.
    # SIGNED via atan2(cross_z, dot) rather than arccos(dot): arccos is
    # ill-conditioned exactly at small twist (d/dcos -> inf as cos -> 1), which
    # is where subjects spend most of their time, and it discards the direction
    # of rotation that trunk twisting VELOCITY needs. Magnitude is unchanged
    # from the previous arccos form — only the sign is new.
    sho_bar = joints["RSHO"] - joints["LSHO"]   # (F,3)
    hip_bar = joints["RHIP"] - joints["LHIP"]   # (F,3)
    sho_bar_xy = np.c_[sho_bar[:, :2], np.zeros(F)]
    hip_bar_xy = np.c_[hip_bar[:, :2], np.zeros(F)]
    twist_cross_z   = np.cross(hip_bar_xy, sho_bar_xy)[:, 2]
    trunk_twist_deg = np.degrees(np.arctan2(twist_cross_z,
                                            _dot(hip_bar_xy, sho_bar_xy)))

    # ── Neck ─────────────────────────────────────────────────────────────────
    # Neck sagittal angle in the same pelvis frame, then subtract the trunk tilt
    # so the result is neck flexion RELATIVE TO THE TRUNK (REBA convention),
    # sign preserved (+ flexion, − extension).
    neck_vec = joints["HD"] - joints["C7"]           # (F,3)
    n_up  = _dot(neck_vec, up)
    n_fwd = _dot(neck_vec, forward)
    n_rt  = _dot(neck_vec, right)
    neck_sagittal_world = np.degrees(np.arctan2(n_fwd, n_up))
    neck_flex_deg = neck_sagittal_world - trunk_sagittal_deg   # signed, rel. to trunk

    # Neck lateral bending: mediolateral deviation magnitude.
    neck_lateral_deg = np.degrees(np.arctan2(
        np.abs(n_rt), np.sqrt(n_up ** 2 + n_fwd ** 2 + 1e-9)))

    # ── Upper arm elevation (both sides) ────────────────────────────────────
    # "Neutral hanging" = direction opposite to trunk (arm hanging at side).
    hanging = -trunk_unit   # (F,3) pointing downward along trunk

    def _upper_arm(sho: np.ndarray, elb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Total 3D elevation from neutral hanging, plus lateral abduction angle."""
        arm_vec = elb - sho  # shoulder→elbow
        elev = _angle_deg(arm_vec, hanging)   # 0° = hanging, 90° = horizontal
        # Abduction: mediolateral deviation in the pelvis frame (facing-independent).
        a_rt  = _dot(arm_vec, right)
        a_up  = _dot(arm_vec, up)
        a_fwd = _dot(arm_vec, forward)
        abd = np.degrees(np.arctan2(np.abs(a_rt), np.sqrt(a_up ** 2 + a_fwd ** 2 + 1e-9)))
        return elev, abd

    upper_arm_r_deg, upper_arm_r_abd_deg = _upper_arm(joints["RSHO"], joints["RELB"])
    upper_arm_l_deg, upper_arm_l_abd_deg = _upper_arm(joints["LSHO"], joints["LELB"])

    # ── Elbow (lower arm) flexion ────────────────────────────────────────────
    # Elbow interior angle then convert to flexion (0° = fully extended).
    elbow_r_interior = _angle_at_joint(joints["RSHO"], joints["RELB"], joints["RWR"])
    lower_arm_r_deg = 180.0 - elbow_r_interior   # flexion from straight

    elbow_l_interior = _angle_at_joint(joints["LSHO"], joints["LELB"], joints["LWR"])
    lower_arm_l_deg = 180.0 - elbow_l_interior

    # ── Wrist flexion / deviation ────────────────────────────────────────────
    # Wrist deviation = angular difference of hand from forearm extension.
    wrist_r_deg = np.abs(180.0 - _angle_at_joint(joints["RELB"], joints["RWR"], joints["RHAND"]))
    wrist_l_deg = np.abs(180.0 - _angle_at_joint(joints["LELB"], joints["LWR"], joints["LHAND"]))

    # ── Knee flexion ─────────────────────────────────────────────────────────
    knee_r_deg = 180.0 - _angle_at_joint(joints["RHIP"], joints["RKNE"], joints["RANK"])
    knee_l_deg = 180.0 - _angle_at_joint(joints["LHIP"], joints["LKNE"], joints["LANK"])

    # ── Hip flexion (informational) ──────────────────────────────────────────
    # Angle at hip joint between trunk-down direction and thigh direction.
    thigh_r = joints["RKNE"] - joints["RHIP"]
    thigh_l = joints["LKNE"] - joints["LHIP"]
    # NOTE: this is a THIGH-vs-TRUNK angle, not the anatomical hip angle. PiG's
    # R/LHipAngles measure thigh vs PELVIS; the two differ by the lumbar
    # flexion between pelvis and trunk, which is large during lifting (measured
    # bias +22 deg vs PiG on SUB01/lifting, r=0.986 — the shape tracks, the
    # offset is definitional). Reported as informational only and deliberately
    # NOT used by REBA (reba.score_frame takes knee_max, never hip). Do not
    # "validate" it against RHipAngles without first re-defining it against the
    # pelvis frame.
    hip_r_deg = _angle_deg(thigh_r, -trunk_unit)   # 0° = thigh parallel to trunk (standing)
    hip_l_deg = _angle_deg(thigh_l, -trunk_unit)

    return {
        # Trunk
        "trunk_total_deg":     trunk_total_deg,      # scalar inclination from vertical
        "trunk_sagittal_deg":  trunk_sagittal_deg,   # signed: + = forward lean
        "trunk_lateral_deg":   trunk_lateral_deg,    # SIGNED: + = bend to subject's right, +/-90 deg
        "trunk_twist_deg":     trunk_twist_deg,      # SIGNED shoulder-hip bar angle in XY plane
        # Neck
        "neck_flex_deg":       neck_flex_deg,        # signed: + = flexion, − = extension (relative to trunk)
        "neck_lateral_deg":    neck_lateral_deg,     # lateral head tilt magnitude
        # Right arm
        "upper_arm_r_deg":     upper_arm_r_deg,      # 3D elevation from hanging
        "upper_arm_r_abd_deg": upper_arm_r_abd_deg,  # lateral abduction magnitude
        "lower_arm_r_deg":     lower_arm_r_deg,      # elbow flexion (0 = straight)
        "wrist_r_deg":         wrist_r_deg,          # wrist deviation from straight
        # Left arm
        "upper_arm_l_deg":     upper_arm_l_deg,
        "upper_arm_l_abd_deg": upper_arm_l_abd_deg,
        "lower_arm_l_deg":     lower_arm_l_deg,
        "wrist_l_deg":         wrist_l_deg,
        # Legs
        "knee_r_deg":          knee_r_deg,           # knee flexion (0 = straight)
        "knee_l_deg":          knee_l_deg,
        "knee_max_deg":        np.maximum(knee_r_deg, knee_l_deg),
        "hip_r_deg":           hip_r_deg,
        "hip_l_deg":           hip_l_deg,
    }


def angles_to_dataframe(
    angles: dict[str, np.ndarray],
    frame_col: np.ndarray | None = None,
    time_col: np.ndarray | None = None,
) -> "pd.DataFrame":  # noqa: F821
    """Convert angles dict to a tidy DataFrame with optional frame/time columns."""
    import pandas as pd

    df = pd.DataFrame(angles)
    if frame_col is not None:
        df.insert(0, "frame", frame_col)
    if time_col is not None:
        df.insert(1 if frame_col is not None else 0, "time_s", time_col)
    return df
