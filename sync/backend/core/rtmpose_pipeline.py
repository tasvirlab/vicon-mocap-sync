"""
rtmpose_pipeline.py
=====================
Run RTMPose-WholeBody (via rtmlib) on a trial's `<task>_rgb.mp4` and write
`<task>_pose2d.csv` — pixel-space 2D keypoints for the same 21-joint set used
throughout this project (vicon_reba/load.py).

Adapted from the standalone rtmpose_pose2d.py CLI script for use by
routers/pose2d.py's job-based endpoint (progress_cb reporting, batch processing).
See that script's original docstring for full design rationale — repeated briefly
here:

  - 12 of 21 joints map directly to COCO-WholeBody keypoints (shoulders, elbows,
    wrists, hips, knees, ankles).
  - PELCEN/C7 = hip/shoulder midpoint.
  - T10/L45 = lerp(C7, PELCEN, t) — t is a ROUGH placeholder unless overridden
    with calibrated values from dataset_export.py's train-only spine-ratio fit
    (see DEFAULT_T_T10 / DEFAULT_T_L45 below).

  Three joints are DELIBERATELY not the naive COCO-WholeBody mapping, because
  the Vicon marker they are trained against sits elsewhere. Do not "simplify"
  these back to the obvious keypoint:
  - HD    = mean(eyes + ears), NOT the nose. Vicon HD is the centroid of four
            head markers; the nose is anterior and its offset swings with head
            yaw. See the HEAD block in map_to_21_joints().
  - RHAND/LHAND = middle-finger MCP (hand root + 9), NOT the wrist/hand root —
            the Vicon Hand marker is the 3rd metacarpal head.
  - RTOE/LTOE   = SMALL toe, NOT the big toe — the Vicon Foot marker is the 5th
            metatarsal head, on the outer edge of the foot.

SUBJECT SELECTION — the lab footage contains bystanders
──────────────────────────────────────────────────────
process_trial() defaults to VICON-GUIDED selection: the subject is the detected
person whose hip midpoint is nearest the projected Vicon pelvis for that frame,
read from synced_vicon2d_60hz.csv (Vicon-PROJECTED, not RTMPose) (already frame-aligned with <task>_rgb.mp4 and in
the same normalised pixel space). This is a lookup, not a guess, and it is
available for every trial in this dataset because every trial has synchronised
Vicon.

The unguided fallback (highest body confidence on frame 0, then nearest-neighbour
thereafter) is a heuristic and is NOT safe here: several bystanders stand in
frame, they are frequently LESS occluded than a subject bent over a load, so one
can win frame 0 and be tracked for the whole trial — and when a frame detects
nobody the tracker re-initialises by confidence and can jump mid-trial. Use it
only to reproduce deployment behaviour, never to build the dataset.

Frames where no detection lies within MAX_PELVIS_MATCH_PX of the projected pelvis
are emitted as NaN with subject_found=False, rather than silently returning the
nearest bystander. run_on_video() logs match statistics and warns when >10% of
frames fail; check those before trusting a trial.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd

from utils.task_paths import Artifacts
from core.vicon_io import read_frame_ids

logger = logging.getLogger(__name__)

# ── COCO-WholeBody keypoint indices (133 total) — [ASSUMPTION: verify with visualize] ──
# Body block 0-16: 0 nose, 1/2 left/right eye, 3/4 left/right ear, 5/6 shoulders,
# 7/8 elbows, 9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles.
# Foot block 17-22: 17 L big toe, 18 L small toe, 19 L heel,
#                   20 R big toe, 21 R small toe, 22 R heel.
# Face 23-90. Hands: left 91-111, right 112-132 (21 keypoints each).
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
# HD is the mean of these four — NOT the nose. See HEAD_KPTS below.
HEAD_KPTS = (L_EYE, R_EYE, L_EAR, R_EAR)
L_SHO, R_SHO = 5, 6
L_ELB, R_ELB = 7, 8
L_WR, R_WR = 9, 10
L_HIP, R_HIP = 11, 12
L_KNE, R_KNE = 13, 14
L_ANK, R_ANK = 15, 16
# SMALL toe (5th metatarsal head side), not the big toe: the Vicon TOE marker is
# placed on the 5th metatarsal head, i.e. the OUTER edge of the foot. The big toe
# (17/20) is medial and would be a systematically different landmark from the 3D
# target the lifter is trained against.
L_SMALLTOE, R_SMALLTOE = 18, 21
L_HAND_START, R_HAND_START = 91, 112
L_HAND_MIDDLE_MCP = L_HAND_START + 9
R_HAND_MIDDLE_MCP = R_HAND_START + 9

# ── Spine-ratio placeholders for the synthetic T10/L45 joints ─────────────────────────
# T10/L45 = lerp(C7, PELCEN, t), where t=0 is C7 and t=1 is the pelvis centre.
# COCO-WholeBody has no spine keypoint, so these two joints carry NO independent
# information here — they are a deterministic function of C7 and PELCEN.
#
# These are PLACEHOLDERS, not measurements. dataset_export.fit_spine_ratio()
# re-fits t from the train-split Vicon 3D and apply_calibrated_spine_2d()
# overwrites these values, so the exported training npz does NOT depend on them.
# They DO determine <task>_pose2d.csv and the /visualize QA overlay — which is
# exactly where a wrong spine must not go unnoticed.
#
# Values below are fitted from SUB01/lifting Vicon 3D (822 frames; per-frame
# SD 0.026 / 0.038, so the ratio is stable within a trial). The previous
# 0.45/0.85 guesses were each off by ~0.10 in OPPOSITE directions, compressing
# the modelled spine by ~34% (0.40 separation vs. the true 0.605).
#
# [N=1 CAVEAT] Fitted from ONE subject. Refit across the full cohort once the
# remaining C3D data is processed, and update these two constants together.
DEFAULT_T_T10: float = 0.35   # SUB01 fit: 0.3483
DEFAULT_T_L45: float = 0.95   # SUB01 fit: 0.9531

# ── This project's canonical 21-joint set — matches vicon_reba/load.py exactly ──────────
JOINT_NAMES: list[str] = [
    "HD", "C7", "RSHO", "LSHO", "RELB", "LELB", "RWR", "LWR", "RHAND", "LHAND",
    "T10", "L45", "PELCEN", "RHIP", "LHIP", "RKNE", "LKNE", "RANK", "LANK",
    "RTOE", "LTOE",
]

# HD is NOT here: it is the mean of HEAD_KPTS, computed in map_to_21_joints().
DIRECT_JOINTS: dict[str, int] = {
    "RSHO": R_SHO, "LSHO": L_SHO, "RELB": R_ELB, "LELB": L_ELB,
    "RWR": R_WR, "LWR": L_WR, "RHIP": R_HIP, "LHIP": L_HIP,
    "RKNE": R_KNE, "LKNE": L_KNE, "RANK": R_ANK, "LANK": L_ANK,
    "RTOE": R_SMALLTOE, "LTOE": L_SMALLTOE,
    "RHAND": R_HAND_MIDDLE_MCP, "LHAND": L_HAND_MIDDLE_MCP,
}

# Minimum eye/ear keypoints above threshold before HD is emitted. Two is the
# fewest that still averages rather than trusting a single point.
MIN_HEAD_KPTS = 2

CONF_THRESHOLD_DEFAULT = 0.3
ProgressCb = Optional[Callable[[int, str], None]]

# Max pixel distance (in the normalised 1080p frame) between a detected person's
# hip midpoint and the projected Vicon pelvis before the match is rejected.
# A subject's whole body spans a few hundred pixels at 1080p, so a pelvis more than ~250 px away is a different person,
# not a noisy detection of the right one. Frames failing this emit NaN rather
# than a confident wrong answer.
MAX_PELVIS_MATCH_PX = 250.0


def load_vicon_pelvis_2d(synced2d_csv: Path) -> Optional[np.ndarray]:
    """Per-video-frame projected Vicon pelvis (N, 2) in rgb.mp4 pixel space.

    synced_vicon2d_60hz.csv (Vicon-PROJECTED, not RTMPose) is written by the Overlay stage: one row per frame of
    <task>_rgb.mp4, containing the Vicon skeleton projected through the camera
    calibration and rescaled into the normalised 1080p output space. It is
    therefore already frame-aligned with the video the detector reads — no
    re-projection, calibration loading, or rotation handling needed here.

    Returns None if the file is absent (deployment has no Vicon), so callers
    fall back to the unguided heuristic.
    """
    if synced2d_csv is None or not Path(synced2d_csv).exists():
        return None
    df = pd.read_csv(synced2d_csv)
    if "PELCEN_X" not in df.columns or "PELCEN_Y" not in df.columns:
        return None
    return df[["PELCEN_X", "PELCEN_Y"]].to_numpy(dtype=np.float64)


def select_tracked_person(
    keypoints: np.ndarray, scores: np.ndarray, prev_hip_mid: np.ndarray | None,
    ref_pelvis: np.ndarray | None = None,
    max_match_px: float = MAX_PELVIS_MATCH_PX,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray | None, float]:
    """Pick the subject from a multi-person detection.

    Selection priority:
      1. ref_pelvis (projected Vicon pelvis for THIS frame) — a lookup, not a
         guess. Picks the detected person whose hip midpoint is nearest, and
         REJECTS the frame if even the nearest is > max_match_px away.
      2. prev_hip_mid — nearest to the previous frame's pick.
      3. highest mean body confidence.

    Why (1) matters: the lab footage contains several bystanders. Under (3) a
    bystander can win frame 0 — they are often LESS occluded than a subject bent
    over a load — and (2) then follows that bystander for the whole trial. Worse,
    when a frame detects nobody the caller resets prev_hip_mid, so (3) re-runs
    mid-trial and can jump to a different person. Since every trial in this
    dataset has synchronised Vicon, the subject's true image position is known
    for every frame and none of that guessing is necessary.

    Args:
        keypoints: (P, 133, 2) detected people.
        scores:    (P, 133) keypoint confidences.
        prev_hip_mid: previous frame's chosen hip midpoint, or None.
        ref_pelvis: (2,) projected Vicon pelvis for this frame, or None/NaN.
        max_match_px: rejection radius for the ref_pelvis match.

    Returns:
        (kp, sc, hip_mid, match_dist)
        kp/sc are None when the frame is rejected (no person matches the
        reference). match_dist is the pelvis distance in px, or NaN when no
        reference was used.
    """
    body_conf = scores[:, [L_SHO, R_SHO, L_HIP, R_HIP, L_KNE, R_KNE]].mean(axis=1)
    hip_mids = (keypoints[:, L_HIP] + keypoints[:, R_HIP]) / 2.0

    match_dist = float("nan")
    if ref_pelvis is not None and np.all(np.isfinite(ref_pelvis)):
        dist = np.linalg.norm(hip_mids - np.asarray(ref_pelvis)[None, :], axis=1)
        idx = int(np.argmin(dist))
        match_dist = float(dist[idx])
        if match_dist > max_match_px:
            # Nearest detection is too far from where Vicon says the subject is:
            # the subject was missed and only bystanders were detected. Emitting
            # the nearest bystander here would be worse than emitting nothing.
            return None, None, None, match_dist
    elif prev_hip_mid is not None:
        dist = np.linalg.norm(hip_mids - prev_hip_mid[None, :], axis=1)
        idx = int(np.argmin(dist))
    else:
        idx = int(np.argmax(body_conf))

    kp, sc = keypoints[idx], scores[idx]
    return kp, sc, hip_mids[idx], match_dist


def map_to_21_joints(
    kp: np.ndarray, sc: np.ndarray, conf_threshold: float, t_t10: float, t_l45: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map one person's 133 COCO-WholeBody keypoints to this project's 21-joint layout.

    Returns:
        joints_2d: (21, 2) pixel coords — NaN wherever `valid` is False.
        valid:     (21,) bool — the VISIBILITY mask. True only where the
            original RTMPose detection (or, for a derived joint, all of its
            source keypoints) passed conf_threshold. Never set True by
            anything downstream — a joint later filled by interpolation must
            stay False here so it isn't mistaken for an original detection.
        conf:      (21,) float — RAW confidence, computed regardless of
            conf_threshold, so a below-threshold joint still carries a real
            number here instead of being indistinguishable from "never
            measured". For a derived joint (HD/C7/PELCEN/T10/L45) this is the
            mean of its source keypoints' scores.
    """
    joints_2d = np.full((len(JOINT_NAMES), 2), np.nan)
    valid = np.zeros(len(JOINT_NAMES), dtype=bool)
    conf = np.full(len(JOINT_NAMES), np.nan)

    def _get(idx: int) -> tuple[np.ndarray, bool, float]:
        ok = sc[idx] >= conf_threshold
        return (kp[idx] if ok else np.full(2, np.nan)), ok, float(sc[idx])

    for name, idx in DIRECT_JOINTS.items():
        j = JOINT_NAMES.index(name)
        joints_2d[j], valid[j], conf[j] = _get(idx)

    # ── HEAD ─────────────────────────────────────────────────────────────────
    # HD = mean of the eye/ear keypoints, NOT the nose.
    #
    # The Vicon HD this is trained against is the centroid of four head markers
    # (LFHD/RFHD/LBHD/RBHD), i.e. a point inside the skull. The nose is anterior
    # and inferior to that, and — crucially — the nose-to-centroid offset is NOT
    # constant: it sweeps as the head yaws, while the centroid barely moves. A
    # lifter cannot learn an offset that depends on unobserved head orientation,
    # so the nose injects head-pose noise straight into HD, and HD drives the
    # neck angle (neck_vec = HD - C7) feeding REBA's neck score.
    #
    # Averaging left/right eyes and ears cancels that yaw dependence by symmetry
    # and lands near the head centre. This matches the documented 21-joint spec.
    #
    # [LIMITATION] The yaw cancellation only holds while BOTH sides are visible.
    # With the head turned away, only the near-side eye+ear may pass threshold
    # and HD is biased laterally by ~half a head width — still far better than
    # the nose (which is biased anteriorly by more, and varies with yaw), but not
    # equivalent to the 4-marker Vicon centroid. Frames below MIN_HEAD_KPTS emit
    # NaN rather than a guess.
    head_scores = [float(sc[i]) for i in HEAD_KPTS]
    head_pts = [kp[i] for i in HEAD_KPTS if sc[i] >= conf_threshold]
    j_hd = JOINT_NAMES.index("HD")
    conf[j_hd] = float(np.mean(head_scores))
    if len(head_pts) >= MIN_HEAD_KPTS:
        joints_2d[j_hd] = np.mean(np.stack(head_pts, axis=0), axis=0)
        valid[j_hd] = True
    else:
        joints_2d[j_hd] = np.full(2, np.nan)
        valid[j_hd] = False

    (l_sho, l_sho_ok, l_sho_c), (r_sho, r_sho_ok, r_sho_c) = _get(L_SHO), _get(R_SHO)
    (l_hip, l_hip_ok, l_hip_c), (r_hip, r_hip_ok, r_hip_c) = _get(L_HIP), _get(R_HIP)
    sho_ok, hip_ok = l_sho_ok and r_sho_ok, l_hip_ok and r_hip_ok
    sho_conf, hip_conf = (l_sho_c + r_sho_c) / 2.0, (l_hip_c + r_hip_c) / 2.0

    c7_proxy = (l_sho + r_sho) / 2.0
    pelcen_proxy = (l_hip + r_hip) / 2.0

    for name, point, ok, c in [("C7", c7_proxy, sho_ok, sho_conf),
                               ("PELCEN", pelcen_proxy, hip_ok, hip_conf)]:
        j = JOINT_NAMES.index(name)
        joints_2d[j] = point if ok else np.full(2, np.nan)
        valid[j] = ok
        conf[j] = c

    spine_ok = sho_ok and hip_ok
    spine_conf = (sho_conf + hip_conf) / 2.0
    for name, t in [("T10", t_t10), ("L45", t_l45)]:
        j = JOINT_NAMES.index(name)
        joints_2d[j] = (1 - t) * c7_proxy + t * pelcen_proxy if spine_ok else np.full(2, np.nan)
        valid[j] = spine_ok
        conf[j] = spine_conf

    return joints_2d, valid, conf


def run_on_video(
    video_path: Path,
    conf_threshold: float = CONF_THRESHOLD_DEFAULT,
    t_t10: float = DEFAULT_T_T10,
    t_l45: float = DEFAULT_T_L45,
    device: str = "cpu",
    mode: str = "balanced",
    progress_cb: ProgressCb = None,
    progress_range: tuple[int, int] = (0, 100),
    vicon_pelvis_2d: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Run RTMPose-WholeBody over every frame of a video and map to the 21-joint set.

    Args:
        progress_cb: optional (pct, step) callback for job progress reporting.
        progress_range: (lo, hi) — frame progress within this video is mapped to
            this sub-range of the overall job's 0-100 scale (for batch processing).
        vicon_pelvis_2d: optional (N, 2) projected Vicon pelvis per video frame,
            from load_vicon_pelvis_2d(synced_vicon2d_60hz.csv (Vicon-PROJECTED, not RTMPose)). When given, the
            subject is identified by proximity to the true projected pelvis
            instead of by the bystander-prone confidence/nearest-neighbour
            heuristic. Strongly recommended for dataset preparation — see
            select_tracked_person(). Must be frame-aligned with video_path
            (both are per-frame over <task>_rgb.mp4); a length mismatch raises.

    Returns:
        DataFrame with one row per frame: frame, <JOINT>_X/_Y, plus
        `subject_match_px` (distance from the detected hip midpoint to the
        projected Vicon pelvis, NaN when unguided) and `subject_found` (False
        when no detection matched the reference and the row is all-NaN).
    """
    from rtmlib import Wholebody

    wholebody = Wholebody(to_openpose=False, mode=mode, backend="onnxruntime", device=device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    if vicon_pelvis_2d is not None and len(vicon_pelvis_2d) != total_frames:
        raise ValueError(
            f"vicon_pelvis_2d has {len(vicon_pelvis_2d)} rows but "
            f"{video_path.name} has {total_frames} frames. These must be "
            f"frame-aligned — synced_vicon2d_60hz.csv and rgb.mp4 come from the same "
            f"Overlay run, so a mismatch means they are from different runs."
        )

    lo, hi = progress_range
    rows = []
    prev_hip_mid = None
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ref = None
        if vicon_pelvis_2d is not None and frame_idx < len(vicon_pelvis_2d):
            ref = vicon_pelvis_2d[frame_idx]

        keypoints, scores = wholebody(frame)
        match_px, found = float("nan"), False
        valid = np.zeros(len(JOINT_NAMES), dtype=bool)
        conf = np.full(len(JOINT_NAMES), np.nan)
        if keypoints.shape[0] == 0:
            joints_2d = np.full((len(JOINT_NAMES), 2), np.nan)
            prev_hip_mid = None
        else:
            kp, sc, hip_mid, match_px = select_tracked_person(
                keypoints, scores, prev_hip_mid, ref_pelvis=ref)
            if kp is None:
                # Detections exist but none is the subject (all bystanders).
                joints_2d = np.full((len(JOINT_NAMES), 2), np.nan)
                prev_hip_mid = None
            else:
                joints_2d, valid, conf = map_to_21_joints(kp, sc, conf_threshold, t_t10, t_l45)
                prev_hip_mid = hip_mid
                found = True

        row = {"frame": frame_idx}
        for j, name in enumerate(JOINT_NAMES):
            row[f"{name}_X"] = joints_2d[j, 0]
            row[f"{name}_Y"] = joints_2d[j, 1]
            row[f"{name}_conf"] = conf[j]
            row[f"{name}_valid"] = bool(valid[j])
        row["subject_match_px"] = match_px
        row["subject_found"] = found
        rows.append(row)
        frame_idx += 1

        if progress_cb and frame_idx % 10 == 0:
            pct = lo + (hi - lo) * min(frame_idx / total_frames, 1.0)
            progress_cb(int(pct), f"Frame {frame_idx}/{total_frames}")

    cap.release()
    df = pd.DataFrame(rows)

    if vicon_pelvis_2d is not None and len(df):
        d = df["subject_match_px"].to_numpy(dtype=float)
        fin = d[np.isfinite(d)]
        n_missed = int((~df["subject_found"]).sum())
        if fin.size:
            logger.info(
                "Vicon-guided subject selection: median match %.1f px, "
                "p95 %.1f px, max %.1f px; %d/%d frames had no matching "
                "detection.", float(np.median(fin)), float(np.percentile(fin, 95)),
                float(fin.max()), n_missed, len(df))
        if n_missed > 0.10 * len(df):
            logger.warning(
                "%d/%d frames (%.1f%%) had no detection within %.0f px of the "
                "projected Vicon pelvis. Check the overlay alignment and the "
                "detector before trusting this trial.",
                n_missed, len(df), 100.0 * n_missed / len(df), MAX_PELVIS_MATCH_PX)
    return df


# ── Skeleton connectivity (indices into JOINT_NAMES) ────────────────────────────────────
_SKELETON_EDGES_NAMED: list[tuple[str, str]] = [
    ("HD", "C7"), ("C7", "T10"), ("T10", "L45"), ("L45", "PELCEN"),   # head + spine
    ("C7", "RSHO"), ("C7", "LSHO"),                                     # clavicles
    ("RSHO", "RELB"), ("RELB", "RWR"), ("RWR", "RHAND"),               # right arm
    ("LSHO", "LELB"), ("LELB", "LWR"), ("LWR", "LHAND"),               # left arm
    ("PELCEN", "RHIP"), ("PELCEN", "LHIP"),                            # pelvis
    ("RHIP", "RKNE"), ("RKNE", "RANK"), ("RANK", "RTOE"),             # right leg
    ("LHIP", "LKNE"), ("LKNE", "LANK"), ("LANK", "LTOE"),            # left leg
]
SKELETON_EDGES: list[tuple[int, int]] = [
    (JOINT_NAMES.index(a), JOINT_NAMES.index(b)) for a, b in _SKELETON_EDGES_NAMED
]


def draw_skeleton(
    frame: np.ndarray, joints_2d: np.ndarray, valid: np.ndarray, *, labels: bool = False
) -> np.ndarray:
    """Draw the connected 21-joint skeleton + joint dots onto `frame` in place (BGR).

    Bones drawn only when both endpoints exist. Green = both endpoints confident,
    orange = at least one endpoint below the confidence threshold. Dots: green =
    confident, red = below threshold.
    """
    for i, j in SKELETON_EDGES:
        xi, yi = joints_2d[i]
        xj, yj = joints_2d[j]
        if np.isnan(xi) or np.isnan(xj):
            continue
        color = (0, 220, 90) if (valid[i] and valid[j]) else (60, 160, 255)  # BGR
        cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)), color, 2, cv2.LINE_AA)

    for name, (x, y), ok_ in zip(JOINT_NAMES, joints_2d, valid):
        if np.isnan(x):
            continue
        color = (0, 255, 0) if ok_ else (0, 0, 255)  # BGR: green / red
        cv2.circle(frame, (int(x), int(y)), 4, color, -1)
        if labels:
            cv2.putText(frame, name, (int(x) + 6, int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, color, 1, cv2.LINE_AA)
    return frame


def visualize_first_frame(
    video_path: Path, out_path: Path,
    conf_threshold: float = CONF_THRESHOLD_DEFAULT,
    t_t10: float = DEFAULT_T_T10, t_l45: float = DEFAULT_T_L45,
    device: str = "cpu", mode: str = "balanced",
) -> None:
    """Draw the mapped 21-joint skeleton on the video's first frame for QA verification."""
    from rtmlib import Wholebody

    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise FileNotFoundError(f"Could not read first frame of {video_path}")

    wholebody = Wholebody(to_openpose=False, mode=mode, backend="onnxruntime", device=device)
    keypoints, scores = wholebody(frame)
    if keypoints.shape[0] == 0:
        raise RuntimeError("No person detected in first frame — pick a different trial to verify.")

    # Unguided on purpose: this is the QA view used to eyeball whether the
    # detector picks the subject WITHOUT Vicon help.
    kp, sc, _, _ = select_tracked_person(keypoints, scores, None)
    joints_2d, valid, _conf = map_to_21_joints(kp, sc, conf_threshold, t_t10, t_l45)
    draw_skeleton(frame, joints_2d, valid, labels=True)

    cv2.imwrite(str(out_path), frame)


def stream_video_annotated(
    video_path: Path,
    conf_threshold: float = CONF_THRESHOLD_DEFAULT,
    t_t10: float = DEFAULT_T_T10, t_l45: float = DEFAULT_T_L45,
    device: str = "cpu", mode: str = "balanced",
    max_width: int = 960, jpeg_quality: int = 80,
):
    """Generator yielding annotated MJPEG parts — the live 21-joint skeleton drawn on
    each frame, frame by frame at model speed. Writes nothing to disk.

    Loads the model + opens the video lazily on first iteration, and always releases
    the capture in `finally` so a client disconnect (which closes this generator)
    stops inference and frees the file.
    """
    from rtmlib import Wholebody

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    prev_hip_mid = None
    try:
        wholebody = Wholebody(to_openpose=False, mode=mode, backend="onnxruntime", device=device)
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            keypoints, scores = wholebody(frame)
            if keypoints.shape[0] == 0:
                prev_hip_mid = None
            else:
                kp, sc, prev_hip_mid, _ = select_tracked_person(
                    keypoints, scores, prev_hip_mid)
                if kp is not None:
                    joints_2d, valid, _conf = map_to_21_joints(kp, sc, conf_threshold, t_t10, t_l45)
                    draw_skeleton(frame, joints_2d, valid)

            if max_width and frame.shape[1] > max_width:
                scale = max_width / frame.shape[1]
                frame = cv2.resize(frame, (max_width, int(round(frame.shape[0] * scale))))

            ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if ok_enc:
                yield boundary + buf.tobytes() + b"\r\n"
    finally:
        cap.release()


def process_trial(
    task_id: str,
    trial_dir: Path,
    progress_cb: ProgressCb = None,
    progress_range: tuple[int, int] = (0, 100),
    **kwargs,
) -> Path:
    """Run RTMPose-WholeBody on one trial's <task_id>_rgb.mp4.

    Automatically uses <task_id>_synced_vicon2d_60hz.csv — the projected Vicon pelvis
    written by the Overlay stage — to identify the subject, when that file is
    present. This is what keeps a bystander from being tracked instead of the
    worker (see select_tracked_person()). It is opt-OUT rather than opt-in
    because every trial in this dataset has synchronised Vicon, so the guess is
    never necessary during dataset preparation; pass use_vicon_guidance=False
    to reproduce the unguided deployment-style behaviour.

    Args:
        task_id:   composite id used for filename prefixes, e.g. "SUB01_lifting_T01"
            — NOT necessarily trial_dir.name, since trial_dir may now be nested
            (outputs/SUB01/lifting/T01/) where .name is just "T01".
        trial_dir: the resolved folder the files actually live in.

    Returns:
        Path to the written <task_id>_pose2d.csv.

    Raises:
        FileNotFoundError: if the rgb.mp4 is missing.
    """
    video_path = trial_dir / Artifacts.RGB_VIDEO
    if not video_path.exists():
        raise FileNotFoundError(f"No rgb.mp4 found at {video_path}")

    # Verify rgb.mp4's frame count matches synced_60hz.csv's frame IDs BEFORE
    # running RTMPose — run_on_video() writes exactly one row per video frame,
    # so a mismatch here means the two artifacts are from different Overlay
    # runs and pose2d_rtmpose.csv would be silently mis-paired with the wrong
    # 3D frames downstream. Failing here (before loading the model or running
    # any inference) is much cheaper than only catching this later in
    # dataset_export.py, after the RTMPose pass has already been spent.
    synced_csv = trial_dir / Artifacts.SYNCED_60HZ
    if synced_csv.exists():
        synced_frame_ids = read_frame_ids(synced_csv, ("synced_video_frame", "frame"))
        if synced_frame_ids is not None:
            cap = cv2.VideoCapture(str(video_path))
            video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if len(synced_frame_ids) != video_frame_count:
                raise ValueError(
                    f"{task_id}: rgb.mp4 has {video_frame_count} frames but "
                    f"{Artifacts.SYNCED_60HZ} has {len(synced_frame_ids)} frame "
                    f"IDs — stale/inconsistent artifacts (likely Overlay was "
                    f"re-run without also re-running downstream stages, or "
                    f"vice versa). Re-run Overlay before 2D Pose."
                )

    use_guidance = kwargs.pop("use_vicon_guidance", True)
    pelvis_2d = None
    if use_guidance:
        synced2d = trial_dir / Artifacts.SYNCED2D_60HZ
        pelvis_2d = load_vicon_pelvis_2d(synced2d)
        if pelvis_2d is None:
            logger.warning(
                "%s not found — falling back to the UNGUIDED person heuristic, "
                "which can lock onto a bystander. Re-run Overlay with "
                "export_2d_csv=True to enable Vicon-guided selection.",
                synced2d.name)
        else:
            logger.info("Vicon-guided subject selection enabled from %s (%d frames)",
                        synced2d.name, len(pelvis_2d))

    df = run_on_video(video_path, progress_cb=progress_cb,
                      progress_range=progress_range,
                      vicon_pelvis_2d=pelvis_2d, **kwargs)
    out_path = trial_dir / Artifacts.POSE2D_RTMPOSE
    df.to_csv(out_path, index=False)
    return out_path


def process_trials_batch(
    trials: list[tuple[str, Path]],
    progress_cb: ProgressCb = None,
    **kwargs,
) -> list[dict]:
    """Run process_trial() over a list of (task_id, trial_dir) pairs.

    Args:
        trials: list of (task_id, trial_dir) — see process_trial().

    Returns:
        List of {"task_name": str, "status": "ok"|"failed", "output"/"error": str}.
    """
    results = []
    n = len(trials)
    for i, (task_id, trial_dir) in enumerate(trials):
        lo = int(100 * i / n)
        hi = int(100 * (i + 1) / n)
        if progress_cb:
            progress_cb(lo, f"Processing {task_id} ({i + 1}/{n})…")
        try:
            out_path = process_trial(task_id, trial_dir, progress_cb=progress_cb,
                                      progress_range=(lo, hi), **kwargs)
            results.append({"task_name": task_id, "status": "ok", "output": str(out_path)})
        except Exception as e:
            results.append({"task_name": task_id, "status": "failed", "error": str(e)})
    return results


# ── Self-check ───────────────────────────────────────────────────────────────

def _demo() -> None:
    """Assert-based self-check on map_to_21_joints() — no video/model needed."""
    n_kpts = 133
    kp = np.zeros((n_kpts, 2))
    sc = np.zeros(n_kpts)

    # Give every DIRECT_JOINTS source keypoint a distinct position and a
    # confidence ABOVE threshold, except RWR (below threshold).
    conf_threshold = 0.3
    for name, idx in DIRECT_JOINTS.items():
        kp[idx] = [float(idx), float(idx) * 2]
        sc[idx] = 0.9
    sc[R_WR] = 0.1   # below threshold — must come back invalid but not NaN-confidence

    # Head: 3 of 4 keypoints above threshold (>= MIN_HEAD_KPTS=2) -> HD valid.
    for i in HEAD_KPTS:
        kp[i] = [10.0, 10.0]
        sc[i] = 0.8
    sc[R_EAR] = 0.05   # one head kpt below threshold

    joints_2d, valid, conf = map_to_21_joints(kp, sc, conf_threshold, t_t10=0.35, t_l45=0.95)

    assert joints_2d.shape == (21, 2)
    assert valid.shape == (21,)
    assert conf.shape == (21,)

    j_rwr = JOINT_NAMES.index("RWR")
    assert not valid[j_rwr], "RWR is below threshold, must be invalid"
    assert np.isnan(joints_2d[j_rwr]).all(), "invalid joint's coordinate must be NaN"
    assert np.isclose(conf[j_rwr], 0.1), \
        "confidence must be the RAW score even when below threshold, not NaN"

    j_relb = JOINT_NAMES.index("RELB")
    assert valid[j_relb] and np.isclose(conf[j_relb], 0.9)

    j_hd = JOINT_NAMES.index("HD")
    assert valid[j_hd], "3/4 head keypoints pass MIN_HEAD_KPTS=2"
    expected_hd_conf = float(np.mean([sc[i] for i in HEAD_KPTS]))
    assert np.isclose(conf[j_hd], expected_hd_conf), \
        "HD confidence must average ALL 4 head keypoints, not just the passing ones"

    # T10/L45 confidence derives from C7/PELCEN, which derive from shoulders/hips.
    j_c7, j_pelcen = JOINT_NAMES.index("C7"), JOINT_NAMES.index("PELCEN")
    j_t10 = JOINT_NAMES.index("T10")
    assert np.isclose(conf[j_t10], (conf[j_c7] + conf[j_pelcen]) / 2.0)

    print("rtmpose_pipeline self-check OK")


if __name__ == "__main__":
    _demo()
