"""
skeleton.py
===========
Skeleton definition and drawing helpers for overlay_pipeline.py.

Exports:
  SKELETON    -- list of (joint_A, joint_B) bone connections
  BONE_COLOR  -- BGR colour per bone tuple
  draw_skeleton -- render skeleton onto a frame with alpha blending
  draw_hud      -- render frame/time/visibility info onto a frame
"""

from datetime import timedelta

import cv2
import numpy as np


# ── Skeleton definition ────────────────────────────────────────────────────────
# Labels must match columns in the cleaned Vicon CSV.

SKELETON = [
    # Spine
    ("PELCEN", "L45"),  ("L45", "T10"),   ("T10", "C7"),   ("C7", "HD"),
    # Pelvis to hips
    ("PELCEN", "LHIP"), ("PELCEN", "RHIP"),
    # Left leg
    ("LHIP", "LKNE"),   ("LKNE", "LANK"),  ("LANK", "LTOE"),
    # Right leg
    ("RHIP", "RKNE"),   ("RKNE", "RANK"),  ("RANK", "RTOE"),
    # Left arm
    ("T10", "LSHO"),    ("LSHO", "LELB"),  ("LELB", "LWR"),  ("LWR", "LHAND"),
    # Right arm
    ("T10", "RSHO"),    ("RSHO", "RELB"),  ("RELB", "RWR"),  ("RWR", "RHAND"),
]

# BGR colour per body region
_SPINE = (0,   230, 255)   # yellow
_LLEG  = (50,  200, 255)   # orange
_RLEG  = (50,   80, 255)   # red
_LARM  = (50,  255, 100)   # green
_RARM  = (255,  80,  50)   # blue

BONE_COLOR = {
    ("PELCEN", "L45"): _SPINE, ("L45", "T10"):  _SPINE,
    ("T10",    "C7"):  _SPINE, ("C7",  "HD"):   _SPINE,
    ("PELCEN", "LHIP"): _LLEG, ("LHIP", "LKNE"): _LLEG,
    ("LKNE",  "LANK"): _LLEG,  ("LANK", "LTOE"): _LLEG,
    ("PELCEN", "RHIP"): _RLEG, ("RHIP", "RKNE"): _RLEG,
    ("RKNE",  "RANK"): _RLEG,  ("RANK", "RTOE"): _RLEG,
    ("T10",   "LSHO"): _LARM,  ("LSHO", "LELB"): _LARM,
    ("LELB",  "LWR"):  _LARM,  ("LWR",  "LHAND"): _LARM,
    ("T10",   "RSHO"): _RARM,  ("RSHO", "RELB"): _RARM,
    ("RELB",  "RWR"):  _RARM,  ("RWR",  "RHAND"): _RARM,
}


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw_skeleton(frame, joints_2d: dict, alpha: float = 0.85,
                  bone_thick: int = 3, joint_r: int = 5) -> int:
    """
    Draw colour-coded skeleton onto frame using alpha blending (in-place).

    Args:
        frame:      BGR image — modified in-place.
        joints_2d:  {label: (2,) pixel coords, possibly NaN}.
        alpha:      Blend factor for the skeleton layer (0–1).
        bone_thick: Line thickness for bones.
        joint_r:    Circle radius for joint dots.

    Returns:
        Number of joints with valid in-frame projections.
    """
    H, W = frame.shape[:2]

    def _in(pt):
        return (not np.any(np.isnan(pt))
                and 0 <= int(pt[0]) < W and 0 <= int(pt[1]) < H)

    overlay = frame.copy()
    n_vis   = 0

    for (j1, j2) in SKELETON:
        pt1 = joints_2d.get(j1)
        pt2 = joints_2d.get(j2)
        if pt1 is not None and pt2 is not None and _in(pt1) and _in(pt2):
            col = BONE_COLOR.get((j1, j2), (200, 200, 200))
            cv2.line(overlay,
                     (int(round(pt1[0])), int(round(pt1[1]))),
                     (int(round(pt2[0])), int(round(pt2[1]))),
                     col, bone_thick, cv2.LINE_AA)

    for label, pt in joints_2d.items():
        if _in(pt):
            n_vis += 1
            cx, cy = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(overlay, (cx, cy), joint_r + 1, (0, 0, 0),   -1, cv2.LINE_AA)
            cv2.circle(overlay, (cx, cy), joint_r,     (255, 255, 255), -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    return n_vis


def draw_hud(frame, video_frame: int, vicon_time_s: float,
             n_vis: int, total: int, video_fps: float, vicon_fps: float,
             in_range: bool):
    """
    Draw HUD text: frame numbers, timestamps, joint visibility count.

    Args:
        frame:        BGR image — modified in-place.
        video_frame:  0-indexed video frame number.
        vicon_time_s: Vicon time in seconds from XCP START_TIME.
        n_vis:        Number of joints currently visible in frame.
        total:        Total number of joints in the model.
        video_fps:    Video frame rate (Hz).
        vicon_fps:    Vicon sample rate (Hz).
        in_range:     Whether vicon_time_s falls within the cleaned data range.
    """
    def _fmt(t_s: float) -> str:
        td      = timedelta(seconds=max(0.0, t_s))
        total_s = int(td.total_seconds())
        ms      = int((td.total_seconds() - total_s) * 1000)
        return f"{total_s // 60:02d}:{total_s % 60:02d}.{ms:03d}"

    status = "IN RANGE" if in_range else "OUT OF RANGE"
    lines  = [
        f"Video  F:{video_frame:05d}  {_fmt(video_frame / video_fps)}",
        f"Vicon  T:{vicon_time_s:8.3f}s  {_fmt(vicon_time_s)}  {status}",
        f"Joints: {n_vis}/{total} visible",
    ]
    for i, line in enumerate(lines):
        y = 34 + 30 * i
        cv2.putText(frame, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.80, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.80, (200, 255, 200), 1, cv2.LINE_AA)
