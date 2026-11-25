"""
pnp_solver.py
=============
Robust PnP solving functions for body-mode extrinsic calibration.

Provides:
  gather_correspondences   -- build 3D<->2D pairs from clicked frames
  per_point_residuals      -- reprojection error per correspondence
  reprojection_rms         -- aggregate RMS over a point set
  solve_at_offset          -- RANSAC + LM PnP at a fixed correspondence set
  find_sync_and_solve      -- grid-search sync offset that minimises RMS
  solve_with_label_search  -- auto-resolve click-slot → Vicon-label mapping
"""

import numpy as np
import cv2

from .vicon_io import interp_marker_at_time


def gather_correspondences(accepted: dict, video_fps: float,
                            vicon: dict, marker_labels: list,
                            offset_s: float) -> tuple:
    """
    Build stacked 3D<->2D correspondences for a given sync offset.

    Args:
        accepted:       {frame_idx: [(x, y), …]} from the click session.
        video_fps:      RGB video frame rate.
        vicon:          Dict from read_vicon_trajectories.
        marker_labels:  Vicon labels in click-slot order.
        offset_s:       Vicon time = video_time + offset_s.

    Returns:
        (obj_pts (M,3), img_pts (M,2), frame_ids (M,))
    """
    obj, img, fids = [], [], []
    for fidx, pts in accepted.items():
        t_vic  = fidx / video_fps + offset_s
        triple, ok = [], True
        for lab in marker_labels:
            p = interp_marker_at_time(
                vicon["data"][lab], vicon["frames"], vicon["fps"], t_vic)
            if np.any(np.isnan(p)):
                ok = False
                break
            triple.append(p)
        if not ok:
            continue
        for p3, p2 in zip(triple, pts):
            obj.append(p3)
            img.append(p2)
            fids.append(fidx)
    return (np.asarray(obj,  dtype=np.float64),
            np.asarray(img,  dtype=np.float64),
            np.asarray(fids, dtype=int))


def per_point_residuals(obj, img, rvec, tvec, K, dist) -> np.ndarray:
    """Return per-point reprojection error in pixels, shape (M,)."""
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return np.sqrt(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))


def reprojection_rms(obj, img, rvec, tvec, K, dist) -> float:
    """Return RMS reprojection error (px) for a point set."""
    return float(np.sqrt(np.mean(
        per_point_residuals(obj, img, rvec, tvec, K, dist) ** 2)))


def solve_at_offset(obj, img, K, dist, ransac_thresh: float = 8.0):
    """
    Robust PnP at a fixed correspondence set.

    RANSAC finds inliers; LM refines on inliers only so a few mis-clicks
    cannot bias the solution.

    Args:
        obj:          (M, 3) float64 — 3D world points (metres).
        img:          (M, 2) float64 — 2D image points (pixels).
        K:            (3, 3) camera matrix.
        dist:         (1, N) distortion coefficients.
        ransac_thresh: RANSAC inlier threshold in pixels.

    Returns:
        (rvec, tvec, inlier_rms, inlier_mask) or None if PnP fails.
    """
    if len(obj) < 6:
        return None
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj, img, K, dist,
        reprojectionError=ransac_thresh, confidence=0.999,
        iterationsCount=300, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or inliers is None or len(inliers) < 6:
        return None
    mask        = np.zeros(len(obj), dtype=bool)
    mask[inliers.flatten()] = True
    rvec, tvec  = cv2.solvePnPRefineLM(obj[mask], img[mask], K, dist, rvec, tvec)
    rms         = reprojection_rms(obj[mask], img[mask], rvec, tvec, K, dist)
    return rvec, tvec, rms, mask


def find_sync_and_solve(accepted: dict, video_fps: float, vicon: dict,
                         marker_labels: list, K, dist,
                         step_s: float = None, verbose: bool = False,
                         offset_hint_s: float = None,
                         hint_window_s: float = 2.0) -> dict:
    """
    Search the video→Vicon time offset that minimises reprojection error.

    Scores by (max inliers, then min RMS) to avoid degenerate few-point
    fits that trivially minimise RMS.

    Args:
        accepted:       {frame_idx: [(x, y), …]}.
        video_fps:      RGB video frame rate.
        vicon:          Dict from read_vicon_trajectories.
        marker_labels:  Vicon labels in click order.
        K:              Camera matrix.
        dist:           Distortion coefficients.
        step_s:         Sync search step in seconds (default: 1/vicon_fps).
        verbose:        Print progress to stdout.
        offset_hint_s:  Wall-clock estimated sync offset (seconds).
                        When provided, search is narrowed to
                        [hint - hint_window_s, hint + hint_window_s] instead
                        of the full clip overlap range.  Dramatically improves
                        accuracy when XCP timestamp sync is available.
        hint_window_s:  Half-width of the search window around the hint (s).
                        Default 2.0 s covers typical camera clock drift.

    Returns:
        Dict with rvec, tvec, rms, offset_s, n_inliers, n_points,
        obj, img, fids, mask.
    """
    if step_s is None:
        step_s = 1.0 / vicon["fps"]

    vic_dur = (len(vicon["frames"]) - 1) / vicon["fps"]
    t_rgb   = np.array([f / video_fps for f in accepted.keys()])
    o_min   = -float(t_rgb.min())
    o_max   = vic_dur - float(t_rgb.max())
    if o_max < o_min:
        raise RuntimeError(
            "RGB clip and Vicon trajectory do not overlap. "
            "Check that the Vicon CSV covers the same recording period."
        )

    # Narrow search window around XCP wall-clock hint
    if offset_hint_s is not None:
        o_min = max(o_min, offset_hint_s - hint_window_s)
        o_max = min(o_max, offset_hint_s + hint_window_s)
        if o_max < o_min:
            # Hint is outside valid overlap — fall back to full range
            o_min = -float(t_rgb.min())
            o_max = vic_dur - float(t_rgb.max())

    offsets    = np.arange(o_min, o_max + step_s, step_s)
    best, best_score = None, None

    for k, off in enumerate(offsets):
        obj, img, fids = gather_correspondences(
            accepted, video_fps, vicon, marker_labels, off)
        sol = solve_at_offset(obj, img, K, dist)
        if sol is None:
            continue
        rvec, tvec, rms, mask = sol
        n_in  = int(mask.sum())
        score = (n_in, -rms)
        if best_score is None or score > best_score:
            best_score = score
            best = {"rvec": rvec, "tvec": tvec, "rms": rms,
                    "offset_s": float(off), "n_inliers": n_in,
                    "n_points": len(obj)}

    if best is None:
        raise RuntimeError(
            "PnP failed at every sync offset. "
            "Check marker labels and ensure at least 3 frames are clicked."
        )

    # Re-gather at winning offset to include arrays in result
    obj, img, fids = gather_correspondences(
        accepted, video_fps, vicon, marker_labels, best["offset_s"])
    rvec, tvec, rms, mask = solve_at_offset(obj, img, K, dist)
    best.update({"rvec": rvec, "tvec": tvec, "rms": rms,
                 "n_inliers": int(mask.sum()), "n_points": len(obj),
                 "obj": obj, "img": img, "fids": fids, "mask": mask})
    return best


def solve_with_label_search(accepted: dict, video_fps: float, vicon: dict,
                              base_labels: list, K, dist,
                              step_s: float = None) -> dict:
    """
    Auto-resolve click-slot → Vicon-label mapping for board-corner mode.

    Tries all 24 permutations at coarse 50 ms step, then refines the
    winning permutation at the fine step.  Used only when markers are
    indistinguishable (board corner mode with 4 identical reflective balls).

    Returns:
        Result dict from find_sync_and_solve with 'label_order' added.
    """
    from itertools import permutations
    coarse = 0.05
    best, best_perm, best_score = None, None, None

    for perm in permutations(range(4)):
        labels = [base_labels[i] for i in perm]
        try:
            res = find_sync_and_solve(
                accepted, video_fps, vicon, labels, K, dist,
                step_s=coarse, verbose=False)
        except RuntimeError:
            continue
        score = (res["n_inliers"], -res["rms"])
        if best_score is None or score > best_score:
            best, best_perm, best_score = res, labels, score

    if best is None:
        raise RuntimeError(
            "Could not solve for any label permutation. "
            "Check that all 4 corners were clicked consistently."
        )

    final = find_sync_and_solve(
        accepted, video_fps, vicon, best_perm, K, dist,
        step_s=step_s, verbose=False)
    final["label_order"] = best_perm
    return final
