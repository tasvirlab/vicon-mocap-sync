"""
face_blur.py
=============
Detect and blur every face in a video frame-stably, not frame-by-frame.

Raw per-frame face detection (MediaPipe) flickers: a head turn, a shadow, or
a camera shake can drop detection confidence for a frame or two, making the
blur box disappear and reappear — jarring in the output video. This module
fixes that with a two-pass pipeline:

  Pass 1 (build_face_tracks): detect faces in every frame, feed them through
    a ByteTrack multi-object tracker (supervision.ByteTrack) so each face
    keeps a stable ID across short detection gaps (lost_track_buffer below
    bridges roughly max_gap_seconds of consecutive misses). Each track's box
    sequence is then linearly interpolated across remaining gaps (up to the
    same limit — longer gaps are left as a real disappearance, not papered
    over), expanded by PAD_FRAC to cover the full head (raw boxes hug
    eyes/nose/mouth tightly), and smoothed with a moving-average filter so
    box edges don't jitter frame to frame.

  Pass 2 (apply): re-decode the video and blur each frame using the
    precomputed, smoothed per-frame boxes from pass 1 — no detection runs
    in this pass.
"""
from __future__ import annotations

import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_full_range/float16/1/blaze_face_full_range.tflite"
)
_MODEL_PATH = Path(__file__).parent.parent / "models_cache" / "blaze_face_full_range.tflite"

PAD_FRAC = 0.20            # expand each detected box by 20% per side to cover the full head
SMOOTH_WINDOW = 5          # moving-average window (frames) over box coordinates
MIN_DETECTION_CONFIDENCE = 0.5

ProgressCb = Optional[Callable[[int, str], None]]

_detector = None


def _load_detector():
    global _detector
    if _detector is not None:
        return _detector

    import mediapipe as mp

    if not _MODEL_PATH.exists():
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)

    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(_MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    )
    _detector = mp.tasks.vision.FaceDetector.create_from_options(options)
    return _detector


def _detect_raw_boxes(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run the detector on one frame. Returns (xyxy (N,4) float32, confidence (N,)
    float32), unpadded, in this frame's pixel coordinates."""
    import mediapipe as mp

    detector = _load_detector()
    h, w = frame.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_image)

    boxes, confs = [], []
    for det in result.detections:
        box = det.bounding_box
        x1, y1 = max(0, box.origin_x), max(0, box.origin_y)
        x2 = min(w, box.origin_x + box.width)
        y2 = min(h, box.origin_y + box.height)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        confs.append(det.categories[0].score if det.categories else MIN_DETECTION_CONFIDENCE)

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.array(boxes, dtype=np.float32), np.array(confs, dtype=np.float32)


def _pad_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[float, float, float, float]:
    pad_x, pad_y = (x2 - x1) * PAD_FRAC, (y2 - y1) * PAD_FRAC
    return (max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y))


def _blur_box(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> None:
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if x2 <= x1 or y2 <= y1:
        return
    k = max(15, min(x2 - x1, y2 - y1) // 2)
    k = k + 1 if k % 2 == 0 else k  # GaussianBlur kernel must be odd
    roi = frame[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average along axis 0. Shrinks the window near the
    edges instead of zero-padding, so box sizes don't shrink at track ends."""
    n = arr.shape[0]
    half = window // 2
    out = np.empty_like(arr)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = arr[lo:hi].mean(axis=0)
    return out


def _interpolate_and_smooth(
    by_frame: dict[int, tuple[float, float, float, float]], max_gap_frames: int,
) -> dict[int, tuple[float, float, float, float]]:
    """Fill gaps up to max_gap_frames by linear interpolation between the
    nearest detections before/after, then moving-average smooth the
    coordinates. Gaps longer than max_gap_frames are left unfilled — that's
    treated as a real disappearance (e.g. the person left frame), not a
    detector hiccup."""
    known = sorted(by_frame)
    if not known:
        return {}

    filled = dict(by_frame)
    for f0, f1 in zip(known, known[1:]):
        gap = f1 - f0
        if gap <= 1 or gap - 1 > max_gap_frames:
            continue
        b0, b1 = np.array(by_frame[f0]), np.array(by_frame[f1])
        for f in range(f0 + 1, f1):
            alpha = (f - f0) / gap
            filled[f] = tuple((1 - alpha) * b0 + alpha * b1)

    frames_sorted = sorted(filled)
    boxes_arr = np.array([filled[f] for f in frames_sorted], dtype=np.float64)
    smoothed = _moving_average(boxes_arr, SMOOTH_WINDOW)
    return {f: tuple(smoothed[i]) for i, f in enumerate(frames_sorted)}


def build_face_tracks(
    video_path: Path,
    rotate_fn: Callable[[np.ndarray], np.ndarray],
    max_gap_seconds: float = 1.0,
    progress_cb: ProgressCb = None,
) -> dict[int, list[tuple[float, float, float, float]]]:
    """Pass 1: detect + track faces across the whole video.

    Args:
        rotate_fn: must apply the SAME rotation the caller's pass 2 will use
            — otherwise boxes won't line up with the frames they get blurred
            onto. Typically `lambda f: rotate_frame(f, rotate)`.

    Returns:
        {frame_idx: [(x1, y1, x2, y2), ...]} — smoothed, padded, gap-filled
        boxes for every tracked face, ready to blur directly in pass 2.
    """
    import supervision as sv

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    lost_track_buffer = max(1, round(fps * max_gap_seconds))
    tracker = sv.ByteTrack(lost_track_buffer=lost_track_buffer, frame_rate=fps)

    raw_tracks: dict[int, dict[int, tuple[float, float, float, float]]] = defaultdict(dict)
    frame_size: Optional[tuple[int, int]] = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = rotate_fn(frame)
        if frame_size is None:
            frame_size = (frame.shape[1], frame.shape[0])

        boxes, confs = _detect_raw_boxes(frame)
        tracked = tracker.update_with_detections(sv.Detections(xyxy=boxes, confidence=confs))
        for box, tid in zip(tracked.xyxy, tracked.tracker_id):
            raw_tracks[int(tid)][frame_idx] = tuple(float(v) for v in box)

        frame_idx += 1
        if progress_cb and frame_idx % 30 == 0:
            progress_cb(int(100 * frame_idx / total_frames), f"Detecting faces {frame_idx}/{total_frames}…")

    cap.release()
    if frame_size is None:
        return {}

    w, h = frame_size
    max_gap_frames = max(1, round(fps * max_gap_seconds))

    per_frame: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for by_frame in raw_tracks.values():
        for fidx, box in _interpolate_and_smooth(by_frame, max_gap_frames).items():
            per_frame[fidx].append(_pad_box(*box, w, h))

    return per_frame


def apply(
    frame: np.ndarray, frame_idx: int, per_frame_boxes: dict[int, list[tuple[float, float, float, float]]],
) -> np.ndarray:
    """Pass 2: blur frame using the precomputed boxes from build_face_tracks()
    for this frame_idx, in place. No detection runs here."""
    for x1, y1, x2, y2 in per_frame_boxes.get(frame_idx, []):
        _blur_box(frame, x1, y1, x2, y2)
    return frame
