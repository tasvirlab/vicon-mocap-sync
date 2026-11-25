"""Pydantic schemas for all API request/response models."""

from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel


# ── Job lifecycle ──────────────────────────────────────────────────────────

class JobStatus(BaseModel):
    job_id: str
    status: str          # queued | running | completed | failed
    progress: int        # 0-100
    step: str            # human-readable current step
    message: str = ""


class JobStarted(BaseModel):
    job_id: str
    message: str = "Job started"


# ── Intrinsic calibration ──────────────────────────────────────────────────

class BoardConfig(BaseModel):
    dictionary: str = "DICT_6X6_250"
    cols: int = 9
    rows: int = 6
    square_length_mm: float = 25.0
    marker_length_mm: float = 18.5
    min_corners: int = 8


class ExtractionConfig(BaseModel):
    sample_interval_frames: int = 5
    blur_threshold: float = 80.0
    max_candidate_frames: int = 400
    min_accepted_frames: int = 15
    target_accepted_frames: int = 50
    diversity_grid_cols: int = 4
    diversity_grid_rows: int = 3


class CalibrationFlags(BaseModel):
    fix_aspect_ratio: bool = False
    zero_tangent_dist: bool = False
    fix_principal_point: bool = False


class IntrinsicRequest(BaseModel):
    board: BoardConfig = BoardConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    flags: CalibrationFlags = CalibrationFlags()
    rms_accept_threshold: float = 0.5
    rms_warn_threshold: float = 1.0


class CameraMatrix(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float
    matrix_3x3: List[List[float]]


class DistortionCoeffs(BaseModel):
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float
    vector: List[float]


class IntrinsicResult(BaseModel):
    job_id: str
    status: str
    rms_px: float
    quality: str
    num_frames: int
    image_width: int
    image_height: int
    camera_matrix: CameraMatrix
    distortion_coefficients: DistortionCoeffs
    per_frame_rms: List[float]
    output_dir: str
    files: dict


# ── Extrinsic calibration ──────────────────────────────────────────────────

class SyncConfig(BaseModel):
    mode: str = "manual"       # manual | auto | timecode
    video_frame: int = 0
    vicon_frame: int = 0
    offset_frames: int = 0


class SolverConfig(BaseModel):
    use_ransac: bool = True
    ransac_reprojection_threshold_px: float = 8.0
    ransac_confidence: float = 0.999
    refine_lm: bool = True
    methods: List[str] = ["ITERATIVE", "EPNP", "SQPNP"]


class ExtrinsicRequest(BaseModel):
    board: BoardConfig = BoardConfig()
    sync: SyncConfig = SyncConfig()
    solver: SolverConfig = SolverConfig()
    rms_warn_px: float = 3.0
    rms_bad_px: float = 8.0


class CameraPosition(BaseModel):
    x: float
    y: float
    z: float
    distance_from_origin_m: float


class EulerAngles(BaseModel):
    rx: float
    ry: float
    rz: float


class PerMarkerError(BaseModel):
    marker: str
    error_px: float


class ExtrinsicResult(BaseModel):
    job_id: str
    status: str
    method: str
    quality: str
    rms_px: float
    n_inliers: int
    n_points: int
    # Frame dimensions used for solvePnP — K is valid for this resolution/orientation
    calibration_frame_width: Optional[int] = None
    calibration_frame_height: Optional[int] = None
    rvec: List[float]
    tvec_mm: List[float]
    R_matrix: List[List[float]]
    world_to_cam_4x4: List[List[float]]
    cam_to_world_4x4: List[List[float]]
    camera_position_world_mm: CameraPosition
    euler_angles_deg: EulerAngles
    per_marker_errors: List[PerMarkerError]
    output_dir: str
    files: dict


# ── Projects ───────────────────────────────────────────────────────────────

class ProjectEntry(BaseModel):
    id: str
    name: str
    type: str              # intrinsic | extrinsic
    created_at: str
    rms_px: float
    quality: str
    output_dir: str
    thumbnail: Optional[str] = None


class ProjectList(BaseModel):
    projects: List[ProjectEntry]
    total: int
