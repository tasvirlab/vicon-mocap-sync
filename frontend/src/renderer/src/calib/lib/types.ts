// ── Shared TypeScript types ────────────────────────────────────────────────

export interface JobStatus {
  job_id:   string
  status:   'queued' | 'running' | 'completed' | 'failed'
  progress: number
  step:     string
  message?: string
  error?:   string
}

export interface BoardConfig {
  dictionary:         string
  cols:               number
  rows:               number
  square_length_mm:   number
  marker_length_mm:   number
  min_corners:        number
}

export interface CameraMatrix {
  fx: number
  fy: number
  cx: number
  cy: number
  matrix_3x3: number[][]
}

export interface DistortionCoeffs {
  k1: number
  k2: number
  p1: number
  p2: number
  k3: number
  vector: number[]
}

export interface IntrinsicResult {
  job_id:                  string
  rms_px:                  number
  quality:                 string
  num_frames:              number
  image_width:             number
  image_height:            number
  camera_matrix:           CameraMatrix
  distortion_coefficients: DistortionCoeffs
  per_frame_rms:           number[]
  output_dir:              string
  files: {
    intrinsic_params_npz:   string
    intrinsic_params_json:  string
    reprojection_plot:      string
    undistorted_preview:    string
    distortion_map:         string
    calibration_report:     string
  }
}

export interface CameraPosition {
  x: number
  y: number
  z: number
  distance_from_origin_m: number
}

export interface EulerAngles {
  rx: number
  ry: number
  rz: number
}

export interface PerMarkerError {
  marker:   string
  error_px: number
}

export interface ExtrinsicResult {
  job_id:                    string
  method:                    string
  quality:                   string
  rms_px:                    number
  n_inliers:                 number
  n_points:                  number
  rvec:                      number[]
  tvec_mm:                   number[]
  R_matrix:                  number[][]
  world_to_cam_4x4:          number[][]
  cam_to_world_4x4:          number[][]
  camera_position_world_mm:  CameraPosition
  euler_angles_deg:          EulerAngles
  per_marker_errors:         PerMarkerError[]
  output_dir:                string
  files: {
    extrinsic_params_npz:  string
    extrinsic_params_json: string
    overlay_frame:         string
    reprojection_plot:     string
    error_heatmap:         string
    alignment_report:      string
  }
}

export interface ProjectEntry {
  id:         string
  name:       string
  type:       'intrinsic' | 'extrinsic'
  created_at: string
  rms_px:     number
  quality:    string
  output_dir: string
  thumbnail?: string
}

export type QualityLevel = 'excellent' | 'good' | 'acceptable' | 'poor'

/** Backend sends e.g. "EXCELLENT (<0.5 px)"; map it to the four design-system levels. */
export function qualityLevel(q: string): QualityLevel {
  const u = (q || '').toUpperCase()
  if (u.startsWith('EXCELLENT')) return 'excellent'
  if (u.startsWith('GOOD'))      return 'good'
  if (u.startsWith('ACCEPT'))    return 'acceptable'
  return 'poor'
}

/** CSS color (theme token) for a quality string; never color-only, always paired with a label. */
export function qualityColor(q: string): string {
  return { excellent: 'rgb(var(--color-success))', good: 'rgb(var(--color-accent))',
           acceptable: 'rgb(var(--color-warning))', poor: 'rgb(var(--color-danger))' }[qualityLevel(q)]
}
