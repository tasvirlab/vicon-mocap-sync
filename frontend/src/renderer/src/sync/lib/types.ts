// ── Shared TypeScript types — ViconSync Studio v2 ──────────────────────────────

export interface JobStatus {
  job_id:   string
  status:   'queued' | 'running' | 'completed' | 'failed'
  progress: number
  step:     string
  message?: string
  error?:   string
}

// ── Sync result (wall-clock + optional cross-correlation) ──────────────────────

export interface SyncResult {
  job_id:                        string
  task_name:                     string
  sync_method:                   string   // 'wall_clock_timestamps' | 'cross_correlation_refined'
  accuracy_note:                 string
  xcp_auto_note?:                string   // set when XCP was omitted and auto-matched from the raw Vicon folder
  vicon_fps:                     number
  video_fps:                     number
  sync_offset_s:                 number
  manual_frame_offset:           number
  first_valid_raw_vicon_frame:   number
  last_valid_raw_vicon_frame:    number
  first_valid_vicon_time_s:      number
  video_at_vicon_start_s:        number
  overlap:                       boolean
  vicon_capture_start_local_iso: string
  video_creation_time_utc_iso:   string
  video_creation_time_source:    string
  timezone_offset_hours:         number
  video_rotation:                string
  output_dir:                    string
  sync_json_path:                string
  has_diagnostic_png:            boolean
  processing_time_s:             number
  // Clock correction (h) that would land video frame 0 at the start of valid
  // Vicon data, solved directly from the wall-clock timestamps — see
  // core/sync_pipeline.py. Timezone is not involved: it cancels out of the
  // offset math, so only this field can fix a "no overlap" error.
  suggested_clock_correction_hours: number
  // Cross-correlation refinement (present when the refine step ran)
  refine_confidence?:            number
  refine_conf_label?:            'HIGH' | 'MEDIUM' | 'LOW'
  refine_signal_used?:           string
  refine_correction_applied?:    boolean
  refine_correction_ms?:         number
  refine_all_signals?:           Record<string, { confidence?: number; lag_ms?: number }>
  needs_review?:                 boolean
}

// ── Overlay result ─────────────────────────────────────────────────────────────

export interface OverlayResult {
  job_id:              string
  task_name:           string
  overlay_video:       string
  plain_video?:        string | null   // undistorted RGB, no skeleton — for training
  frame_mapping_csv:   string
  synced_csv:          string
  synced2d_csv?:       string | null   // camera-projected 2D pixel joints, same frames as synced_csv
  video_fps?:          number          // real measured fps this trial was exported at —
                                       // "synced_60hz.csv" is a fixed artifact name, not a
                                       // literal rate; some trials are genuinely ~30fps
  has_diagnostic_png:  boolean
  total_video_frames:  number
  trimmed_frames:      number
  trimmed_duration_s:  number
  full_skeleton_pct:   number
  output_dir:          string
  processing_time_s:   number
}

// ── Projects ───────────────────────────────────────────────────────────────────

export interface ProjectEntry {
  id:                 string   // task_name (composite "SUB01_lifting_T01" or legacy flat name)
  name:               string
  created_at:         string
  sync_method?:       string
  sync_offset_s?:     number
  video_rotation?:    string
  video_fps?:         number   // real measured fps — some trials (e.g. SUB02/03) are
                                // ~30fps, not the ~60fps most trials use; shown in the UI
                                // so the fixed "synced_60hz.csv" artifact name is never
                                // mistaken for a literal rate
  has_overlay:        boolean
  has_synced_csv:     boolean
  has_diagnostic_png: boolean
  output_dir:         string
  subject_id?:        string | null   // e.g. "SUB01" — null for legacy/unassigned
  task?:               string | null   // e.g. "lifting"
  trial?:              string | null   // e.g. "T01"
}

// ── Preview / task info ────────────────────────────────────────────────────────

export interface TaskInfo {
  task_name:           string
  total_frames:        number
  video_fps:           number
  vicon_fps:           number
  manual_frame_offset: number
  sync_offset_s:       number
  sync_method:         string
}

export interface PreviewResult {
  image_b64:    string   // base64 JPEG
  in_range:     boolean
  vicon_time_s: number
  total_frames: number
  video_fps:    number
}

// ── REBA result ────────────────────────────────────────────────────────────────

export interface StatSummary {
  mean:   number
  min:    number
  max:    number
  median: number
  std:    number
}

export interface RebaResult {
  job_id:         string
  task_name:      string
  n_frames:       number
  params:         { load: number; coupling: number; activity: number; unilateral: boolean }
  reba_summary: {
    n_frames:          number
    mean:              number
    median:            number
    max:               number
    min:               number
    std:               number
    peak_frame:        number
    risk_distribution: Record<string, number>
    peak_score_a:      number
    peak_score_b:      number
    peak_trunk:        number
    peak_neck:         number
    peak_legs:         number
    peak_upper_arm:    number
    peak_lower_arm:    number
    peak_wrist:        number
  }
  component_stats: Record<string, StatSummary>
  angle_summary:   Record<string, StatSummary>
  peak_detail:     Record<string, number | string>
  output_dir:      string
  processing_time_s: number
}

// ── Pose2D (RTMPose-WholeBody) result ───────────────────────────────────────────

export interface TrialListEntry {
  task_name:       string
  has_pose2d?:     boolean        // pose2d task list
  has_synced_csv?: boolean        // dataset task list
  subject_id?:     string | null  // dataset task list
}

export interface Pose2DTrialResult {
  task_name: string
  status:    'ok' | 'failed'
  output?:   string
  error?:    string
}

export interface Pose2DResult {
  job_id:        string
  n_trials:      number
  n_succeeded:   number
  n_failed:      number
  trials:        Pose2DTrialResult[]
  params: {
    conf_threshold: number
    t_t10: number
    t_l45: number
    device: string
    mode: string
  }
  processing_time_s: number
}

export interface Pose2DVisualizeResult {
  image_b64: string
}

// ── Dataset export result ───────────────────────────────────────────────────────

export interface SkippedTrial {
  task_name: string
  reason:    string
}

export interface DatasetExportResult {
  job_id:             string
  export_dir:         string
  timestamp:          string
  train_npz:          string
  test_npz:           string
  visualize_png:      string | null
  calibrated_t_t10:   number
  calibrated_t_l45:   number
  train_subjects:     string[]
  test_subjects:      string[]
  n_train_trials:     number
  n_test_trials:      number
  n_train_windows:    number
  n_test_windows:     number
  window_size:        number
  step_size:          number
  skipped_trials:     SkippedTrial[]
  joint_names:        string[]
  processing_time_s:  number
}

export interface DatasetExportEntry {
  timestamp:     string
  train_npz:     string
  test_npz:      string
  has_visualize: boolean
}

// ── Helpers ────────────────────────────────────────────────────────────────────

export function syncMethodLabel(method: string): string {
  if (method === 'cross_correlation_refined') return 'Cross-correlation refined'
  if (method === 'wall_clock_timestamps')     return 'Wall-clock timestamps'
  return method
}

export function syncMethodColor(method: string): string {
  if (method === 'cross_correlation_refined') return 'rgb(var(--color-success))'  // refined (best)
  return 'rgb(var(--color-warning))'                            // wall-clock only
}

// ── Analyse (Channel 2 + 3 figure rendering) ──────────────────────────────────

export interface PostureBout {
  start_frame:    number
  end_frame:      number
  duration_s:     number
  mean_angle_deg: number
  peak_angle_deg: number
  /** ISO ANGLE verdict — independent of how long the posture was held. */
  angle_status:    'acceptable' | 'duration_limited' | 'not_recommended'
  /** ISO DURATION verdict — never conflate with angle_status. */
  duration_status: 'within_limit' | 'exceeds_limit' | 'not_applicable'
  iso_limit_s:    number | null
}

export interface PostureSegmentSummary {
  n_bouts:                  number
  total_high_risk_s:        number
  longest_bout_s:           number
  percent_task_time:        number
  entry_threshold_deg:      number
  n_static_holds:           number
  static_evaluated:         boolean
  n_iso_limit_exceeded?:    number
  bouts?:                   PostureBout[]
}

export interface AnalyseResult {
  input_kind: 'angles' | 'joints' | 'dynamics'
  filename:   string
  n_frames:   number
  fps:        number
  task_id:    string
  figures: {
    reba?:             string   // base64 PNG — Channel 1
    posture_duration?: string   // base64 PNG — Channel 2
    trunk_dynamics?:   string   // base64 PNG — Channel 3
  }
  reba?: {
    summary: Record<string, number | string | Record<string, number>>
    params:  { load: number; coupling: number; activity: number; unilateral: boolean }
  }
  warnings: string[]
  posture_duration?: {
    iso_11226_values_verified: boolean
    iso_assessment: { applicable: boolean; reason: string }
    bout_definition: string
    hysteresis: { enter_s: number; exit_s: number; onset_backdated: boolean; offset_backdated: boolean }
    static_criteria?: { min_duration_s?: number; iso_max_load_score?: number; thresholds_frozen?: boolean; evaluated?: boolean }
    segments:   Record<string, PostureSegmentSummary>
  }
  trunk_dynamics?: {
    cutoff_hz:          number | null
    repetition:         Record<string, number | boolean | null>
    marras_predictors:  Record<string, number | null>
    marras_probability: {
      applicable:   boolean
      probability:  number | null
      reason?:      string
      lift_rate_source?: string
    }
  }
}
