import axios from 'axios'
import type {
  JobStatus, SyncResult, OverlayResult, ProjectEntry, TaskInfo, PreviewResult, RebaResult,
  TrialListEntry, Pose2DResult, Pose2DVisualizeResult, DatasetExportResult, DatasetExportEntry,
  AnalyseResult,
} from './types'

const BASE = 'http://127.0.0.1:8766'
const api  = axios.create({ baseURL: BASE, timeout: 30_000 })

// ── Health ─────────────────────────────────────────────────────────────────────

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await api.get('/api/health', { timeout: 3000 })
    return res.data?.status === 'ok'
  } catch {
    return false
  }
}

// ── Settings (data folder location) ─────────────────────────────────────────────

export interface AppSettings {
  data_dir: string
  outputs_dir: string
  temp_dir: string
}

export async function getSettings(): Promise<AppSettings> {
  const res = await api.get('/api/settings')
  return res.data
}

export async function setDataDir(dataDir: string): Promise<AppSettings> {
  const form = new FormData()
  form.append('data_dir', dataDir)
  const res = await api.post('/api/settings', form)
  return res.data
}

/** Opens the OS's native folder-picker dialog on the machine running the backend.
 * No timeout — the dialog can stay open indefinitely while the user browses. */
export async function browseNativeFolder(): Promise<string | null> {
  const res = await api.post('/api/settings/browse-native', null, { timeout: 0 })
  return res.data.path
}

// ── Sync ───────────────────────────────────────────────────────────────────────

export async function startSync(
  videoFile:      File,
  viconFile:      File,
  xcpFile:        File | null,
  config:         object,
  intrinsicsFile: File | null = null,
  extrinsicsFile: File | null = null,
): Promise<{ job_id: string }> {
  const form = new FormData()
  form.append('video',     videoFile)
  form.append('vicon_csv', viconFile)
  // xcp is optional — when omitted, the backend tries to auto-match the
  // same-stem sibling next to the raw Vicon CSV before failing loudly.
  if (xcpFile) form.append('xcp', xcpFile)
  if (intrinsicsFile) form.append('intrinsics', intrinsicsFile)
  if (extrinsicsFile) form.append('extrinsics', extrinsicsFile)
  form.append('config', JSON.stringify(config))
  const res = await api.post('/api/sync/start', form, { timeout: 0 })
  return res.data
}

export async function checkSiblingXcp(
  subject: string, filename: string,
): Promise<{ found: boolean; filename: string | null }> {
  const res = await api.get('/api/sync/check-sibling', { params: { subject, filename } })
  return res.data
}

// Fetches the actual sibling XCP bytes so the caller can drop it into the
// dropzone as a real File — same pattern as Vicon.vue's fallback-CSV auto-fill.
export async function fetchSiblingXcp(subject: string, filename: string): Promise<Blob> {
  const res = await api.get('/api/sync/fetch-sibling', {
    params: { subject, filename }, responseType: 'blob',
  })
  return res.data
}

export async function downloadViconCsv(jobId: string): Promise<Blob> {
  const res = await api.get(`/api/vicon/${jobId}/download`, { responseType: 'blob', timeout: 30_000 })
  return res.data
}

export async function getSyncResult(jobId: string): Promise<SyncResult> {
  const res = await api.get(`/api/sync/${jobId}/result`)
  return res.data
}

export async function reapplyOffset(
  taskName: string,
  manualFrameOffset: number,
): Promise<SyncResult> {
  const res = await api.post(`/api/sync/${taskName}/reapply`, {
    manual_frame_offset: manualFrameOffset,
  })
  return res.data
}

// ── Overlay ────────────────────────────────────────────────────────────────────

export async function startOverlay(
  taskName:          string,
  rotate:            string | null = null,
  exportPlain:       boolean       = true,
  intrinsicsFile:    File | null   = null,
  extrinsicsFile:    File | null   = null,
  blurFacesRgb:      boolean       = false,
  export2dCsv:       boolean       = true,
): Promise<{ job_id: string }> {
  const form = new FormData()
  form.append('task_name',      taskName)
  form.append('export_plain',   String(exportPlain))
  form.append('blur_faces_rgb', String(blurFacesRgb))
  form.append('export_2d_csv',  String(export2dCsv))
  if (rotate)         form.append('rotate',     rotate)
  if (intrinsicsFile) form.append('intrinsics', intrinsicsFile)
  if (extrinsicsFile) form.append('extrinsics', extrinsicsFile)
  const res = await api.post('/api/overlay/start', form, { timeout: 0 })
  return res.data
}

export async function getOverlayResult(jobId: string): Promise<OverlayResult> {
  const res = await api.get(`/api/overlay/${jobId}/result`)
  return res.data
}

export async function getTaskInfo(taskName: string): Promise<TaskInfo> {
  const res = await api.get(`/api/overlay/${taskName}/info`)
  return res.data
}

export async function previewOverlayFrame(
  taskName:                    string,
  videoFrame:                  number,
  manualFrameOffsetOverride:   number | null = null,
  rotate:                      string | null = null,
  // Not-yet-saved manual spatial correction (see lib/api_extrinsic_correction.ts) —
  // applied for this preview only, {} or undefined means "no correction".
  extrinsicDelta?:             Record<string, number>,
): Promise<PreviewResult> {
  const form = new FormData()
  form.append('task_name',   taskName)
  form.append('video_frame', String(videoFrame))
  if (manualFrameOffsetOverride !== null)
    form.append('manual_frame_offset_override', String(manualFrameOffsetOverride))
  if (rotate)
    form.append('rotate', rotate)
  if (extrinsicDelta)
    for (const [k, v] of Object.entries(extrinsicDelta)) form.append(k, String(v))
  const res = await api.post('/api/overlay/preview', form, { timeout: 30_000 })
  return res.data
}

// ── Projects ───────────────────────────────────────────────────────────────────

export async function listProjects(): Promise<ProjectEntry[]> {
  const res = await api.get('/api/projects')
  return res.data.projects ?? []
}

export async function listTaskNames(): Promise<string[]> {
  const res = await api.get('/api/projects/task-names')
  return res.data.task_names ?? []
}

export interface CanonicalTask { value: string; label: string; folder: string }

// The fixed 8-task whitelist (protocol order) for the Task dropdown on the
// trial-creation pages — so a task can only ever be a canonical name.
export async function listCanonicalTasks(): Promise<CanonicalTask[]> {
  const res = await api.get('/api/projects/canonical-tasks')
  return res.data.tasks ?? []
}

export async function deleteProject(taskName: string): Promise<void> {
  await api.delete(`/api/projects/${taskName}`)
}

export async function openProjectFolder(taskName: string): Promise<void> {
  await api.post(`/api/projects/${taskName}/open`)
}

export function projectFileUrl(taskName: string, filename: string): string {
  return `${BASE}/api/projects/${taskName}/file/${filename}`
}

/** URL for a file inside a trial's output folder.
 *
 * Routed through /api/projects/<task_id>/file/<filename>, which resolves the
 * path server-side via task_paths.trial_dir(). It must NOT build the path
 * itself: it used to return `/outputs/<task_id>/<filename>`, a flat layout that
 * has not existed since outputs moved to SUB01/01_lifting/<task_id>/ — so every
 * call 404'd. The server is the only thing that knows the layout; keep it that
 * way so a future layout change touches one file.
 *
 * `filename` may include a subfolder, e.g. `qa/overlay.mp4`.
 */
export function outputFileUrl(taskName: string, filename: string): string {
  return projectFileUrl(taskName, filename)
}

// ── REBA ───────────────────────────────────────────────────────────────────────

export async function startReba(
  taskName:   string,
  load:       number = 0,
  coupling:   number = 0,
  activity:   number = 1,
  unilateral: boolean = false,
): Promise<{ job_id: string }> {
  const form = new FormData()
  form.append('task_name',  taskName)
  form.append('load',       String(load))
  form.append('coupling',   String(coupling))
  form.append('activity',   String(activity))
  form.append('unilateral', String(unilateral))
  const res = await api.post('/api/reba/start', form, { timeout: 120_000 })
  return res.data
}

export async function getRebaResult(jobId: string): Promise<RebaResult> {
  const res = await api.get(`/api/reba/${jobId}/result`)
  return res.data
}

export async function getSavedReba(taskName: string): Promise<RebaResult> {
  const res = await api.get(`/api/reba/${taskName}/saved`)
  return res.data
}

// ── Pose2D (RTMPose-WholeBody) ──────────────────────────────────────────────────

export async function listPose2DTasks(): Promise<TrialListEntry[]> {
  const res = await api.get('/api/pose2d/tasks')
  return res.data.tasks ?? []
}

// T10/L45 spine ratios: keep in sync with DEFAULT_T_T10 / DEFAULT_T_L45 in
// backend/core/rtmpose_pipeline.py (fitted from SUB01 Vicon; N=1, refit on full cohort).
export async function startPose2D(
  taskNames: string[],
  confThreshold = 0.3,
  tT10 = 0.35,
  tL45 = 0.95,
  device: 'cpu' | 'cuda' = 'cuda',
  mode: 'lightweight' | 'balanced' | 'performance' = 'balanced',
): Promise<{ job_id: string }> {
  const form = new FormData()
  form.append('task_names', JSON.stringify(taskNames))
  form.append('conf_threshold', String(confThreshold))
  form.append('t_t10', String(tT10))
  form.append('t_l45', String(tL45))
  form.append('device', device)
  form.append('mode', mode)
  const res = await api.post('/api/pose2d/start', form, { timeout: 0 })
  return res.data
}

export async function getPose2DResult(jobId: string): Promise<Pose2DResult> {
  const res = await api.get(`/api/pose2d/${jobId}/result`)
  return res.data
}

export async function visualizePose2D(
  taskName: string, confThreshold = 0.3, tT10 = 0.35, tL45 = 0.95,
  device: 'cpu' | 'cuda' = 'cuda', mode: 'lightweight' | 'balanced' | 'performance' = 'balanced',
): Promise<Pose2DVisualizeResult> {
  const form = new FormData()
  form.append('task_name', taskName)
  form.append('conf_threshold', String(confThreshold))
  form.append('t_t10', String(tT10))
  form.append('t_l45', String(tL45))
  form.append('device', device)
  form.append('mode', mode)
  const res = await api.post('/api/pose2d/visualize', form, { timeout: 60_000 })
  return res.data
}

/** URL for the live MJPEG skeleton stream — assign to an <img src>. Transient, saves nothing. */
export function pose2dLiveUrl(
  taskName: string, confThreshold = 0.3, tT10 = 0.35, tL45 = 0.95,
  device: 'cpu' | 'cuda' = 'cuda', mode: 'lightweight' | 'balanced' | 'performance' = 'balanced',
): string {
  const q = new URLSearchParams({
    task_name: taskName,
    conf_threshold: String(confThreshold),
    t_t10: String(tT10),
    t_l45: String(tL45),
    device, mode,
    _t: String(Date.now()),   // cache-bust so re-runs always reconnect
  })
  return `${BASE}/api/pose2d/live?${q.toString()}`
}

// ── Dataset export ──────────────────────────────────────────────────────────────

export async function listDatasetTasks(): Promise<TrialListEntry[]> {
  const res = await api.get('/api/dataset/tasks')
  return res.data.tasks ?? []
}

export async function listDatasetExports(): Promise<DatasetExportEntry[]> {
  const res = await api.get('/api/dataset/exports')
  return res.data.exports ?? []
}

export async function startDatasetExport(
  taskNames: string[] | null,
  windowSize = 243,
  stepSize = 81,
  testSubjects: string[] | null = null,
  testFraction = 0.2,
): Promise<{ job_id: string }> {
  const form = new FormData()
  form.append('task_names', JSON.stringify(taskNames))
  form.append('window_size', String(windowSize))
  form.append('step_size', String(stepSize))
  form.append('test_subjects', JSON.stringify(testSubjects))
  form.append('test_fraction', String(testFraction))
  const res = await api.post('/api/dataset/start', form, { timeout: 0 })
  return res.data
}

export async function getDatasetResult(jobId: string): Promise<DatasetExportResult> {
  const res = await api.get(`/api/dataset/${jobId}/result`)
  return res.data
}

export function datasetDownloadUrl(timestamp: string, filename: string): string {
  return `${BASE}/api/dataset/download/${timestamp}/${filename}`
}

// ── SSE job stream ─────────────────────────────────────────────────────────────

export function openJobStream(
  type:    'sync' | 'overlay' | 'reba' | 'pose2d' | 'dataset',
  jobId:   string,
  onEvent: (status: JobStatus) => void,
  onDone?: () => void,
  onError?: (err: Event) => void,
): EventSource {
  const es = new EventSource(`${BASE}/api/${type}/${jobId}/stream`)

  es.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data) as JobStatus
      onEvent(data)
      if (data.status === 'completed' || data.status === 'failed') {
        es.close()
        onDone?.()
      }
    } catch { /* ignore parse errors */ }
  }

  es.onerror = (e) => {
    onError?.(e)
    es.close()
  }

  return es
}

/** Promise wrapper around openJobStream — for sequential batch runs (Overlay,
 * REBA) that await one trial's job before starting the next. */
export function runJobToCompletion(
  type: 'sync' | 'overlay' | 'reba' | 'pose2d' | 'dataset',
  jobId: string,
): Promise<JobStatus> {
  return new Promise((resolve, reject) => {
    openJobStream(
      type, jobId,
      (status) => {
        if (status.status === 'completed') resolve(status)
        else if (status.status === 'failed') reject(new Error(status.error ?? 'Processing failed'))
      },
      undefined,
      () => reject(new Error('Lost connection to backend')),
    )
  })
}

// ── Analyse — Channel 2 + 3 figures from one uploaded CSV ─────────────────────

/** Render the posture-duration and trunk-dynamics figures for an uploaded CSV.
 *  Read-only on the backend: nothing is written to outputs/. Long timeout —
 *  matplotlib rendering of a multi-thousand-frame trial takes a few seconds. */
export async function analyseFigures(
  file: File,
  opts: { dynamicsJson?: File | null; fps?: number | null; enterS?: number; exitS?: number; taskId?: string;
          load?: number; coupling?: number; activity?: number; unilateral?: boolean } = {},
): Promise<AnalyseResult> {
  const form = new FormData()
  form.append('file', file)
  if (opts.dynamicsJson) form.append('dynamics_json', opts.dynamicsJson)
  if (opts.fps)          form.append('fps', String(opts.fps))
  if (opts.enterS != null) form.append('enter_s', String(opts.enterS))
  if (opts.exitS  != null) form.append('exit_s',  String(opts.exitS))
  if (opts.taskId)       form.append('task_id', opts.taskId)
  // Channel 1 task metadata — entered, never inferred from pose.
  if (opts.load       != null) form.append('load', String(opts.load))
  if (opts.coupling   != null) form.append('coupling', String(opts.coupling))
  if (opts.activity   != null) form.append('activity', String(opts.activity))
  if (opts.unilateral != null) form.append('unilateral', String(opts.unilateral))
  const res = await api.post('/api/analyse/figures', form, { timeout: 0 })
  return res.data
}
