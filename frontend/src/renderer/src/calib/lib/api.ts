import axios from 'axios'
import type { JobStatus, IntrinsicResult, ExtrinsicResult, ProjectEntry } from './types'

const BASE = 'http://127.0.0.1:8765'
const api  = axios.create({ baseURL: BASE, timeout: 30_000 })

// ── Health ────────────────────────────────────────────────────────────────────

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await api.get('/api/health', { timeout: 3000 })
    return res.data?.status === 'ok'
  } catch {
    return false
  }
}

// ── Intrinsic ─────────────────────────────────────────────────────────────────

export async function startIntrinsic(
  videoFile: File,
  config: object,
): Promise<{ job_id: string }> {
  const form = new FormData()
  form.append('video', videoFile)
  form.append('config', JSON.stringify(config))
  const res = await api.post('/api/intrinsic/start', form)
  return res.data
}

export async function startIntrinsicFromImages(
  imageFiles: File[],
  config: object,
): Promise<{ job_id: string }> {
  const form = new FormData()
  for (const f of imageFiles) form.append('images', f)
  form.append('config', JSON.stringify(config))
  const res = await api.post('/api/intrinsic/start', form, { timeout: 120_000 })
  return res.data
}

export async function getIntrinsicResult(jobId: string): Promise<IntrinsicResult> {
  const res = await api.get(`/api/intrinsic/${jobId}/result`)
  return { ...res.data, job_id: jobId }
}

export function intrinsicFileUrl(jobId: string, filename: string): string {
  return `${BASE}/api/intrinsic/${jobId}/file/${filename}`
}

// ── Extrinsic ─────────────────────────────────────────────────────────────────

export async function startExtrinsic(
  videoFile:     File,
  viconFile:     File | null,
  intrinsicFile: File,
  config:        object,
  xcpFile?:      File | null,
): Promise<{ job_id: string }> {
  const form = new FormData()
  form.append('video',     videoFile)
  if (viconFile) form.append('vicon', viconFile)   // omit when using rigid_body_pose mode
  form.append('intrinsic', intrinsicFile)
  if (xcpFile) form.append('xcp', xcpFile)
  form.append('config',    JSON.stringify(config))
  const res = await api.post('/api/extrinsic/start', form, { timeout: 60_000 })
  return res.data
}

export async function getExtrinsicResult(jobId: string): Promise<ExtrinsicResult> {
  const res = await api.get(`/api/extrinsic/${jobId}/result`)
  return { ...res.data, job_id: jobId }
}

export function extrinsicFileUrl(jobId: string, filename: string): string {
  return `${BASE}/api/extrinsic/${jobId}/file/${filename}`
}

// ── Projects ──────────────────────────────────────────────────────────────────

export async function listProjects(): Promise<ProjectEntry[]> {
  const res = await api.get('/api/projects')
  return res.data.projects ?? []
}

export async function deleteProject(id: string): Promise<void> {
  await api.delete(`/api/projects/${id}`)
}

export async function openProjectFolder(id: string): Promise<void> {
  await api.post(`/api/projects/${id}/open`)
}

export function projectThumbnailUrl(id: string): string {
  return `${BASE}/api/projects/${id}/thumbnail`
}

export function projectFileUrl(id: string, filename: string): string {
  return `${BASE}/api/projects/${id}/file/${filename}`
}

// ── Extrinsic Body ────────────────────────────────────────────────────────────

export async function extractBodyFrame(
  videoFile: File,
  frameIdx:  number,
  rotate:    string = 'none',
): Promise<{ frame_b64: string; total_frames: number; fps: number }> {
  const form = new FormData()
  form.append('video',     videoFile)
  form.append('frame_idx', String(frameIdx))
  form.append('rotate',    rotate)
  const res = await api.post('/api/extrinsic-body/frame', form, { timeout: 30_000 })
  return res.data
}

export async function startBodyExtrinsic(
  videoFile:     File,
  viconFile:     File,
  intrinsicFile: File,
  clicks:        Record<string, [number, number][]>,
  jointCount:    number,
  xcpFile?:      File | null,
  viconTz?:      number,
  rotate?:       string,
  subjectId?:    string,
): Promise<{ job_id: string }> {
  const form = new FormData()
  form.append('video',       videoFile)
  form.append('vicon',       viconFile)
  form.append('intrinsic',   intrinsicFile)
  form.append('clicks',      JSON.stringify(clicks))
  form.append('joint_count', String(jointCount))
  if (xcpFile)                form.append('xcp',        xcpFile)
  if (viconTz != null)        form.append('vicon_tz',   String(viconTz))
  if (rotate != null)         form.append('rotate',     rotate)
  if (subjectId)               form.append('subject_id', subjectId)
  const res = await api.post('/api/extrinsic-body/start', form, { timeout: 60_000 })
  return res.data
}

export async function getBodyResult(jobId: string): Promise<any> {
  const res = await api.get(`/api/extrinsic-body/${jobId}/result`)
  return { ...res.data, job_id: jobId }
}

export function bodyFileUrl(jobId: string, filename: string): string {
  return `${BASE}/api/extrinsic-body/${jobId}/file/${filename}`
}

// ── SSE job stream ────────────────────────────────────────────────────────────

export function openJobStream(
  type:      'intrinsic' | 'extrinsic' | 'extrinsic-body',
  jobId:     string,
  onEvent:   (status: JobStatus) => void,
  onDone?:   () => void,
  onError?:  (err: Event) => void,
): EventSource {
  const url = `${BASE}/api/${type}/${jobId}/stream`
  const es  = new EventSource(url)

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

// ── Settings ──────────────────────────────────────────────────────────────────

export interface CalibSettings {
  exports_dir: string
}

export async function getCalibSettings(): Promise<CalibSettings> {
  const res = await api.get('/api/settings')
  return res.data
}

export async function setExportsDir(exportsDir: string): Promise<CalibSettings> {
  const form = new FormData()
  form.append('exports_dir', exportsDir)
  const res = await api.post('/api/settings', form)
  return res.data
}
