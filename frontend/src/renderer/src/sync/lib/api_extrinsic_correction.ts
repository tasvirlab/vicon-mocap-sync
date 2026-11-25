// Manual spatial ("Spatial Fine-tune") correction — separate file rather than
// growing api.ts further (already over this project's file-size ceiling).
import axios from 'axios'

const BASE = 'http://127.0.0.1:8766'
const api  = axios.create({ baseURL: BASE, timeout: 15_000 })

export interface ExtrinsicDelta {
  dx: number; dy: number; dz: number
  droll: number; dpitch: number; dyaw: number
}

export const ZERO_EXTRINSIC_DELTA: ExtrinsicDelta = {
  dx: 0, dy: 0, dz: 0, droll: 0, dpitch: 0, dyaw: 0,
}

export async function saveExtrinsicCorrection(
  taskName: string,
  delta: ExtrinsicDelta,
): Promise<{ saved: boolean; rvec: number[]; tvec_m: number[] }> {
  const form = new FormData()
  form.append('dx', String(delta.dx))
  form.append('dy', String(delta.dy))
  form.append('dz', String(delta.dz))
  form.append('droll', String(delta.droll))
  form.append('dpitch', String(delta.dpitch))
  form.append('dyaw', String(delta.dyaw))
  const res = await api.post(
    `/api/extrinsic-correction/${taskName}/save`, form, { timeout: 15_000 })
  return res.data
}
