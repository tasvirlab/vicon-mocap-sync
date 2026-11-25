import { ref } from 'vue'

interface CalibHandoff { intrinsics: File | null; extrinsics: File | null }

const handoff = ref<CalibHandoff | null>(null)

export function useCalibHandoff() {
  function set(intrinsics: File | null, extrinsics: File | null) {
    if (intrinsics || extrinsics) handoff.value = { intrinsics, extrinsics }
  }
  function consume(): CalibHandoff | null {
    const v = handoff.value
    handoff.value = null
    return v
  }
  return { set, consume }
}
