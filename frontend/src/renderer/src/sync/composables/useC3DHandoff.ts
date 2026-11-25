import { ref } from 'vue'

interface C3DHandoff { jobId: string; taskName: string; filename: string }

// module-level — survives page navigation within the SPA
const handoff = ref<C3DHandoff | null>(null)

export function useC3DHandoff() {
  function set(jobId: string, taskName: string, filename: string) {
    handoff.value = { jobId, taskName, filename }
  }
  function consume(): C3DHandoff | null {
    const v = handoff.value
    handoff.value = null
    return v
  }
  return { handoff, set, consume }
}
