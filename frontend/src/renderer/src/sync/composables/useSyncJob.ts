import { ref } from 'vue'
import { openJobStream } from '@renderer/sync/lib/api'
import type { JobStatus } from '@renderer/sync/lib/types'

export function useSyncJob() {
  const jobId     = ref<string | null>(null)
  const jobStatus = ref<JobStatus | null>(null)
  const isRunning = ref(false)
  const error     = ref<string | null>(null)
  let es: EventSource | null = null

  function reset() {
    es?.close()
    es = null
    jobId.value     = null
    jobStatus.value = null
    isRunning.value = false
    error.value     = null
  }

  function startJob(id: string, type: 'sync' | 'overlay' | 'reba' | 'pose2d' | 'dataset') {
    jobId.value     = id
    isRunning.value = true
    error.value     = null
    jobStatus.value = { job_id: id, status: 'queued', progress: 0, step: 'Starting…' }

    es = openJobStream(
      type, id,
      (status) => {
        jobStatus.value = status
        if (status.status === 'failed') {
          error.value = status.error ?? 'Processing failed — check the backend console.'
        }
      },
      () => { isRunning.value = false },
      () => {
        isRunning.value = false
        error.value = 'Lost connection to backend. Is the Python server running?'
      },
    )
  }

  return { jobId, jobStatus, isRunning, error, startJob, reset }
}
