import { ref } from 'vue'
import { openJobStream } from '@renderer/calib/lib/api'
import type { JobStatus } from '@renderer/calib/lib/types'

export function useCalibrationJob() {
  const jobId     = ref<string | null>(null)
  const jobStatus = ref<JobStatus | null>(null)
  const isRunning = ref(false)
  const error     = ref<string | null>(null)
  let es: EventSource | null = null

  function reset() {
    es?.close()
    es              = null
    jobId.value     = null
    jobStatus.value = null
    isRunning.value = false
    error.value     = null
  }

  function startJob(id: string, type: 'intrinsic' | 'extrinsic' | 'extrinsic-body') {
    jobId.value     = id
    isRunning.value = true
    error.value     = null
    jobStatus.value = { job_id: id, status: 'queued', progress: 0, step: 'Starting…' }

    es = openJobStream(
      type, id,
      (status) => {
        jobStatus.value = status
        if (status.status === 'failed') {
          error.value = status.error ?? 'Calibration failed — check the backend console for details.'
        }
      },
      () => { isRunning.value = false },
      () => {
        isRunning.value = false
        error.value = 'Connection to backend lost. Check that the Python server is running.'
      },
    )
  }

  return { jobId, jobStatus, isRunning, error, startJob, reset }
}
