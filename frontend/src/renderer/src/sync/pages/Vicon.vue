<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useRouter } from 'vue-router'
import { Database, Play, RotateCcw, Download, CheckCircle, AlertCircle, Info, ArrowRight, SkipForward } from '@lucide/vue'
import Button        from '@renderer/sync/components/ui/Button.vue'
import PageHeader    from '@renderer/sync/components/ui/PageHeader.vue'
import InfoTip       from '@renderer/sync/components/ui/InfoTip.vue'
import DropZone      from '@renderer/sync/components/ui/DropZone.vue'
import ProgressSteps from '@renderer/sync/components/ui/ProgressSteps.vue'
import StatusBadge   from '@renderer/sync/components/ui/StatusBadge.vue'
import MetricsCard   from '@renderer/sync/components/ui/MetricsCard.vue'
import MetricsGrid   from '@renderer/sync/components/ui/MetricsGrid.vue'
import { useSyncJob } from '@renderer/sync/composables/useSyncJob'
import { useC3DHandoff } from '@renderer/sync/composables/useC3DHandoff'
import { openJobStream, listProjects, listTaskNames, listCanonicalTasks, type CanonicalTask } from '@renderer/sync/lib/api'
import { SUBJECTS, buildTaskId, normalizeTrial, incrementTrial } from '@renderer/sync/lib/taskId'
import type { JobStatus } from '@renderer/sync/lib/types'

const { set: setHandoff } = useC3DHandoff()

const BASE = 'http://127.0.0.1:8766'
const router = useRouter()

const c3dFile      = ref<File | null>(null)
const fallbackCsv  = ref<File | null>(null)
const maxGap       = ref(10)

// Subject/Task/Trial compose the output folder path: outputs/SUB01/<task>/T01/.
// Decoupled from the uploaded filename on purpose — relying on consistent file
// naming silently breaks (see: every prior task overwriting Task1_lifting_01).
const subjectId = ref(SUBJECTS[0])
const task      = ref('')
const trial     = ref('T01')
const taskName  = computed(() => buildTaskId(subjectId.value, task.value, trial.value))

// Auto-fill (not just a hint): as soon as a C3D is picked, look for the
// same-stem raw CSV and drop it straight into the Fallback CSV dropzone as a
// real File — so it shows exactly like a manual upload (name, size, remove
// button), not a separate banner next to a dropzone that still looks empty.
//
// csvAutoFilled tracks whether the CURRENT fallbackCsv came from this
// auto-fill or from the user's own hand, so:
//  - changing Subject never clobbers a file the user picked themselves.
//  - clicking the dropzone's remove button "sticks" — it must not be
//    silently re-filled a moment later, or Remove would do nothing.
const csvAutoFilled  = ref(false)
const csvAutoWarning = ref<string | null>(null)   // shown only when nothing matched

async function onFallbackCsvChange(f: File | null) {
  fallbackCsv.value = f
  csvAutoFilled.value = false   // any explicit dropzone interaction is a manual choice
  csvAutoWarning.value = null
}

watch([c3dFile, subjectId], async ([file, subj]) => {
  csvAutoWarning.value = null
  if (!file) { csvAutoFilled.value = false; return }   // reset/next-trial already cleared fallbackCsv itself
  if (fallbackCsv.value && !csvAutoFilled.value) return   // user's own upload — never touch it
  try {
    const checkRes = await fetch(
      `${BASE}/api/vicon/check-sibling?subject=${encodeURIComponent(subj)}&filename=${encodeURIComponent(file.name)}`,
    )
    if (!checkRes.ok) return
    const { found, filename } = await checkRes.json()
    if (!found || !filename) {
      fallbackCsv.value = null
      csvAutoFilled.value = false
      csvAutoWarning.value = 'No matching CSV found next to the raw Vicon export — proceeding without one'
      return
    }
    const fileRes = await fetch(
      `${BASE}/api/vicon/fetch-sibling?subject=${encodeURIComponent(subj)}&filename=${encodeURIComponent(file.name)}`,
    )
    if (!fileRes.ok) return
    const blob = await fileRes.blob()
    fallbackCsv.value = new File([blob], filename, { type: 'text/csv' })
    csvAutoFilled.value = true
  } catch { /* offline — dropzone just stays empty, no crash */ }
})

const taskNameSuggestions = ref<string[]>([])
listTaskNames().then(names => { taskNameSuggestions.value = names }).catch(() => {})

// Fixed 8-task whitelist for the Task dropdown — a task can only be a canonical
// name, so no free-text typo can create a mislocated flat folder.
const canonicalTasks = ref<CanonicalTask[]>([])
listCanonicalTasks().then(t => {
  canonicalTasks.value = t
  if (!task.value && t.length) task.value = t[0].value
}).catch(() => {})

const jobId     = ref<string | null>(null)
const jobStatus = ref<JobStatus | null>(null)
const isRunning = ref(false)
const error     = ref<string | null>(null)
const result    = ref<any | null>(null)
const fetchErr  = ref<string | null>(null)

const existingTaskNames = ref<Set<string>>(new Set())
function refreshExistingTaskNames() {
  listProjects().then(projects => {
    existingTaskNames.value = new Set(projects.map(p => p.id))
  }).catch(() => { /* offline — collision warning just won't show */ })
}
refreshExistingTaskNames()

const taskNameCollision = computed(() =>
  !!task.value && existingTaskNames.value.has(taskName.value),
)

const canRun = computed(() => !!c3dFile.value && !!task.value.trim() && !isRunning.value)
const displayError = computed(() => error.value || fetchErr.value)
const showProgress = computed(() => isRunning.value && !!jobStatus.value)
const showResult   = computed(() => !isRunning.value && !!result.value)
const showEmpty    = computed(() => !isRunning.value && !result.value && !displayError.value)

const relOutputPath = computed(() => {
  const p = result.value?.output_path?.replace(/\\/g, '/') ?? ''
  const idx = p.indexOf('outputs/')
  return idx >= 0 ? p.slice(idx) : p
})

async function handleRun() {
  if (!c3dFile.value) return
  result.value = null
  fetchErr.value = null
  error.value = null
  isRunning.value = true

  trial.value = normalizeTrial(trial.value)

  const form = new FormData()
  form.append('c3d_file', c3dFile.value)
  if (fallbackCsv.value) form.append('raw_nexus_csv', fallbackCsv.value)
  form.append('task_name', taskName.value)
  form.append('max_gap', String(maxGap.value))

  try {
    const res = await fetch(`${BASE}/api/vicon/start`, { method: 'POST', body: form })
    if (!res.ok) {
      const d = await res.json()
      throw new Error(d.detail || 'Request failed')
    }
    const { job_id } = await res.json()
    jobId.value = job_id
    jobStatus.value = { job_id, status: 'queued', progress: 0, step: 'Starting…' }

    const es = new EventSource(`${BASE}/api/vicon/${job_id}/stream`)
    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data) as JobStatus
        jobStatus.value = data
        if (data.status === 'failed') {
          error.value = data.error ?? 'Processing failed'
          isRunning.value = false
          es.close()
        }
        if (data.status === 'completed') {
          isRunning.value = false
          es.close()
          fetchResult(job_id)
        }
      } catch {}
    }
    es.onerror = () => {
      isRunning.value = false
      error.value = 'Lost connection to backend.'
      es.close()
    }
  } catch (err: unknown) {
    isRunning.value = false
    error.value = `Failed: ${err instanceof Error ? err.message : String(err)}`
  }
}

async function fetchResult(id: string) {
  try {
    const res = await fetch(`${BASE}/api/vicon/${id}/result`)
    const data = await res.json()
    result.value = data
    setHandoff(id, taskName.value, data.output_filename)
    refreshExistingTaskNames()
    listTaskNames().then(names => { taskNameSuggestions.value = names }).catch(() => {})
  } catch (e: unknown) {
    fetchErr.value = `Failed to load result: ${e instanceof Error ? e.message : String(e)}`
  }
}

function handleReset() {
  c3dFile.value = null
  fallbackCsv.value = null
  task.value = ''
  trial.value = 'T01'
  jobId.value = null
  jobStatus.value = null
  isRunning.value = false
  error.value = null
  result.value = null
  fetchErr.value = null
}

function downloadCsv() {
  if (!jobId.value) return
  window.open(`${BASE}/api/vicon/${jobId.value}/download`, '_blank')
}

function goToSync() {
  router.push('/sync')
}

// Same Subject+Task, next Trial (T01 -> T02) — the 8 tasks are each recorded
// 3x per subject, so this is the common next action. Explicit button (not
// automatic on completion) so it never yanks the file dropzone out from under
// someone still reviewing the result.
function prepareNextTrial() {
  c3dFile.value = null
  fallbackCsv.value = null
  trial.value = incrementTrial(trial.value)
  jobId.value = null
  jobStatus.value = null
  isRunning.value = false
  error.value = null
  result.value = null
  fetchErr.value = null
}
</script>

<template>
  <div class="p-6 space-y-5 animate-fade-in">

    <!-- Header -->
    <PageHeader title="C3D → Clean CSV" :icon="Database" color="#8B5CF6" :step="1"
      subtitle="Turn a raw Vicon .c3d recording into a clean 21-joint CSV — no manual Nexus export needed.">
      <template #actions>
        <Button v-if="result || isRunning" variant="ghost" size="sm" @click="handleReset">
          <template #icon><RotateCcw :size="13" /></template>
          Reset
        </Button>
      </template>
    </PageHeader>

    <div class="grid grid-cols-5 gap-5">

      <!-- Left panel -->
      <div class="col-span-2 space-y-4">

        <div class="rounded-xl border border-border p-4 space-y-3">
          <p class="text-xs font-semibold text-muted ">Input</p>
          <DropZone label="Vicon C3D file" description="Processed trial .c3d from Nexus"
            accept=".c3d"
            :file="c3dFile" :disabled="isRunning" @update:file="c3dFile = $event" />

          <div class="border-t border-border pt-3 space-y-2">
            <div class="flex items-center justify-between">
              <span class="text-xs text-muted">
                Fallback CSV <span class="text-accent">(optional)</span>
                <span v-if="csvAutoFilled" class="ml-1 text-xs text-accent">· auto-matched</span>
              </span>
              <span class="text-xs text-muted">fills NaN frames from C3D</span>
            </div>
            <DropZone label="Nexus raw CSV export (.csv)"
              description="Direct Nexus 'Export ASCII' file — vicon_clean.py logic runs automatically"
              accept=".csv"
              :file="fallbackCsv" :disabled="isRunning" @update:file="onFallbackCsvChange" />
            <div v-if="csvAutoWarning"
              class="flex items-center gap-1.5 bg-warning/10 border border-warning/30 rounded-lg px-2 py-1.5">
              <Info :size="11" class="text-warning flex-shrink-0" />
              <p class="text-xs text-warning">{{ csvAutoWarning }}</p>
            </div>
            <p class="text-xs text-muted leading-relaxed">
              The backend runs vicon_clean.py internally (Harrington HJC, gap fill, 21 joints),
              then fills any NaN frames from C3D output frame-by-frame.
            </p>
          </div>

          <div class="grid grid-cols-3 gap-2">
            <label>
              <span class="text-xs text-muted">Subject</span>
              <select v-model="subjectId" :disabled="isRunning"
                class="mt-1 w-full bg-surface border border-border rounded-lg px-2 py-1.5 text-xs text-text focus:outline-none focus:border-accent disabled:opacity-50">
                <option v-for="s in SUBJECTS" :key="s" :value="s">{{ s }}</option>
              </select>
            </label>
            <label class="col-span-2">
              <span class="text-xs text-muted">Task</span>
              <select v-model="task" :disabled="isRunning"
                class="mt-1 w-full bg-surface border border-border rounded-lg px-2 py-1.5 text-xs text-text focus:outline-none focus:border-accent disabled:opacity-50">
                <option v-for="t in canonicalTasks" :key="t.value" :value="t.value">{{ t.label }}</option>
              </select>
            </label>
          </div>
          <label>
            <span class="text-xs text-muted">Trial</span>
            <input v-model="trial" placeholder="T01" :disabled="isRunning"
              @blur="trial = normalizeTrial(trial)"
              class="mt-1 w-full bg-surface border border-border rounded-lg px-2 py-1.5 text-xs text-text focus:outline-none focus:border-accent disabled:opacity-50" />
          </label>

          <div class="bg-surface rounded-lg px-3 py-2 border border-border">
            <div class="flex items-center justify-between">
              <span class="text-xs text-muted">Output folder</span>
              <span class="text-xs text-muted font-mono">outputs/{{ subjectId }}/{{ task || '…' }}/</span>
            </div>
            <p class="text-xs text-muted mt-1">File: <span class="font-mono">{{ taskName }}_clean.csv</span> — other trials of this task share the same folder.</p>
            <p v-if="taskNameCollision" class="text-xs text-warning mt-1 flex items-center gap-1">
              <AlertCircle :size="10" />
              "{{ taskName }}" already exists — running will overwrite it.
            </p>
          </div>

          <label>
            <div class="flex justify-between text-xs text-muted mb-1">
              <span>Max gap fill (frames)</span>
              <span class="font-mono text-text">{{ maxGap }}</span>
            </div>
            <input type="range" v-model.number="maxGap" min="0" max="30" step="1"
              :disabled="isRunning" class="w-full" />
            <p class="text-xs text-muted mt-0.5">Linear interpolation for marker gaps ≤ this length.</p>
          </label>
        </div>

        <!-- PiG info -->
        <div class="rounded-xl border border-border p-4 space-y-2">
          <p class="text-xs font-semibold text-muted ">
            Joint Centre Detection
            <InfoTip text="Plug-in Gait (PiG) is Vicon's biomechanical model. Its virtual markers sit at true anatomical joint centres — more accurate than raw skin-surface markers. If PiG outputs aren't found, the app falls back to surface markers + Harrington regression." />
          </p>
          <div class="space-y-1.5 text-xs text-muted">
            <div class="flex items-start gap-2">
              <span class="text-success mt-0.5">●</span>
              <span><strong class="text-text">Hip (RFEP/LFEP)</strong> — Proximal femur = true HJC from PiG</span>
            </div>
            <div class="flex items-start gap-2">
              <span class="text-success mt-0.5">●</span>
              <span><strong class="text-text">Knee (RFEO/LFEO)</strong> — Distal femur = true KJC from PiG</span>
            </div>
            <div class="flex items-start gap-2">
              <span class="text-success mt-0.5">●</span>
              <span><strong class="text-text">Ankle (RTIO/LTIO)</strong> — Distal tibia = true AJC from PiG</span>
            </div>
            <div class="flex items-start gap-2">
              <span class="text-success mt-0.5">●</span>
              <span><strong class="text-text">Elbow (RHUO/LHUO)</strong> — Humerus distal = EJC from PiG</span>
            </div>
            <div class="flex items-start gap-2">
              <span class="text-success mt-0.5">●</span>
              <span><strong class="text-text">Wrist (RRAO/LRAO)</strong> — Radius distal = WJC from PiG</span>
            </div>
            <div class="flex items-start gap-2">
              <span class="text-warning mt-0.5">○</span>
              <span>Falls back to surface markers + Harrington regression if PiG not detected.</span>
            </div>
          </div>
        </div>

        <Button variant="primary" size="lg" :full-width="true"
          :disabled="!canRun" :loading="isRunning" @click="handleRun">
          <template v-if="!isRunning" #icon><Play :size="15" /></template>
          {{ isRunning ? 'Processing C3D…' : 'Process C3D File' }}
        </Button>

        <div v-if="displayError"
          class="flex items-start gap-2 bg-danger/10 border border-danger/30 rounded-xl p-3">
          <AlertCircle :size="14" class="text-danger flex-shrink-0 mt-0.5" />
          <p class="text-xs text-danger leading-relaxed whitespace-pre-wrap">{{ displayError }}</p>
        </div>
      </div>

      <!-- Right panel -->
      <div class="col-span-3 space-y-4">
        <Transition name="pf" mode="out-in">

          <!-- Progress -->
          <div v-if="showProgress" key="prog"
            class="rounded-xl border border-border p-5 space-y-4">
            <div class="flex items-center justify-between">
              <span class="text-xs font-semibold text-muted ">Processing…</span>
              <StatusBadge status="running" />
            </div>
            <ProgressSteps :job-status="jobStatus" />
          </div>

          <!-- Result -->
          <div v-else-if="showResult" key="res" class="space-y-4">

            <!-- Fallback CSV auto-match note (no CSV uploaded, backend searched the raw Vicon folder) -->
            <div v-if="result.fallback_csv_auto_note"
              class="flex items-start gap-2 rounded-xl p-3"
              :class="result.fallback_csv_used
                ? 'bg-accent/10 border border-accent/30'
                : 'bg-warning/10 border border-warning/30'">
              <Info :size="13" class="flex-shrink-0 mt-0.5"
                :class="result.fallback_csv_used ? 'text-accent' : 'text-warning'" />
              <p class="text-xs" :class="result.fallback_csv_used ? 'text-accent' : 'text-warning'">
                {{ result.fallback_csv_auto_note }}
              </p>
            </div>

            <!-- PiG detection banner -->
            <div class="rounded-xl border p-4"
              :class="result.pig_detected
                ? 'bg-success/10 border-success/40'
                : 'bg-warning/10 border-warning/40'">
              <div class="flex items-center gap-2 mb-2">
                <component :is="result.pig_detected ? CheckCircle : Info"
                  :size="14"
                  :class="result.pig_detected ? 'text-success' : 'text-warning'" />
                <span class="text-sm font-semibold"
                  :class="result.pig_detected ? 'text-success' : 'text-warning'">
                  {{ result.pig_detected
                    ? 'Plug-in Gait detected — anatomical joint centres used'
                    : 'No PiG model outputs — surface markers + Harrington used' }}
                </span>
              </div>
              <div v-if="result.pig_outputs_used?.length" class="space-y-0.5">
                <p v-for="line in result.pig_outputs_used" :key="line"
                  class="text-xs text-success/80 font-mono">✓ {{ line }}</p>
              </div>
            </div>

            <!-- Metrics -->
            <MetricsGrid :cols="3">
              <MetricsCard label="Frame rate"   :value="result.frame_rate_hz" unit="Hz" :delay="0" />
              <MetricsCard label="Input frames" :value="result.total_c3d_frames" :delay="0.05" />
              <MetricsCard label="Output frames" :value="result.output_frames" :delay="0.1" />
            </MetricsGrid>
            <MetricsGrid :cols="2">
              <MetricsCard label="First valid frame" :value="result.first_valid_frame" :delay="0.15" />
              <MetricsCard label="Processing" :value="result.processing_time_s?.toFixed(1)" unit="s" :delay="0.2" />
            </MetricsGrid>

            <!-- Fallback CSV merge stats -->
            <div v-if="result.fallback_csv_used && Object.keys(result.fallback_frames_filled ?? {}).length"
              class="bg-accent/5 border border-accent/20 rounded-xl p-3 space-y-1.5">
              <p class="text-xs font-semibold text-accent">✓ Filled from Nexus CSV (C3D was missing these):</p>
              <div class="grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs font-mono">
                <template v-for="(n, joint) in result.fallback_frames_filled" :key="joint">
                  <span class="text-text">{{ joint }}</span>
                  <span class="text-muted">{{ n }} frames recovered</span>
                </template>
              </div>
            </div>
            <div v-else-if="result.fallback_csv_used && !Object.keys(result.fallback_frames_unfillable ?? {}).length && !(result.permanently_nan_joints ?? []).length"
              class="text-xs text-success px-1">
              ✓ Nexus CSV loaded — C3D was complete, no gaps needed filling.
            </div>

            <!-- Joints permanently NaN in both sources -->
            <div v-if="(result.permanently_nan_joints ?? []).length"
              class="bg-destructive/10 border border-destructive/30 rounded-xl p-3 space-y-1.5">
              <p class="text-xs font-semibold text-destructive">
                Permanently absent — missing in both C3D and Nexus CSV:
              </p>
              <p class="text-xs text-muted leading-relaxed">
                These joints could not be recovered from either source.
                Cause: physical marker not placed, not labeled in Nexus, or occluded
                for the entire trial in both recordings.
                Fix in Nexus (label marker, gap-fill, re-export) then re-run.
              </p>
              <div class="flex flex-wrap gap-x-3 gap-y-0.5 text-xs font-mono">
                <span v-for="j in result.permanently_nan_joints" :key="j"
                  class="text-destructive font-semibold">{{ j }}</span>
              </div>
            </div>

            <!-- Joints partially NaN (some frames recovered, some still missing) -->
            <div v-if="(result.partially_nan_joints ?? []).length"
              class="bg-warning/10 border border-warning/30 rounded-xl p-3 space-y-1.5">
              <p class="text-xs font-semibold text-warning">Partially recovered — some frames still NaN:</p>
              <div class="grid grid-cols-3 gap-x-3 gap-y-0.5 text-xs font-mono">
                <template v-for="j in result.partially_nan_joints" :key="j">
                  <span class="text-warning">{{ j }}</span>
                  <span class="text-muted col-span-2">{{ result.nan_pct_per_joint?.[j] }}% NaN remaining</span>
                </template>
              </div>
              <p class="text-xs text-muted">
                Marker was occluded for part of the trial in both sources.
                Fix: open trial in Nexus, gap-fill affected markers, re-export.
              </p>
            </div>

            <!-- Output file -->
            <div class="bg-surface rounded-xl border border-border p-3 flex items-center gap-3">
              <CheckCircle :size="14" class="text-success flex-shrink-0" />
              <div class="flex-1 min-w-0">
                <p class="text-xs font-medium text-text truncate">{{ result.output_filename }}</p>
                <p class="text-xs text-muted truncate font-mono">{{ relOutputPath }}</p>
              </div>
              <Button size="sm" variant="ghost" @click="downloadCsv">
                <template #icon><Download :size="12" /></template>
                Download
              </Button>
            </div>

            <!-- Next step -->
            <div class="bg-accent/5 border border-accent/20 rounded-xl p-3 space-y-2">
              <p class="text-xs font-semibold text-accent mb-2">Next step</p>
              <button
                class="w-full flex items-center justify-center gap-2 bg-accent/10 hover:bg-accent/20 border border-accent/30 text-accent rounded-lg px-3 py-2.5 text-xs font-semibold transition-colors"
                @click="goToSync"
              >
                Go to Sync
                <ArrowRight :size="12" />
              </button>
              <p class="text-xs text-muted mt-1.5">
                Upload this CSV as the "Vicon CSV" input in the Sync page.
              </p>
              <button
                class="w-full flex items-center justify-center gap-2 bg-surface hover:bg-accent/10 border border-border hover:border-accent/30 text-text hover:text-accent rounded-lg px-3 py-2.5 text-xs font-semibold transition-colors"
                @click="prepareNextTrial"
              >
                <SkipForward :size="12" />
                Next Trial ({{ subjectId }} / {{ task }} / {{ incrementTrial(trial) }})
              </button>
              <p class="text-xs text-muted mt-1.5">
                Keeps Subject + Task, advances Trial, clears the file drop zones for the next upload.
              </p>
            </div>
          </div>

          <!-- Empty -->
          <div v-else key="empty"
            class="flex flex-col items-center justify-center h-64 text-center text-muted space-y-3">
            <Database :size="40" class="opacity-20" />
            <div>
              <p class="text-sm font-medium">Ready to process</p>
              <p class="text-xs mt-1">Upload a C3D file and click Process</p>
            </div>
          </div>

        </Transition>
      </div>
    </div>
  </div>
</template>

<style scoped>
.pf-enter-active, .pf-leave-active { transition: opacity 0.2s ease; }
.pf-enter-from, .pf-leave-to       { opacity: 0; }
</style>
