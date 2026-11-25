<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { Zap, Play, RotateCcw, AlertCircle, Info, CheckCircle, ArrowRight, Clock, GitMerge, SlidersHorizontal, Film, SkipForward } from '@lucide/vue'
import Button        from '@renderer/sync/components/ui/Button.vue'
import PageHeader    from '@renderer/sync/components/ui/PageHeader.vue'
import DropZone      from '@renderer/sync/components/ui/DropZone.vue'
import ProgressSteps from '@renderer/sync/components/ui/ProgressSteps.vue'
import StatusBadge   from '@renderer/sync/components/ui/StatusBadge.vue'
import MetricsCard   from '@renderer/sync/components/ui/MetricsCard.vue'
import MetricsGrid   from '@renderer/sync/components/ui/MetricsGrid.vue'
import { useSyncJob } from '@renderer/sync/composables/useSyncJob'
import { useC3DHandoff } from '@renderer/sync/composables/useC3DHandoff'
import { useCalibHandoff } from '@renderer/sync/composables/useCalibHandoff'
import CalibrationPicker from '@renderer/sync/components/CalibrationPicker.vue'
import { startSync, getSyncResult, reapplyOffset, downloadViconCsv, listProjects, listTaskNames, listCanonicalTasks, checkSiblingXcp, fetchSiblingXcp, type CanonicalTask } from '@renderer/sync/lib/api'
import { syncMethodLabel, syncMethodColor } from '@renderer/sync/lib/types'
import { SUBJECTS, buildTaskId, splitTaskId, normalizeTrial, incrementTrial } from '@renderer/sync/lib/taskId'
import type { SyncResult } from '@renderer/sync/lib/types'

const ROTATIONS = ['ccw90', 'cw90', '180', 'none']

const router = useRouter()
const { jobId, jobStatus, isRunning, error, startJob, reset: resetJob } = useSyncJob()

// ── File uploads ───────────────────────────────────────────────────────────────
const videoFile      = ref<File | null>(null)
const viconFile      = ref<File | null>(null)
const xcpFile        = ref<File | null>(null)
const intrinsicsFile = ref<File | null>(null)
const extrinsicsFile = ref<File | null>(null)
function onCalibPick(f: { intrinsics: File | null; extrinsics: File | null }) {
  if (f.intrinsics) intrinsicsFile.value = f.intrinsics
  if (f.extrinsics) extrinsicsFile.value = f.extrinsics
}

// ── Subject/Task/Trial — composes the output folder: outputs/SUB01/<task>/T01/ ──
const subjectId = ref(SUBJECTS[0])
const task      = ref('')
const trial     = ref('T01')
const taskName  = computed(() => buildTaskId(subjectId.value, task.value, trial.value))

// Auto-fill (not just a hint): as soon as a Vicon CSV is picked, look for the
// same-stem raw XCP and drop it straight into the dropzone as a real File —
// mirrors Vicon.vue's fallback-CSV auto-fill exactly, same reasoning there
// for why csvAutoFilled/manual-removal tracking matters.
const xcpAutoFilled  = ref(false)
const xcpAutoWarning = ref<string | null>(null)

async function onXcpFileChange(f: File | null) {
  xcpFile.value = f
  xcpAutoFilled.value = false
  xcpAutoWarning.value = null
}

watch([viconFile, subjectId], async ([file, subj]) => {
  xcpAutoWarning.value = null
  if (!file) { xcpAutoFilled.value = false; return }
  if (xcpFile.value && !xcpAutoFilled.value) return   // user's own upload — never touch it
  try {
    const { found, filename } = await checkSiblingXcp(subj, file.name)
    if (!found || !filename) {
      xcpFile.value = null
      xcpAutoFilled.value = false
      xcpAutoWarning.value = 'No matching XCP file found next to the raw Vicon export — upload one manually'
      return
    }
    const blob = await fetchSiblingXcp(subj, file.name)
    xcpFile.value = new File([blob], filename, { type: 'application/xml' })
    xcpAutoFilled.value = true
  } catch { /* offline — dropzone just stays empty, no crash */ }
})

const taskNameSuggestions = ref<string[]>([])
const existingTaskNames   = ref<Set<string>>(new Set())

// A pending C3D→Sync handoff must win over the default task-select below, even
// though the handoff only resolves after onMounted's async CSV download.
// Reading handoff.value here is synchronous, so it can't lose that race.
const { handoff: c3dHandoff, consume: consumeHandoff } = useC3DHandoff()
const hasPendingHandoff = !!c3dHandoff.value

// Fixed 8-task whitelist for the Task dropdown (canonical names only).
const canonicalTasks = ref<CanonicalTask[]>([])
listCanonicalTasks().then(t => {
  canonicalTasks.value = t
  if (!task.value && t.length && !hasPendingHandoff) task.value = t[0].value
}).catch(() => {})
function refreshTaskLists() {
  listTaskNames().then(names => { taskNameSuggestions.value = names }).catch(() => {})
  listProjects().then(projects => {
    existingTaskNames.value = new Set(projects.map(p => p.id))
  }).catch(() => {})
}
refreshTaskLists()

const taskNameCollision = computed(() =>
  !!task.value && existingTaskNames.value.has(taskName.value),
)

// ── Parameters ─────────────────────────────────────────────────────────────────
const timezoneHours      = ref(5.0)
const timeCorrectHours   = ref(0.0)
const viconFps           = ref(100.0)
const manualFrameOffset  = ref(0)
const autoRefine         = ref(true)
const searchRangeS       = ref(4.0)
const videoRotation      = ref(localStorage.getItem('sync_video_rotation') ?? 'ccw90')
watch(videoRotation, (v) => localStorage.setItem('sync_video_rotation', v))
const manualVideoStart   = ref('')
const forceSyncOffset    = ref('')   // manual override if auto-detect fails

// ── Results ────────────────────────────────────────────────────────────────────
const result     = ref<SyncResult | null>(null)
const fetchError = ref<string | null>(null)
const refining   = ref(false)
const showOffset = ref(false)

// ── C3D handoff — auto-fill Vicon CSV from previous C3D page ──────────────────
const { set: setCalibHandoff } = useCalibHandoff()

onMounted(async () => {
  const h = consumeHandoff()
  if (!h) return
  try {
    const blob = await downloadViconCsv(h.jobId)
    viconFile.value = new File([blob], h.filename, { type: 'text/csv' })
    // h.taskName is already a full composite id from the C3D step (e.g.
    // "SUB01_lifting_T01") — decompose it back into the three fields so the
    // UI stays consistent and the user can still see/adjust them. hasPendingHandoff
    // (read synchronously before this handler ran) already stopped the default
    // task-select from claiming task.value, so this always applies now.
    {
      const parts = splitTaskId(h.taskName)
      if (parts) {
        subjectId.value = parts.subjectId
        task.value      = parts.task
        trial.value     = parts.trial
      } else {
        task.value = h.taskName   // legacy/unparseable id — best effort
      }
    }
  } catch { /* silently skip if backend unavailable */ }
})

// Auto-enable refine only when both calib files are present
const canRefine = computed(() => !!intrinsicsFile.value && !!extrinsicsFile.value)
watch(canRefine, (v) => { if (!v) autoRefine.value = false })

watch(() => jobStatus.value?.status, (status) => {
  if (status === 'completed' && jobId.value) {
    fetchError.value = null
    setCalibHandoff(intrinsicsFile.value, extrinsicsFile.value)
    getSyncResult(jobId.value)
      .then(r => {
        result.value = r
        manualFrameOffset.value = r.manual_frame_offset
        refreshTaskLists()
      })
      .catch(err => { fetchError.value = `Failed to load result: ${err?.message}` })
  }
})

// xcp is optional here — if omitted, the backend tries to auto-match the
// same-stem sibling next to the raw Vicon CSV before failing loudly.
const canRun       = computed(() => !!videoFile.value && !!viconFile.value && !!task.value.trim() && !isRunning.value)
const displayError = computed(() => error.value || fetchError.value)
const showProgress = computed(() => isRunning.value && !!jobStatus.value)
const showResult   = computed(() => !isRunning.value && !!result.value)
const showEmpty    = computed(() => !isRunning.value && !result.value && !displayError.value)

async function handleRun() {
  if (!videoFile.value || !viconFile.value || !task.value.trim()) return
  result.value = null
  fetchError.value = null
  showOffset.value = false
  trial.value = normalizeTrial(trial.value)
  try {
    const fso = forceSyncOffset.value.trim()
    const config = {
      task_name:                   taskName.value,
      timezone_hours:              timezoneHours.value,
      video_time_correction_hours: timeCorrectHours.value,
      vicon_fps:                   viconFps.value,
      manual_frame_offset:         manualFrameOffset.value,
      auto_refine:                 autoRefine.value,
      search_range_s:              searchRangeS.value,
      video_rotation:              videoRotation.value,
      manual_video_start_local:    manualVideoStart.value || null,
      force_sync_offset_s:         fso !== '' ? parseFloat(fso) : null,
    }
    const { job_id } = await startSync(
      videoFile.value, viconFile.value, xcpFile.value,
      config, intrinsicsFile.value, extrinsicsFile.value,
    )
    startJob(job_id, 'sync')
  } catch (err: unknown) {
    fetchError.value = `Failed to start: ${err instanceof Error ? err.message : String(err)}`
  }
}

async function handleReapply() {
  if (!result.value) return
  refining.value = true
  try {
    const updated = await reapplyOffset(result.value.task_name, manualFrameOffset.value)
    result.value = { ...result.value, ...updated }
  } catch (e: unknown) {
    fetchError.value = `Reapply failed: ${e instanceof Error ? e.message : String(e)}`
  } finally {
    refining.value = false
  }
}

function handleReset() {
  resetJob()
  videoFile.value = viconFile.value = xcpFile.value = null
  intrinsicsFile.value = extrinsicsFile.value = null
  result.value = fetchError.value = null
  task.value = ''
  trial.value = 'T01'
  forceSyncOffset.value = ''
  showOffset.value = false
}

// Same Subject+Task, next Trial (T01 -> T02) — keeps calibration files (same
// camera setup across trials) but clears the per-trial video/Vicon/XCP inputs
// and any per-trial offset override. Explicit button, not automatic.
function prepareNextTrial() {
  resetJob()
  videoFile.value = viconFile.value = xcpFile.value = null
  result.value = fetchError.value = null
  trial.value = incrementTrial(trial.value)
  forceSyncOffset.value = ''
  showOffset.value = false
}

function goToOverlay() {
  router.push('/overlay')
}
</script>

<template>
  <div class="p-6 space-y-5 animate-fade-in">

    <!-- Header -->
    <PageHeader title="Temporal Sync" :icon="Zap" color="#10B981" :step="2"
      subtitle="Align the Vicon clock to the video clock — wall-clock from XCP + ffprobe, with optional cross-correlation refinement.">
      <template #actions>
        <Button v-if="result || isRunning" variant="ghost" size="sm" @click="handleReset">
          <template #icon><RotateCcw :size="13" /></template>
          Reset
        </Button>
      </template>
    </PageHeader>

    <div class="grid grid-cols-5 gap-5">

      <!-- ── Left panel ──────────────────────────────────────────────────────── -->
      <div class="col-span-2 space-y-4">

        <!-- Required files -->
        <div class="rounded-xl border border-border p-4 space-y-3">
          <p class="text-xs font-semibold text-muted ">Required Files</p>
          <DropZone label="RGB video" description="MP4 · MOV · AVI"
            accept="video/*,.mov"
            :file="videoFile" :disabled="isRunning" @update:file="videoFile = $event" />
          <DropZone label="Vicon CSV (cleaned)" description="Joint_Coordinates CSV from Nexus"
            accept=".csv,.txt"
            :file="viconFile" :disabled="isRunning" @update:file="viconFile = $event" />
          <DropZone :label="xcpAutoFilled ? 'Vicon XCP file (auto-matched)' : 'Vicon XCP file (optional)'"
            description="Capture start time for wall-clock sync — leave empty to auto-match the same-stem file next to the raw Vicon CSV"
            accept=".xcp"
            :file="xcpFile" :disabled="isRunning" @update:file="onXcpFileChange" />
          <div v-if="xcpAutoWarning"
            class="flex items-center gap-1.5 bg-warning/10 border border-warning/30 rounded-lg px-2 py-1.5">
            <Info :size="11" class="text-warning flex-shrink-0" />
            <p class="text-xs text-warning">{{ xcpAutoWarning }}</p>
          </div>
        </div>

        <!-- Optional calibration -->
        <div class="rounded-xl border border-border p-4 space-y-3">
          <div class="flex items-center justify-between">
            <p class="text-xs font-semibold text-muted ">Calibration (for refine)</p>
            <span class="text-xs text-muted">optional</span>
          </div>
          <CalibrationPicker :disabled="isRunning" @pick="onCalibPick" />
          <DropZone label="Intrinsics JSON" description="intrinsic.json from the Intrinsic page"
            accept=".json"
            :file="intrinsicsFile" :disabled="isRunning" @update:file="intrinsicsFile = $event" />
          <DropZone label="Extrinsics JSON" description="extrinsics.json from the Extrinsic page"
            accept=".json"
            :file="extrinsicsFile" :disabled="isRunning" @update:file="extrinsicsFile = $event" />

          <!-- Auto-refine toggle -->
          <label class="flex items-center gap-2.5 cursor-pointer"
            :class="{ 'opacity-40 cursor-not-allowed': !canRefine }">
            <input type="checkbox" v-model="autoRefine" :disabled="isRunning || !canRefine"
              class="w-3.5 h-3.5 accent-success" />
            <div>
              <span class="text-xs font-medium text-text">Enable cross-correlation refinement</span>
              <p class="text-xs text-muted">
                Multi-signal: GRF (C3D) → marker velocity → projected Vicon.
                Best signal auto-selected. Calibration files optional but improve accuracy.
              </p>
            </div>
          </label>

          <div v-if="autoRefine" class="space-y-2">
            <label>
              <div class="flex justify-between text-xs text-muted mb-1">
                <span>Search window</span>
                <span class="font-mono text-text">±{{ searchRangeS }}s</span>
              </div>
              <input type="range" v-model.number="searchRangeS" min="0.5" max="5" step="0.5"
                :disabled="isRunning" class="w-full" />
            </label>
          </div>
        </div>

        <!-- Parameters -->
        <div class="rounded-xl border border-border p-4 space-y-3">
          <p class="text-xs font-semibold text-muted ">Parameters</p>

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
            <p v-if="taskNameCollision" class="text-xs text-warning mt-1 flex items-center gap-1">
              <AlertCircle :size="10" />
              "{{ taskName }}" already exists — running will overwrite it.
            </p>
          </div>

          <div class="grid grid-cols-2 gap-2">
            <label>
              <span class="text-xs text-muted">Timezone (UTC+N)</span>
              <input type="number" v-model.number="timezoneHours" step="0.5" :disabled="isRunning"
                class="mt-1 w-full bg-surface border border-border rounded-lg px-2 py-1.5 text-xs text-text focus:outline-none focus:border-accent disabled:opacity-50" />
            </label>
            <label>
              <span class="text-xs text-muted">Clock correction (h)</span>
              <input type="number" v-model.number="timeCorrectHours" step="1" :disabled="isRunning"
                class="mt-1 w-full bg-surface border border-border rounded-lg px-2 py-1.5 text-xs text-text focus:outline-none focus:border-accent disabled:opacity-50" />
            </label>
            <label>
              <span class="text-xs text-muted">Vicon FPS</span>
              <input type="number" v-model.number="viconFps" step="1" :disabled="isRunning"
                class="mt-1 w-full bg-surface border border-border rounded-lg px-2 py-1.5 text-xs text-text focus:outline-none focus:border-accent disabled:opacity-50" />
            </label>
            <label>
              <span class="text-xs text-muted">Manual frame offset</span>
              <input type="number" v-model.number="manualFrameOffset" step="1" :disabled="isRunning"
                class="mt-1 w-full bg-surface border border-border rounded-lg px-2 py-1.5 text-xs text-text focus:outline-none focus:border-accent disabled:opacity-50" />
            </label>
          </div>

          <label>
            <span class="text-xs text-muted">Video rotation</span>
            <select v-model="videoRotation" :disabled="isRunning"
              class="mt-1 w-full bg-surface border border-border rounded-lg px-2 py-1.5 text-xs text-text focus:outline-none focus:border-accent disabled:opacity-50">
              <option v-for="r in ROTATIONS" :key="r" :value="r">{{ r }}</option>
            </select>
          </label>

          <label>
            <span class="text-xs text-muted">Manual video start (override ffprobe)</span>
            <input v-model="manualVideoStart" placeholder="YYYY-MM-DD HH:MM:SS.fff  (leave blank for ffprobe)"
              :disabled="isRunning"
              class="mt-1 w-full bg-surface border border-border rounded-lg px-3 py-1.5 text-xs text-text font-mono focus:outline-none focus:border-accent disabled:opacity-50" />
          </label>

          <label>
            <span class="text-xs text-muted">Force sync offset (s) — override if auto fails</span>
            <input v-model="forceSyncOffset" placeholder="e.g. 3.332  (leave blank for auto)"
              :disabled="isRunning"
              class="mt-1 w-full bg-surface border border-border rounded-lg px-3 py-1.5 text-xs text-text font-mono focus:outline-none focus:border-accent disabled:opacity-50" />
            <p class="text-xs text-muted mt-0.5">Use the known-good value from a previous working run. Bypasses XCP/ffprobe entirely.</p>
          </label>
        </div>

        <!-- Run -->
        <Button variant="primary" size="lg" :full-width="true"
          :disabled="!canRun" :loading="isRunning" @click="handleRun">
          <template v-if="!isRunning" #icon><Play :size="15" /></template>
          {{ isRunning ? (autoRefine ? 'Synchronising (cross-corr)…' : 'Synchronising…') : 'Run Synchronisation' }}
        </Button>

        <div v-if="displayError" class="flex items-start gap-2 bg-danger/10 border border-danger/30 rounded-xl p-3">
          <AlertCircle :size="14" class="text-danger flex-shrink-0 mt-0.5" />
          <p class="text-xs text-danger leading-relaxed whitespace-pre-wrap">{{ displayError }}</p>
        </div>
      </div>

      <!-- ── Right panel ─────────────────────────────────────────────────────── -->
      <div class="col-span-3 space-y-4">
        <Transition name="pf" mode="out-in">

          <!-- Progress -->
          <div v-if="showProgress" key="progress"
            class="rounded-xl border border-border p-5 space-y-4">
            <div class="flex items-center justify-between">
              <span class="text-xs font-semibold text-muted ">Synchronising…</span>
              <StatusBadge status="running" />
            </div>
            <ProgressSteps :job-status="jobStatus" />
            <div v-if="autoRefine" class="flex items-start gap-2 bg-surface rounded-lg p-2.5">
              <Info :size="12" class="text-accent mt-0.5 flex-shrink-0" />
              <p class="text-xs text-muted">Cross-correlation is running — this takes 1–3 min for a typical trial video.</p>
            </div>
          </div>

          <!-- Results -->
          <div v-else-if="showResult" key="results" class="space-y-4">

            <!-- XCP auto-matched (no file uploaded, backend found it next to the raw export) -->
            <div v-if="result!.xcp_auto_note"
              class="flex items-start gap-2 bg-accent/10 border border-accent/30 rounded-xl p-3">
              <Info :size="13" class="text-accent flex-shrink-0 mt-0.5" />
              <p class="text-xs text-accent">{{ result!.xcp_auto_note }}</p>
            </div>

            <!-- ⚠ NO OVERLAP — most prominent warning when sync is wrong -->
            <div v-if="!result!.overlap"
              class="bg-danger/10 border-2 border-danger/60 rounded-xl p-4 space-y-3">
              <div class="flex items-center gap-2">
                <AlertCircle :size="16" class="text-danger flex-shrink-0" />
                <p class="text-sm font-bold text-danger">No overlap — sync offset is wrong</p>
              </div>
              <p class="text-xs text-danger/90 leading-relaxed">
                Video frame 0 maps to Vicon time
                <strong>{{ result!.video_at_vicon_start_s.toFixed(1) }}s</strong>,
                but valid Vicon data is only between
                <strong>{{ result!.first_valid_vicon_time_s.toFixed(1) }}s</strong> and
                <strong>{{ (result!.last_valid_raw_vicon_frame / result!.vicon_fps).toFixed(1) }}s</strong>.
                The skeleton cannot be projected and cross-correlation was skipped.
              </p>
              <div class="bg-danger/10 border border-danger/30 rounded-lg p-3 space-y-1.5 text-xs text-danger/80">
                <p class="font-semibold">How to fix:</p>
                <p>1. Check <strong>Timezone (UTC+N)</strong> — must match the Vicon PC's UTC offset (e.g. 5 for UTC+5).</p>
                <p>2. Check <strong>Clock correction (h)</strong> — only needed if the iPhone clock was wrong at recording time.</p>
                <p>   · The value is 0 when the phone clock was correct; otherwise it is the whole-hour error.</p>
                <p>3. Or paste the known-good offset into <strong>Force sync offset</strong> and re-run.</p>
              </div>
              <div class="flex items-center justify-between gap-2 bg-accent/10 border border-accent/30 rounded-lg p-2.5">
                <p class="text-xs text-text">
                  Solved directly from this file's timestamps — Clock correction
                  <strong class="font-mono">{{ result!.suggested_clock_correction_hours >= 0 ? '+' : '' }}{{ result!.suggested_clock_correction_hours.toFixed(4) }}</strong>
                  should land frame 0 at the start of valid Vicon data.
                </p>
                <Button size="sm" variant="secondary"
                  @click="timeCorrectHours = result!.suggested_clock_correction_hours">
                  Use this value
                </Button>
              </div>
              <div class="text-xs font-mono text-danger/70 space-y-0.5">
                <p>Vicon start (local):  {{ result!.vicon_capture_start_local_iso.replace('T', ' ').slice(0, 23) }}</p>
                <p>Video start  (UTC):   {{ result!.video_creation_time_utc_iso.replace('T', ' ').slice(0, 23) }}</p>
                <p>Computed offset:      {{ result!.sync_offset_s.toFixed(2) }}s  ({{ (result!.sync_offset_s / 3600).toFixed(2) }}h)</p>
                <p>Source: {{ result!.video_creation_time_source }}</p>
              </div>
            </div>

            <!-- ✓ GOOD SYNC — method + offset banner -->
            <div v-if="result!.overlap" class="rounded-xl border p-4 flex items-center justify-between"
              :style="{ background: `color-mix(in srgb, ${syncMethodColor(result!.sync_method)} 10%, transparent)`, borderColor: syncMethodColor(result!.sync_method) }">
              <div class="flex-1 min-w-0">
                <div class="flex items-center gap-2 mb-1">
                  <component :is="result!.sync_method === 'cross_correlation_refined' ? GitMerge : Clock"
                    :size="12" :style="{ color: syncMethodColor(result!.sync_method) }" />
                  <span class="text-xs text-muted">{{ syncMethodLabel(result!.sync_method) }}</span>
                </div>
                <p class="text-xs text-muted leading-relaxed">{{ result!.accuracy_note }}</p>
              </div>
              <div class="text-right flex-shrink-0 ml-4">
                <p class="text-3xl font-bold font-mono"
                  :style="{ color: syncMethodColor(result!.sync_method) }">
                  {{ result!.sync_offset_s > 0 ? '+' : '' }}{{ result!.sync_offset_s.toFixed(4) }}s
                </p>
                <p class="text-xs text-muted">sync offset</p>
                <p class="text-xs mt-0.5 text-success">✓ overlap detected</p>
              </div>
            </div>

            <!-- Cross-correlation detail card (shown when refine was run) -->
            <div v-if="result!.refine_signal_used"
              class="rounded-xl border p-3 space-y-2"
              :class="result!.refine_conf_label === 'HIGH'   ? 'bg-success/5 border-success/20' :
                      result!.refine_conf_label === 'MEDIUM' ? 'bg-warning/5 border-warning/20' :
                                                               'bg-destructive/5 border-destructive/20'">
              <div class="flex items-center justify-between">
                <span class="text-xs font-semibold"
                  :class="result!.refine_conf_label === 'HIGH'   ? 'text-success' :
                          result!.refine_conf_label === 'MEDIUM' ? 'text-warning' :
                                                                   'text-destructive'">
                  Cross-correlation — {{ result!.refine_conf_label }} confidence
                </span>
                <span class="text-xs font-mono text-muted">
                  {{ result!.refine_correction_applied ? ((result!.refine_correction_ms ?? 0) > 0 ? '+' : '') + result!.refine_correction_ms + ' ms applied' : 'no correction (too low)' }}
                </span>
              </div>
              <div class="grid grid-cols-3 gap-2 text-xs font-mono">
                <div v-for="(sig, name) in (result!.refine_all_signals ?? {})" :key="name"
                  class="rounded p-1.5"
                  :class="name === result!.refine_signal_used
                    ? 'bg-accent/15 border border-accent/30'
                    : 'bg-surface border border-border'">
                  <p class="font-semibold truncate"
                    :class="name === result!.refine_signal_used ? 'text-accent' : 'text-muted'">
                    {{ name }}{{ name === result!.refine_signal_used ? ' ★' : '' }}
                  </p>
                  <p class="text-text">conf {{ (sig as any).confidence?.toFixed(2) }}</p>
                  <p class="text-muted">{{ (sig as any).lag_ms > 0 ? '+' : '' }}{{ (sig as any).lag_ms }} ms</p>
                </div>
              </div>
              <p v-if="result!.refine_conf_label === 'LOW'" class="text-xs text-destructive">
                All signals low confidence — wall-clock offset kept unchanged.
                Increase search window, check calibration files, or use Fine-Tune Manual Offset.
              </p>
            </div>

            <!-- Timestamp details (always shown) -->
            <div class="rounded-xl border border-border p-4 space-y-2">
              <p class="text-xs font-semibold text-muted  mb-2">Clock Timestamps</p>
              <div class="grid grid-cols-2 gap-3 text-xs">
                <div>
                  <p class="text-muted mb-0.5">Vicon start (local UTC+{{ result!.timezone_offset_hours }})</p>
                  <p class="font-mono text-text text-xs">{{ result!.vicon_capture_start_local_iso.replace('T', ' ').slice(0, 23) }}</p>
                </div>
                <div>
                  <p class="text-muted mb-0.5">Video start (UTC, after correction)</p>
                  <p class="font-mono text-text text-xs">{{ result!.video_creation_time_utc_iso.replace('T', ' ').slice(0, 23) }}</p>
                </div>
              </div>
              <div class="grid grid-cols-2 gap-3 text-xs text-muted pt-1">
                <div>Video frame 0 → Vicon: <span class="font-mono text-text">{{ result!.video_at_vicon_start_s.toFixed(3) }}s (fr {{ (result!.video_at_vicon_start_s * result!.vicon_fps).toFixed(0) }})</span></div>
                <div>Valid Vicon range: <span class="font-mono text-text">fr {{ result!.first_valid_raw_vicon_frame }}–{{ result!.last_valid_raw_vicon_frame }}</span></div>
                <div>Source: <span class="text-text">{{ result!.video_creation_time_source }}</span></div>
                <div>Rotation: <span class="font-mono text-text">{{ result!.video_rotation }}</span></div>
              </div>
            </div>

            <!-- Metrics (only when overlap is good) -->
            <MetricsGrid v-if="result!.overlap" :cols="3">
              <MetricsCard label="Video FPS"   :value="result!.video_fps.toFixed(3)" unit="Hz" :delay="0" />
              <MetricsCard label="Vicon FPS"   :value="result!.vicon_fps" unit="Hz" :delay="0.05" />
              <MetricsCard label="Processing"  :value="result!.processing_time_s.toFixed(1)" unit="s" :delay="0.1" />
            </MetricsGrid>

            <!-- Task info -->
            <div class="bg-surface rounded-xl border border-border p-3 flex items-center gap-3">
              <component :is="result!.overlap ? CheckCircle : AlertCircle" :size="14"
                :class="result!.overlap ? 'text-success' : 'text-warning'" class="flex-shrink-0" />
              <div class="flex-1 min-w-0">
                <p class="text-xs font-medium text-text truncate">{{ result!.task_name }}</p>
                <p class="text-xs text-muted truncate">{{ result!.output_dir }}</p>
              </div>
            </div>

            <!-- Manual offset fine-tune -->
            <div class="rounded-xl border border-border p-4 space-y-3">
              <div class="flex items-center justify-between">
                <div class="flex items-center gap-2">
                  <SlidersHorizontal :size="13" class="text-accent" />
                  <span class="text-xs font-semibold text-muted ">Fine-tune Manual Offset</span>
                </div>
                <button class="text-xs text-accent hover:text-accent/80 transition-colors"
                  @click="showOffset = !showOffset">
                  {{ showOffset ? 'Collapse' : 'Expand' }}
                </button>
              </div>

              <template v-if="showOffset">
                <div class="flex items-center gap-3">
                  <label class="flex-1">
                    <div class="flex justify-between text-xs text-muted mb-1">
                      <span>Manual frame offset</span>
                      <span class="font-mono text-text">{{ manualFrameOffset }} fr = {{ (manualFrameOffset / result!.video_fps).toFixed(4) }}s</span>
                    </div>
                    <input type="range" v-model.number="manualFrameOffset"
                      :min="-Math.floor(result!.video_fps * 10)"
                      :max=" Math.floor(result!.video_fps * 10)"
                      class="w-full" />
                  </label>
                  <Button size="sm" :loading="refining" @click="handleReapply">
                    <template #icon><CheckCircle :size="12" /></template>
                    Apply
                  </Button>
                </div>
                <p class="text-xs text-muted">Updates the sync JSON without re-running the full pipeline. The overlay will use this offset.</p>
              </template>
            </div>

            <!-- Next step -->
            <div class="bg-accent/5 border border-accent/20 rounded-xl p-3 space-y-2">
              <p class="text-xs font-semibold text-accent mb-2">Next step</p>
              <button
                class="w-full flex items-center justify-center gap-2 bg-accent/10 hover:bg-accent/20 border border-accent/30 text-accent rounded-lg px-3 py-2.5 text-xs font-semibold transition-colors"
                @click="goToOverlay"
              >
                <Film :size="13" />
                Generate Skeleton Overlay
                <ArrowRight :size="12" />
              </button>
              <p class="text-xs text-muted mt-1.5">The overlay page will auto-detect the task '{{ result!.task_name }}'.</p>
              <button
                class="w-full flex items-center justify-center gap-2 bg-surface hover:bg-accent/10 border border-border hover:border-accent/30 text-text hover:text-accent rounded-lg px-3 py-2.5 text-xs font-semibold transition-colors"
                @click="prepareNextTrial"
              >
                <SkipForward :size="12" />
                Next Trial ({{ subjectId }} / {{ task }} / {{ incrementTrial(trial) }})
              </button>
              <p class="text-xs text-muted mt-1.5">
                Keeps Subject + Task + calibration files, advances Trial, clears video/Vicon/XCP for the next upload.
              </p>
            </div>
          </div>

          <!-- Empty -->
          <div v-else-if="showEmpty" key="empty"
            class="flex flex-col items-center justify-center h-64 text-center text-muted space-y-3">
            <Zap :size="40" class="opacity-20" />
            <div>
              <div class="text-sm font-medium">Ready to synchronise</div>
              <div class="text-xs mt-1">Upload video, Vicon CSV, and XCP file, then click Run</div>
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
