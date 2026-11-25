<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import {
  Film, Play, RotateCcw, ExternalLink, AlertCircle, Info,
  CheckCircle, XCircle, FolderOpen, SlidersHorizontal, Eye,
} from '@lucide/vue'
import Button        from '@renderer/sync/components/ui/Button.vue'
import PageHeader    from '@renderer/sync/components/ui/PageHeader.vue'
import DropZone      from '@renderer/sync/components/ui/DropZone.vue'
import ProgressSteps from '@renderer/sync/components/ui/ProgressSteps.vue'
import StatusBadge   from '@renderer/sync/components/ui/StatusBadge.vue'
import MetricsCard   from '@renderer/sync/components/ui/MetricsCard.vue'
import MetricsGrid   from '@renderer/sync/components/ui/MetricsGrid.vue'
import SpatialFineTune from '@renderer/sync/components/SpatialFineTune.vue'
import { useSyncJob } from '@renderer/sync/composables/useSyncJob'
import { useCalibHandoff } from '@renderer/sync/composables/useCalibHandoff'
import CalibrationPicker from '@renderer/sync/components/CalibrationPicker.vue'
import {
  listProjects, startOverlay, getOverlayResult,
  openProjectFolder, outputFileUrl,
  getTaskInfo, previewOverlayFrame, reapplyOffset,
  runJobToCompletion,
} from '@renderer/sync/lib/api'
import { ZERO_EXTRINSIC_DELTA, type ExtrinsicDelta } from '@renderer/sync/lib/api_extrinsic_correction'
import type { ProjectEntry, OverlayResult, TaskInfo } from '@renderer/sync/lib/types'

const ROTATIONS = [
  { value: '',      label: 'From sync JSON (default)' },
  { value: 'ccw90', label: 'Counter-clockwise 90°' },
  { value: 'cw90',  label: 'Clockwise 90°' },
  { value: '180',   label: '180°' },
  { value: 'none',  label: 'No rotation' },
]

const { jobId, jobStatus, isRunning, error, startJob, reset: resetJob } = useSyncJob()

// ── Task selection ─────────────────────────────────────────────────────────────
const tasks          = ref<ProjectEntry[]>([])
const selectedTask   = ref('')
const loadingTasks   = ref(false)

// ── Optional calibration upload (if not already in task folder) ───────────────
const intrinsicsFile = ref<File | null>(null)
const extrinsicsFile = ref<File | null>(null)
function onCalibPick(f: { intrinsics: File | null; extrinsics: File | null }) {
  if (f.intrinsics) intrinsicsFile.value = f.intrinsics
  if (f.extrinsics) extrinsicsFile.value = f.extrinsics
}
const rotateOverride = ref('')
const exportPlain  = ref(true)    // also export clean RGB video for training
const blurFacesRgb = ref(false)   // blur every face in _rgb.mp4 (privacy)
const export2dCsv  = ref(true)    // also write synced_vicon2d_60hz.csv (Vicon-PROJECTED 2D, NOT RTMPose)

// ── Results ────────────────────────────────────────────────────────────────────
const result     = ref<OverlayResult | null>(null)
const fetchError = ref<string | null>(null)

// ── Batch queue — separate from selectedTask (which drives preview/fine-tune) ────
const batchSelected = ref<Set<string>>(new Set())
const batchRunning  = ref(false)
const batchResults  = ref<{ task_name: string; status: 'ok' | 'error'; error?: string }[]>([])
const batchProgress = ref({ current: 0, total: 0 })

// Most trials run ~60fps; SUB02/03 are genuinely ~30fps (real capture, not a
// bug) — flag anything more than 5fps off the common rate so it reads as
// "different from the rest," not just informational text easy to skim past.
const COMMON_FPS = 60
function isOffRate(fps: number): boolean {
  return Math.abs(fps - COMMON_FPS) > 5
}

function toggleBatch(name: string) {
  const s = new Set(batchSelected.value)
  s.has(name) ? s.delete(name) : s.add(name)
  batchSelected.value = s
}
function batchSelectAll() { batchSelected.value = new Set(tasks.value.map(t => t.id)) }
function batchSelectNone() { batchSelected.value = new Set() }

// Runs each queued trial's own already-saved sync offset — no live preview step,
// so this is for trials you've already spot-checked, not first-time alignment.
async function runBatch() {
  const names = [...batchSelected.value]
  if (names.length === 0) return
  result.value       = null
  fetchError.value   = null
  batchResults.value = []
  batchRunning.value = true
  batchProgress.value = { current: 0, total: names.length }
  for (const name of names) {
    batchProgress.value = { current: batchProgress.value.current + 1, total: names.length }
    try {
      // Never forward a calibration-override upload to a batched trial — it
      // belongs to whichever single task is currently selected, not to every
      // queued trial, and would clobber each trial's own saved extrinsics.
      const { job_id } = await startOverlay(
        name, rotateOverride.value || null, exportPlain.value,
        null, null, blurFacesRgb.value, export2dCsv.value,
      )
      await runJobToCompletion('overlay', job_id)
      batchResults.value = [...batchResults.value, { task_name: name, status: 'ok' }]
    } catch (err: unknown) {
      batchResults.value = [...batchResults.value, {
        task_name: name, status: 'error', error: err instanceof Error ? err.message : String(err),
      }]
    }
  }
  batchRunning.value = false
  loadTasks()
}

// ── Preview / offset fine-tune ─────────────────────────────────────────────────
const taskInfo       = ref<TaskInfo | null>(null)
const previewFrame   = ref(0)
const previewOffset  = ref(0)     // pending offset — user is adjusting
const previewImg     = ref<string | null>(null)
const previewLoading = ref(false)
const previewInRange = ref(true)
const previewViconT  = ref(0)
const savingOffset   = ref(false)
const offsetSaved    = ref(false)
const extrinsicDelta = ref<ExtrinsicDelta>({ ...ZERO_EXTRINSIC_DELTA })

let _fetchTimer: ReturnType<typeof setTimeout> | null = null

async function fetchPreview(immediate = false) {
  if (!selectedTask.value || !taskInfo.value) return
  if (_fetchTimer) { clearTimeout(_fetchTimer); _fetchTimer = null }
  const doFetch = async () => {
    previewLoading.value = true
    try {
      const r = await previewOverlayFrame(
        selectedTask.value,
        previewFrame.value,
        previewOffset.value,
        undefined,
        extrinsicDelta.value,
      )
      previewImg.value     = `data:image/jpeg;base64,${r.image_b64}`
      previewInRange.value = r.in_range
      previewViconT.value  = r.vicon_time_s
    } catch { /* silently ignore — backend errors shown in overlay run */ }
    finally { previewLoading.value = false }
  }
  if (immediate) await doFetch()
  else           _fetchTimer = setTimeout(doFetch, 380)
}

async function loadTaskInfo(name: string) {
  taskInfo.value      = null
  previewImg.value    = null
  extrinsicDelta.value = { ...ZERO_EXTRINSIC_DELTA }
  if (!name) return
  try {
    const info = await getTaskInfo(name)
    taskInfo.value       = info
    previewOffset.value  = info.manual_frame_offset
    previewFrame.value   = Math.floor(info.total_frames / 2)
    fetchPreview(true)
  } catch { /* task may not have sync yet */ }
}

function adjustFrame(delta: number) {
  if (!taskInfo.value) return
  previewFrame.value = Math.max(0, Math.min(previewFrame.value + delta, taskInfo.value.total_frames - 1))
  fetchPreview()
}

function adjustOffset(delta: number) {
  previewOffset.value += delta
  offsetSaved.value = false
  fetchPreview()
}

async function saveOffset() {
  if (!selectedTask.value) return
  savingOffset.value = true
  try {
    await reapplyOffset(selectedTask.value, previewOffset.value)
    if (taskInfo.value) taskInfo.value = { ...taskInfo.value, manual_frame_offset: previewOffset.value }
    offsetSaved.value = true
    setTimeout(() => { offsetSaved.value = false }, 2500)
  } catch { /* ignore */ }
  finally { savingOffset.value = false }
}

// The Sync→Overlay handoff pre-fills extrinsicsFile as a fallback for a
// cleaned temp folder. Once a spatial correction is saved, that stale upload
// must not be resent on the next Generate — it would silently overwrite the
// just-saved correction in extrinsics.json with the original, uncorrected copy.
function onSpatialCorrectionSaved() {
  extrinsicsFile.value = null
  fetchPreview(true)
}

watch(extrinsicDelta, () => fetchPreview(), { deep: true })

watch(selectedTask, (name) => {
  result.value     = null
  fetchError.value = null
  resetJob()
  loadTaskInfo(name)
})

async function loadTasks() {
  loadingTasks.value = true
  try {
    tasks.value = await listProjects()
    // Auto-select the most recent synced task if none selected
    if (!selectedTask.value && tasks.value.length > 0) {
      selectedTask.value = tasks.value[0].id
    }
  } catch { /* offline */ }
  finally { loadingTasks.value = false }
}

const { consume: consumeCalib } = useCalibHandoff()

onMounted(() => {
  loadTasks()
  const calib = consumeCalib()
  if (calib?.intrinsics) intrinsicsFile.value = calib.intrinsics
  if (calib?.extrinsics) extrinsicsFile.value = calib.extrinsics
})

const selectedTaskInfo = computed(() =>
  tasks.value.find(t => t.id === selectedTask.value) ?? null
)

const hasCalibInFolder = computed(() => {
  // We can't directly check what's in the task folder from the frontend,
  // but if sync was run with calibration files, they're stored there.
  // We just let the backend decide and show the error if missing.
  return true
})

watch(() => jobStatus.value?.status, (status) => {
  if (status === 'completed' && jobId.value) {
    fetchError.value = null
    getOverlayResult(jobId.value)
      .then(r => { result.value = r })
      .catch(e => { fetchError.value = `Failed to load result: ${e?.message}` })
  }
})

const canRun       = computed(() => !!selectedTask.value && !isRunning.value && !batchRunning.value)
const canRunBatch  = computed(() => batchSelected.value.size > 0 && !isRunning.value && !batchRunning.value)
const displayError = computed(() => error.value || fetchError.value)
const showProgress    = computed(() => isRunning.value && !!jobStatus.value)
const showBatchResult = computed(() => !batchRunning.value && batchResults.value.length > 0)
const showResult   = computed(() => !isRunning.value && !batchRunning.value && !showBatchResult.value && !!result.value)
const showPreview  = computed(() => !isRunning.value && !batchRunning.value && !showBatchResult.value && !result.value && !!selectedTask.value && !!taskInfo.value)
const showEmpty    = computed(() => !isRunning.value && !batchRunning.value && !showBatchResult.value && !result.value && !showPreview.value)

async function handleRun() {
  if (!selectedTask.value) return
  result.value = null
  fetchError.value = null
  try {
    const { job_id } = await startOverlay(
      selectedTask.value,
      rotateOverride.value || null,
      exportPlain.value,
      intrinsicsFile.value,
      extrinsicsFile.value,
      blurFacesRgb.value,
      export2dCsv.value,
    )
    startJob(job_id, 'overlay')
  } catch (err: unknown) {
    fetchError.value = `Failed to start: ${err instanceof Error ? err.message : String(err)}`
  }
}

function handleReset() {
  resetJob()
  result.value = fetchError.value = null
  intrinsicsFile.value = extrinsicsFile.value = null
  batchResults.value = []
}

function openFolder(taskName: string) {
  openProjectFolder(taskName)
}

function openVideo(r: OverlayResult) {
  // Open the task output folder in Explorer
  openProjectFolder(r.task_name)
}
</script>

<template>
  <div class="p-6 animate-fade-in flex flex-col h-screen gap-5">

    <!-- Header -->
    <PageHeader title="Skeleton Overlay" :icon="Film" color="#06B6D4" :step="3"
      subtitle="Draw the calibrated Vicon skeleton on the synced video to verify the alignment, then export rgb.mp4 + synced.csv.">
      <template #actions>
        <Button v-if="result || isRunning || batchRunning || batchResults.length" variant="ghost" size="sm" @click="handleReset">
          <template #icon><RotateCcw :size="13" /></template>
          Reset
        </Button>
      </template>
    </PageHeader>

    <div class="grid grid-cols-5 gap-5 flex-1 min-h-0">

      <!-- ── Left panel ──────────────────────────────────────────────────────── -->
      <div class="col-span-2 space-y-4 overflow-y-auto">

        <!-- Task selection -->
        <div class="rounded-xl border border-border p-4 space-y-3">
          <div class="flex items-center justify-between">
            <p class="text-xs font-semibold text-muted ">Select Task</p>
            <button class="text-xs text-accent hover:text-accent/80 transition-colors"
              :disabled="loadingTasks"
              @click="loadTasks">
              {{ loadingTasks ? 'Loading…' : 'Refresh' }}
            </button>
          </div>

          <div v-if="tasks.length === 0 && !loadingTasks"
            class="flex items-start gap-2 bg-surface rounded-lg p-2.5">
            <Info :size="12" class="text-muted mt-0.5 flex-shrink-0" />
            <p class="text-xs text-muted">No synced tasks found. Run Sync first.</p>
          </div>

          <template v-else>
            <p class="text-xs text-muted leading-relaxed">
              Click a row to preview/fine-tune it below. Check the box to queue it for
              <strong class="text-text">Run Batch</strong> (uses each trial's already-saved offset).
            </p>
            <div class="flex gap-2 text-xs">
              <button class="text-accent hover:text-accent/80" @click="batchSelectAll">Select all</button>
              <span class="text-muted">·</span>
              <button class="text-accent hover:text-accent/80" @click="batchSelectNone">Clear</button>
              <span class="text-muted ml-auto">{{ batchSelected.size }} / {{ tasks.length }} queued</span>
            </div>
            <div class="space-y-1.5">
              <div
                v-for="task in tasks"
                :key="task.id"
                class="flex items-center gap-2.5 p-2.5 rounded-lg border cursor-pointer transition-colors"
                :class="selectedTask === task.id
                  ? 'border-accent bg-accent/10'
                  : 'border-border hover:border-accent/40 bg-surface'"
                @click="selectedTask = task.id"
              >
                <input type="checkbox" class="w-3.5 h-3.5 accent-accent flex-shrink-0"
                  :checked="batchSelected.has(task.id)"
                  @click.stop @change="toggleBatch(task.id)" />
                <div class="w-2 h-2 rounded-full flex-shrink-0"
                  :class="task.has_overlay ? 'bg-success' : 'bg-accent/60'" />
                <div class="flex-1 min-w-0">
                  <p class="text-xs font-medium text-text truncate">{{ task.name }}</p>
                  <p class="text-xs text-muted">
                    {{ (task.sync_offset_s ?? 0).toFixed(3) }}s offset
                    {{ task.has_overlay ? '· overlay ✓' : '' }}
                    <span v-if="task.video_fps" :class="isOffRate(task.video_fps) ? 'text-warning font-semibold' : ''">
                      · {{ task.video_fps.toFixed(1) }} fps
                    </span>
                  </p>
                </div>
              </div>
            </div>
          </template>

          <div v-if="selectedTaskInfo" class="flex items-start gap-2 bg-surface rounded-lg p-2">
            <Info :size="11" class="text-accent mt-0.5 flex-shrink-0" />
            <p class="text-xs text-muted leading-relaxed">
              Video and Vicon CSV must still be in the temp folder from when sync was run.
              Re-run Sync if the pipeline reports missing files.
            </p>
          </div>
        </div>

        <!-- Optional calibration upload -->
        <div class="rounded-xl border border-border p-4 space-y-3">
          <div class="flex items-center justify-between">
            <p class="text-xs font-semibold text-muted ">Calibration Override</p>
            <span class="text-xs text-muted">optional</span>
          </div>
          <p class="text-xs text-muted">
            Upload only if not already saved during Sync or to use different calibration files.
          </p>
          <CalibrationPicker :disabled="isRunning" @pick="onCalibPick" />
          <DropZone label="Intrinsics JSON" description="intrinsic.json"
            accept=".json"
            :file="intrinsicsFile" :disabled="isRunning" @update:file="intrinsicsFile = $event" />
          <DropZone label="Extrinsics JSON" description="extrinsics.json"
            accept=".json"
            :file="extrinsicsFile" :disabled="isRunning" @update:file="extrinsicsFile = $event" />
        </div>

        <!-- Rotation override -->
        <div class="rounded-xl border border-border p-4 space-y-2">
          <p class="text-xs font-semibold text-muted ">Video Rotation</p>
          <select v-model="rotateOverride" :disabled="isRunning"
            class="w-full bg-surface border border-border rounded-lg px-2 py-1.5 text-xs text-text focus:outline-none focus:border-accent disabled:opacity-50">
            <option v-for="r in ROTATIONS" :key="r.value" :value="r.value">{{ r.label }}</option>
          </select>
        </div>

        <!-- Offset fine-tune -->
        <div v-if="taskInfo" class="rounded-xl border border-border p-4 space-y-3">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-1.5">
              <SlidersHorizontal :size="11" class="text-accent" />
              <p class="text-xs font-semibold text-muted ">Offset Fine-tune</p>
            </div>
            <span class="text-xs text-muted font-mono">saved: {{ taskInfo.manual_frame_offset >= 0 ? '+' : '' }}{{ taskInfo.manual_frame_offset }}</span>
          </div>

          <!-- Adjustment buttons -->
          <div class="flex items-center gap-1">
            <button
              v-for="d in [-5, -1]" :key="d"
              class="flex-1 px-1.5 py-1 rounded-lg bg-surface border border-border text-xs font-mono text-text hover:border-accent/60 hover:bg-accent/5 active:scale-95 transition-all"
              @click="adjustOffset(d)">{{ d }}</button>
            <div class="flex-[2] text-center py-1">
              <p class="text-base font-bold font-mono" :class="previewOffset !== (taskInfo.manual_frame_offset) ? 'text-warning' : 'text-accent'">
                {{ previewOffset >= 0 ? '+' : '' }}{{ previewOffset }}
              </p>
              <p class="text-xs text-muted -mt-0.5">frames</p>
            </div>
            <button
              v-for="d in [1, 5]" :key="d"
              class="flex-1 px-1.5 py-1 rounded-lg bg-surface border border-border text-xs font-mono text-text hover:border-accent/60 hover:bg-accent/5 active:scale-95 transition-all"
              @click="adjustOffset(d)">+{{ d }}</button>
          </div>

          <!-- Save -->
          <Button
            variant="secondary" size="sm" :full-width="true"
            :loading="savingOffset"
            :disabled="savingOffset"
            @click="saveOffset">
            {{ offsetSaved ? '✓ Offset saved' : 'Save offset to sync JSON' }}
          </Button>
        </div>

        <!-- Spatial fine-tune (whole-skeleton position/tilt, not timing) -->
        <SpatialFineTune
          v-if="taskInfo"
          v-model="extrinsicDelta"
          :task-name="selectedTask"
          @saved="onSpatialCorrectionSaved"
        />

        <!-- Export options -->
        <div class="rounded-xl border border-border p-4 space-y-2">
          <p class="text-xs font-semibold text-muted ">Export Options</p>
          <label class="flex items-start gap-2.5 cursor-pointer group">
            <input
              type="checkbox"
              v-model="exportPlain"
              :disabled="isRunning"
              class="mt-0.5 w-3.5 h-3.5 rounded border-border accent-accent cursor-pointer flex-shrink-0"
            />
            <div>
              <p class="text-xs font-medium text-text group-hover:text-accent transition-colors">
                Export plain RGB video for training
              </p>
              <p class="text-xs text-muted mt-0.5">
                Trimmed + undistorted, no skeleton — saved as <span class="font-mono">_rgb.mp4</span>
              </p>
            </div>
          </label>
          <label class="flex items-start gap-2.5 cursor-pointer group">
            <input
              type="checkbox"
              v-model="blurFacesRgb"
              :disabled="isRunning || !exportPlain"
              class="mt-0.5 w-3.5 h-3.5 rounded border-border accent-accent cursor-pointer flex-shrink-0"
            />
            <div>
              <p class="text-xs font-medium text-text group-hover:text-accent transition-colors">
                Blur faces in RGB video
              </p>
              <p class="text-xs text-muted mt-0.5">
                Jitter-stable blur (face-tracked across the whole video) on every
                detected face (subject + bystanders) in <span class="font-mono">_rgb.mp4</span>
              </p>
            </div>
          </label>
          <label class="flex items-start gap-2.5 cursor-pointer group">
            <input
              type="checkbox"
              v-model="export2dCsv"
              :disabled="isRunning"
              class="mt-0.5 w-3.5 h-3.5 rounded border-border accent-accent cursor-pointer flex-shrink-0"
            />
            <div>
              <p class="text-xs font-medium text-text group-hover:text-accent transition-colors">
                Export 2D projected CSV
              </p>
              <p class="text-xs text-muted mt-0.5">
                Vicon skeleton camera-projected to pixels (NOT RTMPose), same frames as <span class="font-mono">synced_60hz.csv</span>
                — saved as <span class="font-mono">synced_vicon2d_60hz.csv</span>
              </p>
            </div>
          </label>
        </div>

        <!-- Run — single (preview/fine-tune) trial -->
        <Button variant="primary" size="lg" :full-width="true"
          :disabled="!canRun" :loading="isRunning" @click="handleRun">
          <template v-if="!isRunning" #icon><Play :size="15" /></template>
          {{ isRunning ? 'Rendering overlay…' : 'Generate Overlay Video' }}
        </Button>

        <!-- Run — batch queue (uses each trial's own saved offset, no preview step) -->
        <Button variant="secondary" size="lg" :full-width="true"
          :disabled="!canRunBatch" :loading="batchRunning" @click="runBatch">
          <template v-if="!batchRunning" #icon><Play :size="15" /></template>
          {{ batchRunning ? `Rendering ${batchProgress.current}/${batchProgress.total}…` : `Run Batch (${batchSelected.size})` }}
        </Button>

        <div v-if="displayError" class="flex items-start gap-2 bg-danger/10 border border-danger/30 rounded-xl p-3">
          <AlertCircle :size="14" class="text-danger flex-shrink-0 mt-0.5" />
          <p class="text-xs text-danger leading-relaxed whitespace-pre-wrap">{{ displayError }}</p>
        </div>
      </div>

      <!-- ── Right panel ─────────────────────────────────────────────────────── -->
      <div class="col-span-3 h-full">
        <Transition name="pf" mode="out-in" class="h-full block">

          <!-- Progress — single (preview/fine-tune) trial -->
          <div v-if="showProgress" key="prog" class="rounded-xl border border-border p-5 space-y-4 h-full">
            <div class="flex items-center justify-between">
              <span class="text-xs font-semibold text-muted ">Rendering Overlay</span>
              <StatusBadge status="running" />
            </div>
            <ProgressSteps :job-status="jobStatus" />
            <div class="flex items-start gap-2 bg-surface rounded-lg p-2.5">
              <Info :size="12" class="text-accent mt-0.5 flex-shrink-0" />
              <p class="text-xs text-muted">Rendering every frame — this takes 2–10 min depending on video length and resolution.</p>
            </div>
          </div>

          <!-- Progress — batch (N trials, one at a time) -->
          <div v-else-if="batchRunning" key="batch-prog" class="rounded-xl border border-border p-5 space-y-4">
            <div class="flex items-center justify-between">
              <span class="text-xs font-semibold text-muted ">
                Rendering batch — trial {{ batchProgress.current }}/{{ batchProgress.total }}
              </span>
              <StatusBadge status="running" />
            </div>
            <div class="w-full h-1.5 bg-surface rounded-full overflow-hidden">
              <div class="h-full bg-accent transition-all"
                :style="{ width: `${(batchProgress.current / batchProgress.total) * 100}%` }" />
            </div>
            <div class="space-y-1">
              <div v-for="r in batchResults" :key="r.task_name"
                class="flex items-center gap-2 p-2 rounded-lg bg-surface text-xs">
                <CheckCircle v-if="r.status === 'ok'" :size="13" class="text-success flex-shrink-0" />
                <XCircle v-else :size="13" class="text-danger flex-shrink-0" />
                <span class="text-text flex-1 truncate">{{ r.task_name }}</span>
              </div>
            </div>
          </div>

          <!-- Batch summary (after all trials finish) -->
          <div v-else-if="showBatchResult" key="batch-res" class="space-y-4">
            <div class="bg-success/10 border border-success/30 rounded-xl p-4">
              <div class="flex items-center gap-2 mb-0.5">
                <CheckCircle :size="13" class="text-success" />
                <p class="text-sm font-semibold text-success">
                  {{ batchResults.filter(r => r.status === 'ok').length }}/{{ batchResults.length }} trials succeeded
                </p>
              </div>
            </div>
            <div class="rounded-xl border border-border p-4 space-y-2">
              <p class="text-xs font-semibold text-muted ">Per-trial results</p>
              <div v-for="r in batchResults" :key="r.task_name"
                class="flex items-center gap-2 p-2 rounded-lg bg-surface text-xs">
                <CheckCircle v-if="r.status === 'ok'" :size="13" class="text-success flex-shrink-0" />
                <XCircle v-else :size="13" class="text-danger flex-shrink-0" />
                <span class="text-text flex-1 truncate">{{ r.task_name }}</span>
                <span v-if="r.error" class="text-xs text-muted truncate max-w-[50%]">{{ r.error }}</span>
              </div>
            </div>
            <div class="bg-surface rounded-xl border border-border p-3 flex items-start gap-2">
              <Info :size="12" class="text-accent mt-0.5 flex-shrink-0" />
              <p class="text-xs text-muted">Click a trial row (left panel) to preview it individually.</p>
            </div>
          </div>

          <!-- Results -->
          <div v-else-if="showResult" key="res" class="space-y-4">

            <!-- Success banner -->
            <div class="bg-success/10 border border-success/30 rounded-xl p-4 flex items-center justify-between">
              <div>
                <div class="flex items-center gap-2 mb-1">
                  <CheckCircle :size="13" class="text-success" />
                  <p class="text-sm font-semibold text-success">Overlay generated</p>
                </div>
                <p class="text-xs text-muted">{{ result!.task_name }}</p>
              </div>
              <Button variant="ghost" size="sm" @click="openVideo(result!)">
                <template #icon><FolderOpen :size="12" /></template>
                Open folder
              </Button>
            </div>

            <!-- Metrics -->
            <MetricsGrid :cols="3">
              <MetricsCard label="Total frames"    :value="result!.total_video_frames"         :delay="0" />
              <MetricsCard label="Trimmed frames"  :value="result!.trimmed_frames"             :delay="0.05" />
              <MetricsCard label="Duration"        :value="result!.trimmed_duration_s.toFixed(2)" unit="s" :delay="0.1" />
            </MetricsGrid>
            <MetricsGrid :cols="2">
              <MetricsCard label="Full skeleton"   :value="result!.full_skeleton_pct.toFixed(1)" unit="%" :delay="0.15" />
              <MetricsCard label="Processing time" :value="result!.processing_time_s.toFixed(1)" unit="s" :delay="0.2" />
            </MetricsGrid>

            <!-- Output files -->
            <div class="rounded-xl border border-border p-4 space-y-2">
              <p class="text-xs font-semibold text-muted  mb-2">Output Files</p>
              <!-- Overlay video -->
              <div class="flex items-center gap-2 py-1">
                <div class="w-1.5 h-1.5 rounded-full bg-success flex-shrink-0" />
                <div class="flex-1 min-w-0">
                  <p class="text-xs font-medium text-text">Overlay video</p>
                  <p class="text-xs text-muted truncate font-mono">{{ result!.overlay_video.split(/[\\/]/).pop() }}</p>
                </div>
              </div>
              <!-- Plain RGB video -->
              <div class="flex items-center gap-2 py-1">
                <div class="w-1.5 h-1.5 rounded-full flex-shrink-0"
                  :class="result!.plain_video ? 'bg-success' : 'bg-border'" />
                <div class="flex-1 min-w-0">
                  <div class="flex items-center gap-1.5">
                    <p class="text-xs font-medium text-text">Plain RGB video</p>
                    <span class="text-[8px] px-1 py-0.5 rounded bg-accent/10 text-accent font-semibold ">training</span>
                  </div>
                  <p class="text-xs text-muted truncate font-mono">
                    {{ result!.plain_video ? result!.plain_video.split(/[\\/]/).pop() : 'not exported' }}
                  </p>
                </div>
              </div>
              <!-- Frame mapping + synced CSV -->
              <div v-for="[label, path] in [
                ['Frame mapping', result!.frame_mapping_csv],
                ['Synced CSV',    result!.synced_csv],
              ]" :key="label"
                class="flex items-center gap-2 py-1">
                <div class="w-1.5 h-1.5 rounded-full bg-success flex-shrink-0" />
                <div class="flex-1 min-w-0">
                  <div class="flex items-center gap-1.5">
                    <p class="text-xs font-medium text-text">{{ label }}</p>
                    <!-- "synced_60hz.csv" is a fixed artifact name, not a literal
                         rate — show the real measured fps so a ~30fps trial
                         (e.g. SUB02/03) is never mistaken for 60Hz. -->
                    <span v-if="label === 'Synced CSV' && result!.video_fps"
                      class="text-[8px] px-1 py-0.5 rounded font-semibold "
                      :class="isOffRate(result!.video_fps) ? 'bg-warning/10 text-warning' : 'bg-accent/10 text-accent'">
                      {{ result!.video_fps.toFixed(1) }} fps
                    </span>
                  </div>
                  <p class="text-xs text-muted truncate font-mono">{{ path.split(/[\\/]/).pop() }}</p>
                </div>
              </div>
              <!-- 2D projected CSV -->
              <div class="flex items-center gap-2 py-1">
                <div class="w-1.5 h-1.5 rounded-full flex-shrink-0"
                  :class="result!.synced2d_csv ? 'bg-success' : 'bg-border'" />
                <div class="flex-1 min-w-0">
                  <p class="text-xs font-medium text-text">2D projected CSV</p>
                  <p class="text-xs text-muted truncate font-mono">
                    {{ result!.synced2d_csv ? result!.synced2d_csv.split(/[\\/]/).pop() : 'not exported' }}
                  </p>
                </div>
              </div>
              <div class="flex items-center gap-2 py-1">
                <div class="w-1.5 h-1.5 rounded-full flex-shrink-0"
                  :class="result!.has_diagnostic_png ? 'bg-success' : 'bg-border'" />
                <div>
                  <p class="text-xs font-medium text-text">Diagnostic PNG</p>
                  <p class="text-xs text-muted">{{ result!.has_diagnostic_png ? 'diagnostic_sync.png' : 'not generated (no cross-correlation)' }}</p>
                </div>
              </div>
            </div>

            <!-- Output path -->
            <div class="bg-surface rounded-xl border border-border p-3">
              <p class="text-xs font-semibold text-muted  mb-1">Output directory</p>
              <p class="text-xs font-mono text-text break-all">{{ result!.output_dir }}</p>
            </div>

            <div v-if="result!.full_skeleton_pct < 80"
              class="flex items-start gap-2 bg-warning/10 border border-warning/30 rounded-xl p-3">
              <AlertCircle :size="13" class="text-warning flex-shrink-0 mt-0.5" />
              <p class="text-xs text-warning">
                Only {{ result!.full_skeleton_pct.toFixed(0) }}% of trimmed frames have full skeleton visibility.
                Consider adjusting the sync offset or checking marker occlusion.
              </p>
            </div>
          </div>

          <!-- Live preview panel -->
          <div v-else-if="showPreview" key="preview" class="rounded-xl border border-border overflow-hidden flex flex-col h-full">

            <!-- Header + frame navigation -->
            <div class="flex items-center justify-between px-4 py-2.5 border-b border-border">
              <div class="flex items-center gap-2">
                <Eye :size="13" class="text-accent" />
                <span class="text-xs font-semibold text-text">Live Preview</span>
                <span class="text-xs text-muted">— adjust offset, then Generate</span>
              </div>
              <!-- Frame stepper -->
              <div class="flex items-center gap-0.5">
                <button
                  v-for="d in [-10, -1]" :key="d"
                  class="px-1.5 py-0.5 rounded text-xs font-mono text-muted hover:text-text hover:bg-surface transition-colors"
                  @click="adjustFrame(d)">{{ d }}</button>
                <input
                  type="number"
                  :value="previewFrame"
                  min="0"
                  :max="taskInfo ? taskInfo.total_frames - 1 : 9999"
                  class="w-16 text-center bg-surface border border-border rounded px-1 py-0.5 text-xs font-mono text-text focus:outline-none focus:border-accent mx-0.5"
                  @change="(e) => { const v = Math.max(0, Math.min(Number((e.target as HTMLInputElement).value), taskInfo ? taskInfo.total_frames - 1 : 9999)); previewFrame = v; fetchPreview(true) }"
                />
                <button
                  v-for="d in [1, 10]" :key="d"
                  class="px-1.5 py-0.5 rounded text-xs font-mono text-muted hover:text-text hover:bg-surface transition-colors"
                  @click="adjustFrame(d)">+{{ d }}</button>
              </div>
            </div>

            <!-- Image area — flex-1 fills remaining height between header and status bar -->
            <div class="relative bg-black flex items-center justify-center flex-1 min-h-0">
              <div v-if="previewLoading"
                class="absolute inset-0 flex items-center justify-center bg-black/60 z-10">
                <div class="flex flex-col items-center gap-2">
                  <div class="w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
                  <span class="text-xs text-muted">Rendering…</span>
                </div>
              </div>
              <img v-if="previewImg" :src="previewImg"
                class="w-full h-full object-contain"
                :class="previewLoading ? 'opacity-40' : 'opacity-100'"
                style="transition: opacity 0.15s" />
              <div v-else-if="!previewLoading"
                class="flex flex-col items-center gap-2 text-muted">
                <Eye :size="32" class="opacity-20" />
                <span class="text-xs">Loading preview…</span>
              </div>
            </div>

            <!-- Status bar -->
            <div class="px-4 py-2 border-t border-border flex items-center justify-between bg-surface/50">
              <span class="text-xs font-mono text-muted">
                Frame <span class="text-text font-semibold">{{ previewFrame }}</span>
                /{{ taskInfo ? taskInfo.total_frames - 1 : '?' }}
                → Vicon <span class="text-text font-semibold">{{ previewViconT.toFixed(3) }}</span>s
              </span>
              <span class="text-xs font-semibold px-2 py-0.5 rounded-full"
                :class="previewInRange
                  ? 'text-success bg-success/10'
                  : 'text-danger bg-danger/10'">
                {{ previewInRange ? '● in range' : '● out of range' }}
              </span>
            </div>

            <!-- Hint -->
            <div class="px-4 py-2.5 border-t border-border">
              <p class="text-xs text-muted leading-relaxed">
                Use the <span class="text-text font-medium">Offset Fine-tune</span> controls (left panel) to shift the skeleton in time.
                Click <span class="text-text font-medium">Save offset</span> before generating the full video.
              </p>
            </div>
          </div>

          <!-- Empty fallback -->
          <div v-else key="empty" class="flex flex-col items-center justify-center h-full text-muted space-y-3">
            <Film :size="40" class="opacity-20" />
            <div class="text-center">
              <p class="text-sm font-medium">No overlay yet</p>
              <p class="text-xs mt-1">Select a synced task to preview, then click Generate</p>
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
