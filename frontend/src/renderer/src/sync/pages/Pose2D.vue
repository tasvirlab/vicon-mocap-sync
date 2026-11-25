<script setup lang="ts">
import { ref, computed, watch, onMounted, onBeforeUnmount } from 'vue'
import {
  Scan, Play, RotateCcw, AlertCircle, Info, CheckCircle, XCircle, Eye, ChevronDown,
  Video, Square,
} from '@lucide/vue'
import Button from '@renderer/sync/components/ui/Button.vue'
import PageHeader from '@renderer/sync/components/ui/PageHeader.vue'
import StatusBadge from '@renderer/sync/components/ui/StatusBadge.vue'
import ProgressSteps from '@renderer/sync/components/ui/ProgressSteps.vue'
import { useSyncJob } from '@renderer/sync/composables/useSyncJob'
import {
  listPose2DTasks, startPose2D, getPose2DResult, visualizePose2D, pose2dLiveUrl,
} from '@renderer/sync/lib/api'
import type { TrialListEntry, Pose2DResult } from '@renderer/sync/lib/types'

const { jobId, jobStatus, isRunning, error, startJob, reset: resetJob } = useSyncJob()

// ── Task selection ───────────────────────────────────────────────────────────────
const tasks = ref<TrialListEntry[]>([])
const selected = ref<Set<string>>(new Set())
const loadingTasks = ref(false)

// ── Parameters ───────────────────────────────────────────────────────────────────
const confThreshold = ref(0.3)
// Spine-ratio placeholders for the synthetic T10/L45 joints (lerp along C7→PELCEN,
// t=0 at C7, t=1 at the pelvis centre). Fitted from SUB01/lifting Vicon 3D — the
// earlier 0.45/0.85 guesses were each off by ~0.10 in opposite directions and
// compressed the modelled spine by ~34%. Keep in sync with DEFAULT_T_T10 /
// DEFAULT_T_L45 in backend/core/rtmpose_pipeline.py.
// [N=1] Fitted from one subject — refit across the full cohort when its C3D lands.
const tT10 = ref(0.35)
const tL45 = ref(0.95)
const device = ref<'cpu' | 'cuda'>('cuda')
const mode = ref<'lightweight' | 'balanced' | 'performance'>('balanced')

// ── Visualize ────────────────────────────────────────────────────────────────────
const visualizeTask = ref('')
const visualizeImg = ref<string | null>(null)
const visualizing = ref(false)
const visualizeError = ref<string | null>(null)

// ── Live preview (transient MJPEG stream, nothing saved) ──────────────────────────
const liveUrl = ref<string | null>(null)
const liveTask = ref('')
function startLive() {
  if (!visualizeTask.value) return
  liveTask.value = visualizeTask.value
  liveUrl.value = pose2dLiveUrl(
    visualizeTask.value, confThreshold.value, tT10.value, tL45.value, device.value, mode.value,
  )
}
function stopLive() { liveUrl.value = null }
onBeforeUnmount(stopLive)   // closing the <img> ends the stream + backend inference

// ── Results ──────────────────────────────────────────────────────────────────────
const result = ref<Pose2DResult | null>(null)
const fetchError = ref<string | null>(null)

async function loadTasks() {
  loadingTasks.value = true
  try {
    tasks.value = await listPose2DTasks()
  } catch { /* offline */ }
  finally { loadingTasks.value = false }
}
onMounted(loadTasks)

function toggle(name: string) {
  const s = new Set(selected.value)
  s.has(name) ? s.delete(name) : s.add(name)
  selected.value = s
}
function selectAll() { selected.value = new Set(tasks.value.map(t => t.task_name)) }
function selectNone() { selected.value = new Set() }

async function handleVisualize() {
  if (!visualizeTask.value) return
  visualizing.value = true
  visualizeError.value = null
  visualizeImg.value = null
  try {
    const r = await visualizePose2D(visualizeTask.value, confThreshold.value, tT10.value, tL45.value, device.value, mode.value)
    visualizeImg.value = `data:image/jpeg;base64,${r.image_b64}`
  } catch (e: unknown) {
    visualizeError.value = e instanceof Error ? e.message : String(e)
  } finally {
    visualizing.value = false
  }
}

async function handleRun() {
  if (selected.value.size === 0) return
  result.value = null
  fetchError.value = null
  try {
    const { job_id } = await startPose2D(
      [...selected.value], confThreshold.value, tT10.value, tL45.value, device.value, mode.value,
    )
    startJob(job_id, 'pose2d')
  } catch (err: unknown) {
    fetchError.value = `Failed to start: ${err instanceof Error ? err.message : String(err)}`
  }
}

function handleReset() {
  resetJob()
  result.value = fetchError.value = null
}

const canRun = computed(() => selected.value.size > 0 && !isRunning.value)
const displayError = computed(() => error.value || fetchError.value)
const showProgress = computed(() => isRunning.value && !!jobStatus.value)
const showResult = computed(() => !isRunning.value && !!result.value)
const showEmpty = computed(() => !isRunning.value && !result.value)

watch(() => jobStatus.value?.status, (status) => {
  if (status === 'completed' && jobId.value) {
    getPose2DResult(jobId.value)
      .then(r => { result.value = r; loadTasks() })
      .catch(e => { fetchError.value = `Failed to load result: ${e?.message}` })
  }
})
</script>

<template>
  <div class="p-6 animate-fade-in flex flex-col h-screen gap-5">

    <PageHeader title="2D Pose" :icon="Scan" color="#0EA5E9" :step="5"
      subtitle="Run RTMPose-WholeBody on the clean rgb.mp4 → 21-joint 2D keypoints (the pose model's input).">
      <template #actions>
        <Button v-if="result || isRunning" variant="ghost" size="sm" @click="handleReset">
          <template #icon><RotateCcw :size="13" /></template>
          Reset
        </Button>
      </template>
    </PageHeader>

    <div class="grid grid-cols-5 gap-5 flex-1 min-h-0">

      <!-- ── Left panel ──────────────────────────────────────────────────────── -->
      <div class="col-span-2 space-y-4 overflow-y-auto">

        <!-- Trial selection -->
        <div class="rounded-xl border border-border p-4 space-y-3">
          <div class="flex items-center justify-between">
            <p class="text-xs font-semibold text-muted ">Select Trials</p>
            <button class="text-xs text-accent hover:text-accent/80" :disabled="loadingTasks" @click="loadTasks">
              {{ loadingTasks ? 'Loading…' : 'Refresh' }}
            </button>
          </div>

          <div v-if="tasks.length === 0 && !loadingTasks" class="flex items-start gap-2 bg-surface rounded-lg p-2.5">
            <Info :size="12" class="text-muted mt-0.5 flex-shrink-0" />
            <p class="text-xs text-muted">No trials with rgb.mp4 found. Run Overlay first.</p>
          </div>

          <template v-else>
            <div class="flex gap-2 text-xs">
              <button class="text-accent hover:text-accent/80" @click="selectAll">Select all</button>
              <span class="text-muted">·</span>
              <button class="text-accent hover:text-accent/80" @click="selectNone">Clear</button>
              <span class="text-muted ml-auto">{{ selected.size }} / {{ tasks.length }} selected</span>
            </div>
            <div class="space-y-1.5 max-h-64 overflow-y-auto">
              <label v-for="task in tasks" :key="task.task_name"
                class="flex items-center gap-2.5 p-2.5 rounded-lg border cursor-pointer transition-colors"
                :class="selected.has(task.task_name) ? 'border-accent/50 bg-accent/10' : 'border-border hover:border-accent/30 bg-surface'">
                <input type="checkbox" class="w-3.5 h-3.5 accent-accent2"
                  :checked="selected.has(task.task_name)" @change="toggle(task.task_name)" />
                <span class="text-xs font-medium text-text truncate flex-1">{{ task.task_name }}</span>
                <span v-if="task.has_pose2d" class="text-xs px-1.5 py-0.5 rounded-full bg-success/10 text-success border border-success/30">has pose2d</span>
              </label>
            </div>
          </template>
        </div>

        <!-- Parameters -->
        <div class="rounded-xl border border-border p-4 space-y-3">
          <p class="text-xs font-semibold text-muted ">Parameters</p>

          <label class="block">
            <div class="flex justify-between text-xs text-muted mb-1">
              <span>Confidence threshold</span>
              <span class="font-mono text-text">{{ confThreshold }}</span>
            </div>
            <input type="range" v-model.number="confThreshold" min="0.05" max="0.9" step="0.05" :disabled="isRunning" class="w-full" />
          </label>

          <div class="grid grid-cols-2 gap-2">
            <label>
              <span class="text-xs text-muted">T10 spine ratio</span>
              <input type="number" v-model.number="tT10" step="0.01" min="0" max="1" :disabled="isRunning"
                class="mt-1 w-full bg-surface border border-border rounded-lg px-2 py-1.5 text-xs text-text focus:outline-none focus:border-accent disabled:opacity-50" />
            </label>
            <label>
              <span class="text-xs text-muted">L45 spine ratio</span>
              <input type="number" v-model.number="tL45" step="0.01" min="0" max="1" :disabled="isRunning"
                class="mt-1 w-full bg-surface border border-border rounded-lg px-2 py-1.5 text-xs text-text focus:outline-none focus:border-accent disabled:opacity-50" />
            </label>
          </div>
          <p class="text-xs text-muted">Placeholder ratios — Dataset Export calibrates these from real 3D data (train subjects only) and overrides them.</p>

          <div class="grid grid-cols-2 gap-2">
            <label>
              <span class="text-xs text-muted">Device</span>
              <div class="relative mt-1">
                <select v-model="device" :disabled="isRunning"
                  class="w-full bg-surface border border-border rounded-lg px-2 py-1.5 text-xs text-text appearance-none focus:outline-none focus:border-accent disabled:opacity-50">
                  <option value="cpu">CPU</option>
                  <option value="cuda">CUDA</option>
                </select>
                <ChevronDown :size="12" class="absolute right-2 top-1/2 -translate-y-1/2 text-muted pointer-events-none" />
              </div>
            </label>
            <label>
              <span class="text-xs text-muted">Mode</span>
              <div class="relative mt-1">
                <select v-model="mode" :disabled="isRunning"
                  class="w-full bg-surface border border-border rounded-lg px-2 py-1.5 text-xs text-text appearance-none focus:outline-none focus:border-accent disabled:opacity-50">
                  <option value="lightweight">Lightweight</option>
                  <option value="balanced">Balanced</option>
                  <option value="performance">Performance</option>
                </select>
                <ChevronDown :size="12" class="absolute right-2 top-1/2 -translate-y-1/2 text-muted pointer-events-none" />
              </div>
            </label>
          </div>
        </div>

        <!-- Verify + live preview -->
        <div class="rounded-xl border border-border p-4 space-y-3">
          <p class="text-xs font-semibold text-muted ">Verify Mapping First</p>
          <p class="text-xs text-muted">Check the joint mapping before running a full batch — multi-person lab footage can mis-track. <strong class="text-text">Check</strong> = first frame only. <strong class="text-text">Live</strong> = watch the whole trial detected frame-by-frame (nothing saved).</p>
          <select v-model="visualizeTask"
            class="w-full bg-surface border border-border rounded-lg px-2 py-1.5 text-xs text-text focus:outline-none focus:border-accent">
            <option value="">Select a trial…</option>
            <option v-for="t in tasks" :key="t.task_name" :value="t.task_name">{{ t.task_name }}</option>
          </select>
          <div class="grid grid-cols-2 gap-2">
            <Button size="sm" variant="ghost" :full-width="true" :disabled="!visualizeTask" :loading="visualizing" @click="handleVisualize">
              <template #icon><Eye :size="13" /></template>
              Check frame
            </Button>
            <Button size="sm" variant="ghost" :full-width="true" :disabled="!visualizeTask || !!liveUrl" @click="startLive">
              <template #icon><Video :size="13" /></template>
              Live preview
            </Button>
          </div>
          <p v-if="visualizeError" class="text-xs text-danger">{{ visualizeError }}</p>
          <img v-if="visualizeImg" :src="visualizeImg" class="w-full rounded-lg border border-border" />
        </div>

        <Button variant="primary" size="lg" :full-width="true" :disabled="!canRun" :loading="isRunning" @click="handleRun">
          <template v-if="!isRunning" #icon><Play :size="15" /></template>
          {{ isRunning ? 'Running RTMPose…' : `Run on ${selected.size} trial(s)` }}
        </Button>

        <div v-if="displayError" class="flex items-start gap-2 bg-danger/10 border border-danger/30 rounded-xl p-3">
          <AlertCircle :size="14" class="text-danger flex-shrink-0 mt-0.5" />
          <p class="text-xs text-danger leading-relaxed whitespace-pre-wrap">{{ displayError }}</p>
        </div>
      </div>

      <!-- ── Right panel ─────────────────────────────────────────────────────── -->
      <div class="col-span-3 h-full overflow-y-auto">
        <Transition name="pf" mode="out-in">

          <!-- Live skeleton preview (transient — nothing saved) -->
          <div v-if="liveUrl" key="live" class="rounded-xl border border-border p-4 space-y-3 h-full flex flex-col">
            <div class="flex items-center justify-between flex-shrink-0">
              <div class="flex items-center gap-2">
                <span class="flex items-center gap-1 text-xs font-bold text-danger">
                  <span class="w-2 h-2 rounded-full bg-danger animate-pulse-slow" /> LIVE
                </span>
                <span class="text-xs font-medium text-text truncate">{{ liveTask }}</span>
              </div>
              <Button size="sm" variant="danger" @click="stopLive">
                <template #icon><Square :size="12" /></template>
                Stop
              </Button>
            </div>
            <div class="flex-1 min-h-0 flex items-center justify-center bg-surface rounded-lg overflow-hidden">
              <img :src="liveUrl" class="max-w-full max-h-full object-contain" alt="Live 2D skeleton detection" />
            </div>
            <p class="text-xs text-muted flex-shrink-0">
              Detected frame-by-frame at model speed ({{ device.toUpperCase() }} · {{ mode }}) — not real-time video speed. Green bones/dots = confident, orange/red = below threshold. Loading the model takes a few seconds before the first frame.
            </p>
          </div>

          <div v-else-if="showProgress" key="prog" class="rounded-xl border border-border p-5 space-y-4">
            <div class="flex items-center justify-between">
              <span class="text-xs font-semibold text-muted ">Processing</span>
              <StatusBadge status="running" />
            </div>
            <ProgressSteps :job-status="jobStatus" />
          </div>

          <div v-else-if="showResult" key="res" class="space-y-4">
            <div class="bg-success/10 border border-success/30 rounded-xl p-4">
              <div class="flex items-center gap-2 mb-0.5">
                <CheckCircle :size="13" class="text-success" />
                <p class="text-sm font-semibold text-success">
                  {{ result!.n_succeeded }}/{{ result!.n_trials }} trials succeeded
                </p>
              </div>
              <p class="text-xs text-muted">{{ result!.processing_time_s }}s</p>
            </div>

            <div class="rounded-xl border border-border p-4 space-y-2">
              <p class="text-xs font-semibold text-muted ">Per-trial results</p>
              <div v-for="t in result!.trials" :key="t.task_name"
                class="flex items-center gap-2 p-2 rounded-lg bg-surface text-xs">
                <CheckCircle v-if="t.status === 'ok'" :size="13" class="text-success flex-shrink-0" />
                <XCircle v-else :size="13" class="text-danger flex-shrink-0" />
                <span class="text-text flex-1 truncate">{{ t.task_name }}</span>
                <span class="text-xs text-muted truncate max-w-[50%]">{{ t.output ?? t.error }}</span>
              </div>
            </div>
          </div>

          <div v-else key="empty" class="flex flex-col items-center justify-center h-full text-muted space-y-3">
            <Scan :size="40" class="opacity-20" />
            <div class="text-center">
              <p class="text-sm font-medium">No 2D poses extracted yet</p>
              <p class="text-xs mt-1">Select trials and click Run</p>
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
