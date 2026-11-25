<script setup lang="ts">
/**
 * BodyCalibPanel — body-marker extrinsic calibration workflow.
 *
 * Flow:
 *   1. Drop video + Vicon CSV + intrinsic JSON
 *   2. Select 6 or 8 joint mode
 *   3. Navigate frames; click joints on ClickCanvas (with magnifier)
 *   4. Accept each frame → list builds up
 *   5. "Solve" button → backend job → results
 */
import { ref, computed, watch } from 'vue'
import { Play, ChevronLeft, ChevronRight, RotateCcw } from '@lucide/vue'
import Button        from '@renderer/calib/components/ui/Button.vue'
import Card          from '@renderer/calib/components/ui/Card.vue'
import AlertBox      from '@renderer/calib/components/ui/AlertBox.vue'
import DropZone      from '@renderer/calib/components/ui/DropZone.vue'
import ClickCanvas   from '@renderer/calib/components/ui/ClickCanvas.vue'
import ExtrinsicResultView from '@renderer/calib/components/ui/ExtrinsicResultView.vue'
import FormField     from '@renderer/calib/components/ui/FormField.vue'
import ProgressSteps from '@renderer/calib/components/ui/ProgressSteps.vue'
import SegmentedControl from '@renderer/calib/components/ui/SegmentedControl.vue'
import { useCalibrationJob } from '@renderer/calib/composables/useCalibrationJob'
import { useSettings } from '@renderer/calib/composables/useSettings'
import {
  extractBodyFrame,
  startBodyExtrinsic,
  getBodyResult,
  bodyFileUrl,
} from '@renderer/calib/lib/api'

interface Pt { x: number; y: number }

const { settings } = useSettings()

// ── Files + rotation ─────────────────────────────────────────────────────────
const videoFile     = ref<File | null>(null)
const viconFile     = ref<File | null>(null)
const intrinsicFile = ref<File | null>(null)
const xcpFile       = ref<File | null>(null)
const viconTzOffset = ref<number>(-(new Date().getTimezoneOffset()) / 60)
const bodyRotate    = ref('none')

// ── Joint mode ────────────────────────────────────────────────────────────────
const jointCount  = ref<6 | 8>(6)
const JOINTS_6    = ['LSHO', 'RSHO', 'LKNE', 'RKNE', 'LANK', 'RANK']
const JOINTS_8    = ['LSHO', 'RSHO', 'LHIP', 'RHIP', 'LKNE', 'RKNE', 'LANK', 'RANK']
const activeJoints = computed(() => jointCount.value === 8 ? JOINTS_8 : JOINTS_6)

// ── Frame navigation ──────────────────────────────────────────────────────────
const frameIdx    = ref(0)
const totalFrames = ref(0)
const videoFps    = ref(60)
const frameImage  = ref('')
const frameCache  = ref<Map<number, string>>(new Map())
const loadingFrame = ref(false)
const frameError  = ref('')

// ── Click state ───────────────────────────────────────────────────────────────
const currentClicks = ref<Pt[]>([])
const acceptedFrames = ref<Map<number, Pt[]>>(new Map())

// ── Job / result ──────────────────────────────────────────────────────────────
const { jobId, jobStatus, isRunning, error: jobError, startJob, reset: resetJob }
  = useCalibrationJob()
const result    = ref<any>(null)
const fetchErr  = ref('')

// ── Computed ──────────────────────────────────────────────────────────────────
const allFramesReady = computed(() =>
  !!videoFile.value && !!viconFile.value && !!intrinsicFile.value)

const canSolve = computed(() =>
  allFramesReady.value && acceptedFrames.value.size >= 3 && !isRunning.value)

const acceptedCount = computed(() => acceptedFrames.value.size)
const currentComplete = computed(
  () => currentClicks.value.length === activeJoints.value.length)

const displayError = computed(() => jobError.value || fetchErr.value)

// ── Frame loading ─────────────────────────────────────────────────────────────
async function fetchFrame(idx: number) {
  if (!videoFile.value) return
  frameError.value = ''

  if (frameCache.value.has(idx)) {
    frameImage.value = frameCache.value.get(idx)!
    currentClicks.value = acceptedFrames.value.get(idx)
      ? [...(acceptedFrames.value.get(idx)!)] : []
    return
  }
  loadingFrame.value = true
  try {
    const res = await extractBodyFrame(videoFile.value, idx, bodyRotate.value)
    frameCache.value.set(idx, res.frame_b64)
    totalFrames.value = res.total_frames
    videoFps.value    = res.fps
    frameImage.value  = res.frame_b64
    currentClicks.value = acceptedFrames.value.get(idx)
      ? [...(acceptedFrames.value.get(idx)!)] : []
  } catch (e: any) {
    frameError.value = `Frame load failed: ${e?.message ?? String(e)}`
  } finally {
    loadingFrame.value = false
  }
}

watch(videoFile, async (f) => {
  if (f) {
    frameCache.value.clear()
    acceptedFrames.value.clear()
    currentClicks.value = []
    frameIdx.value = 0
    await fetchFrame(0)
  } else {
    frameImage.value = ''
  }
})

// Rotation change → clear cache so frames re-fetch with new rotation
watch(bodyRotate, () => {
  frameCache.value.clear()
  if (videoFile.value) fetchFrame(frameIdx.value)
})

function prevFrame() {
  if (frameIdx.value > 0) { frameIdx.value--; fetchFrame(frameIdx.value) }
}
function nextFrame() {
  if (frameIdx.value < totalFrames.value - 1)
    { frameIdx.value++; fetchFrame(frameIdx.value) }
}
function scrubFrame(e: Event) {
  frameIdx.value = parseInt((e.target as HTMLInputElement).value)
  fetchFrame(frameIdx.value)
}

// ── Accept / remove frame ─────────────────────────────────────────────────────
function acceptFrame() {
  if (!currentComplete.value) return
  acceptedFrames.value.set(frameIdx.value, [...currentClicks.value])
}
function removeFrame(idx: number) {
  acceptedFrames.value.delete(idx)
}

// ── Solve ─────────────────────────────────────────────────────────────────────
async function handleSolve() {
  if (!videoFile.value || !viconFile.value || !intrinsicFile.value) return
  result.value = null; fetchErr.value = ''

  // Build clicks dict: {frame_idx_str: [[x,y],...]}
  const clicks: Record<string, [number, number][]> = {}
  acceptedFrames.value.forEach((pts, fidx) => {
    clicks[String(fidx)] = pts.map(p => [p.x, p.y])
  })

  try {
    const { job_id } = await startBodyExtrinsic(
      videoFile.value, viconFile.value, intrinsicFile.value,
      clicks, jointCount.value,
      xcpFile.value,
      viconTzOffset.value,
      bodyRotate.value,
      settings.subject_id)
    startJob(job_id, 'extrinsic-body')
  } catch (e: any) {
    fetchErr.value = `Failed to start: ${e?.message ?? String(e)}`
  }
}

watch(() => jobStatus.value?.status, (s) => {
  if (s === 'completed' && jobId.value) {
    getBodyResult(jobId.value)
      .then(r => { result.value = r })
      .catch(e => { fetchErr.value = `Failed to load result: ${e?.message}` })
  }
})

// ── Reset ─────────────────────────────────────────────────────────────────────
function handleReset() {
  resetJob()
  videoFile.value = null; viconFile.value = null; intrinsicFile.value = null
  xcpFile.value = null
  frameImage.value = ''; frameCache.value.clear()
  acceptedFrames.value.clear(); currentClicks.value = []
  result.value = null; fetchErr.value = ''
}

const ROTATIONS = ['none', 'ccw90', 'cw90', '180']
const JOINT_MODES = [{ value: '6', label: '6 joints' }, { value: '8', label: '8 joints' }]
const nextJoint = computed(() => activeJoints.value[currentClicks.value.length] ?? 'all marked')
const acceptedList = computed(() => [...acceptedFrames.value].sort(([a], [b]) => a - b))
function setJointCount(v: string) {
  jointCount.value = v === '8' ? 8 : 6
  currentClicks.value = []
  acceptedFrames.value.clear()
}

function openFile(filename: string) {
  if (!jobId.value) return
  ;(window as any).electron?.openExternal(bodyFileUrl(jobId.value, filename))
}
</script>

<template>
  <div class="grid grid-cols-[minmax(340px,400px)_minmax(0,1fr)] gap-6 items-start">
    <!-- ═════ Inputs ═════ -->
    <div class="flex flex-col gap-5">
      <Card class="flex flex-col gap-3.5">
        <h2 class="text-base font-semibold">Input files</h2>
        <FormField label="Subject ID" v-model="settings.subject_id" placeholder="SUB01" :disabled="isRunning"
          hint="Output goes to exports/<Subject ID>/." />
        <DropZone compact label="Phone video" description="MP4 · MOV · AVI" accept="video/*"
          :file="videoFile" :disabled="isRunning" @update:file="videoFile = $event" />
        <DropZone compact label="Vicon data (CSV / C3D)" description="CSV (Nexus export) · C3D" accept=".csv,.c3d,.txt"
          :file="viconFile" :disabled="isRunning" @update:file="viconFile = $event" />
        <DropZone compact label="Intrinsic parameters" description="intrinsic.json / intrinsic_params.json / .npz" accept=".json,.npz"
          :file="intrinsicFile" :disabled="isRunning" @update:file="intrinsicFile = $event" />
        <DropZone compact label="Vicon session .xcp (optional)" description="Wall-clock sync, improves accuracy" accept=".xcp"
          :file="xcpFile" :disabled="isRunning" @update:file="xcpFile = $event" />
        <template v-if="xcpFile">
          <AlertBox tone="info"><strong>Timestamp sync active.</strong> Narrows the sync search to a ±2 s window.</AlertBox>
          <FormField label="Session timezone offset" type="number" :step="0.5" unit="h" v-model="viconTzOffset"
            :disabled="isRunning" hint="Vicon PC UTC offset, e.g. +5 for UTC+5." />
        </template>
        <div class="flex flex-col gap-1.5">
          <span class="text-xs font-medium text-muted">Video rotation</span>
          <SegmentedControl mono v-model="bodyRotate" :options="ROTATIONS" :disabled="isRunning" />
          <span class="text-xs text-muted"><span class="font-mono text-text">ccw90</span> for iPhone portrait.</span>
        </div>
      </Card>

      <Card class="flex flex-col gap-3.5">
        <h2 class="text-base font-semibold">Joint mode</h2>
        <SegmentedControl :model-value="String(jointCount)" :options="JOINT_MODES" @update:model-value="setJointCount" />
        <div class="flex flex-wrap gap-1.5">
          <span v-for="(j, i) in activeJoints" :key="j" class="px-2 py-0.5 rounded-sm text-xs font-mono border border-border text-muted">{{ i + 1 }}.{{ j }}</span>
        </div>
        <p class="text-xs text-muted leading-normal">{{ jointCount }} joints per frame. Click them in order. 6 joints: shoulders, knees, ankles (use when hips are occluded). 8 joints add the hips and need hip markers visible.</p>
      </Card>

      <Card class="flex flex-col gap-3">
        <div class="flex justify-between items-center">
          <h2 class="text-base font-semibold">Accepted frames</h2>
          <span :class="['font-mono text-[13px]', acceptedCount >= 3 ? 'text-success' : 'text-muted']">{{ acceptedCount }} / 3 min</span>
        </div>
        <div v-if="acceptedCount === 0" class="border border-dashed border-dim rounded p-4 text-center text-[13px] text-muted">
          No frames yet. Mark all joints, then accept the frame.
        </div>
        <div v-else class="flex flex-col gap-1.5 max-h-40 overflow-y-auto">
          <div v-for="[fidx, pts] in acceptedList" :key="fidx" class="flex justify-between items-center px-3 py-2 border border-border rounded font-mono text-[13px]">
            <span>frame {{ fidx }} · {{ pts.length }} pts</span>
            <button class="text-muted hover:text-danger font-sans text-[13px]" @click="removeFrame(fidx)">Remove</button>
          </div>
        </div>
      </Card>

      <AlertBox v-if="displayError && !result" tone="error">{{ displayError }}</AlertBox>

      <Button variant="primary" size="lg" full-width :disabled="!canSolve" :loading="isRunning" @click="handleSolve">
        <template v-if="!isRunning" #icon><Play :size="18" :stroke-width="1.5" /></template>
        {{ isRunning ? 'Solving…' : 'Solve Body Extrinsic' }}
      </Button>
      <Button v-if="result || isRunning" variant="ghost" @click="handleReset">
        <template #icon><RotateCcw :size="14" :stroke-width="1.5" /></template>Reset
      </Button>
    </div>

    <!-- ═════ Canvas + output ═════ -->
    <div class="flex flex-col gap-5 min-w-0">
      <Card v-if="frameImage || loadingFrame" class="flex flex-col gap-3.5">
        <div class="flex justify-between items-center flex-wrap gap-2.5">
          <h2 class="text-base font-semibold">Click joints · <span class="text-accent font-mono">{{ nextJoint }}</span></h2>
          <div class="flex items-center gap-2">
            <Button :disabled="frameIdx === 0" aria-label="Previous frame" @click="prevFrame"><ChevronLeft :size="16" :stroke-width="1.5" /></Button>
            <span class="font-mono text-[13px] min-w-[110px] text-center">frame {{ frameIdx }} / {{ Math.max(0, totalFrames - 1) }}</span>
            <Button :disabled="frameIdx >= totalFrames - 1" aria-label="Next frame" @click="nextFrame"><ChevronRight :size="16" :stroke-width="1.5" /></Button>
          </div>
        </div>
        <input type="range" :min="0" :max="Math.max(0, totalFrames - 1)" :value="frameIdx" class="w-full" :disabled="loadingFrame" aria-label="Frame" @input="scrubFrame" />

        <div v-if="loadingFrame" class="h-48 grid place-items-center text-muted text-sm">Loading frame…</div>
        <ClickCanvas v-else :frame-image="frameImage" :joints="activeJoints" v-model="currentClicks" />

        <div class="flex items-center gap-3">
          <Button variant="primary" class="flex-1" :disabled="!currentComplete || isRunning" @click="acceptFrame">
            {{ acceptedFrames.has(frameIdx) ? 'Re-accept frame ✓' : 'Accept frame' }}
          </Button>
          <span class="text-xs text-muted">Click {{ activeJoints.length }} joints, then accept</span>
        </div>
        <p v-if="frameError" class="text-xs text-danger">{{ frameError }}</p>
      </Card>

      <Card v-else-if="!videoFile" class="!border-dashed !border-dim h-64 flex flex-col items-center justify-center text-center gap-2">
        <div class="text-lg font-semibold">Drop a video to start</div>
        <div class="text-sm text-muted">Navigate frames → click joints → accept → solve</div>
      </Card>

      <template v-if="isRunning && jobStatus">
        <Card class="!p-6 flex flex-col gap-5">
          <h2 class="text-base font-semibold">Solving body pose…</h2>
          <ProgressSteps :job-status="jobStatus" />
        </Card>
        <div class="grid grid-cols-[1.6fr_1fr_1.2fr] gap-5"><div v-for="i in 3" :key="i" class="h-28 rounded shimmer" /></div>
      </template>

      <ExtrinsicResultView v-if="result && !isRunning" :result="result" :subject="settings.subject_id"
        :caption="`${result.n_inliers}/${result.n_points} inliers · sync Δt ${result.sync_offset_s > 0 ? '+' : ''}${result.sync_offset_s}s${result.sync_method === 'TIMESTAMP+FINE' ? ' · XCP' : ''}`"
        :downloads="[{ label: 'JSON', file: 'extrinsic_params.json' }, { label: 'NPZ', file: 'extrinsic_params.npz' }]"
        :third-metric="{ label: 'Joints used', value: String(result.joint_count), subtitle: 'body markers' }"
        marker-title="Per-joint error" :warn-at="2" :bad-at="5" @open="openFile" />
    </div>
  </div>
</template>
