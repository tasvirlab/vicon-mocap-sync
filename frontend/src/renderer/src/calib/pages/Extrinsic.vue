<script setup lang="ts">
import { ref, watch, computed } from 'vue'
import { Play, RotateCcw, ChevronDown, Check, Circle } from '@lucide/vue'
import Button             from '@renderer/calib/components/ui/Button.vue'
import Card               from '@renderer/calib/components/ui/Card.vue'
import AlertBox           from '@renderer/calib/components/ui/AlertBox.vue'
import BodyCalibPanel     from '@renderer/calib/components/ui/BodyCalibPanel.vue'
import DropZone           from '@renderer/calib/components/ui/DropZone.vue'
import ExtrinsicResultView from '@renderer/calib/components/ui/ExtrinsicResultView.vue'
import FormField          from '@renderer/calib/components/ui/FormField.vue'
import ProgressSteps      from '@renderer/calib/components/ui/ProgressSteps.vue'
import SegmentedControl   from '@renderer/calib/components/ui/SegmentedControl.vue'
import { useCalibrationJob } from '@renderer/calib/composables/useCalibrationJob'
import { useSettings }       from '@renderer/calib/composables/useSettings'
import { startExtrinsic, getExtrinsicResult, extrinsicFileUrl } from '@renderer/calib/lib/api'
import type { ExtrinsicResult } from '@renderer/calib/lib/types'

const SYNC_MODES    = ['manual', 'auto', 'timecode']
const ROTATE_OPTS   = ['none', 'ccw90', 'cw90', '180']
const extrinsicMode = ref<'board' | 'body'>('body')

const { settings } = useSettings()
const { jobId, jobStatus, isRunning, error, startJob, reset: resetJob } = useCalibrationJob()

const videoFile     = ref<File | null>(null)
const viconFile     = ref<File | null>(null)
const intrinsicFile = ref<File | null>(null)
const xcpFile       = ref<File | null>(null)
// Default Vicon timezone to the browser's local UTC offset (same lab as Vicon PC)
const viconTzOffset = ref<number>(-(new Date().getTimezoneOffset()) / 60)
const videoFrame    = ref(0)
const viconFrame    = ref(0)
const syncMode      = ref('manual')
// When .xcp is provided, auto-switch to timecode sync
const effectiveSyncMode = computed(() => xcpFile.value ? 'timecode' : syncMode.value)
const result        = ref<ExtrinsicResult | null>(null)
const fetchError    = ref<string | null>(null)

// ── Vicon outer-corner mapping ────────────────────────────────────────────────
// Maps each physical board corner (top_left / top_right / bottom_left /
// bottom_right) to the Vicon marker name placed there.
// Marker name = the part AFTER the colon: "LEFT 11" not "Test1:LEFT 11".
//
// The backend uses a two-step approach (board-only solvePnP → Kabsch rigid-body
// fit → compose) so the bottom corners do NOT need to be visible in the video.
const CORNER_KEYS = [
  { key: 'top_left',     label: 'Top-left corner' },
  { key: 'top_right',    label: 'Top-right corner' },
  { key: 'bottom_left',  label: 'Bottom-left corner' },
  { key: 'bottom_right', label: 'Bottom-right corner' },
] as const
type CornerKey = typeof CORNER_KEYS[number]['key']

const mappingOpen    = ref(false)
const outerMapping   = ref<Record<CornerKey, string>>({
  top_left:     'LEFT 11',
  top_right:    'Right 11',
  bottom_left:  'LEFT 21',
  bottom_right: 'Right 21',
})
const mappingEnabled = computed(() =>
  Object.values(outerMapping.value).some(v => v.trim() !== '')
)

function buildMarkerMapping(): Record<string, string> | undefined {
  const m: Record<string, string> = {}
  for (const { key } of CORNER_KEYS) {
    const name = outerMapping.value[key].trim()
    if (name) m[key] = name
  }
  return Object.keys(m).length > 0 ? m : undefined
}

watch(() => jobStatus.value?.status, (status) => {
  if (status === 'completed' && jobId.value) {
    fetchError.value = null
    getExtrinsicResult(jobId.value)
      .then(r => { result.value = r })
      .catch(err => { fetchError.value = `Failed to load result: ${err?.message ?? 'unknown error'}` })
  }
})

const canRun = computed(() => !!videoFile.value && !!viconFile.value && !!intrinsicFile.value && !isRunning.value)
const displayError = computed(() => error.value || fetchError.value)
const showProgress = computed(() => isRunning.value && !!jobStatus.value)
const showResult   = computed(() => !isRunning.value && !!result.value)
const showEmpty    = computed(() => !isRunning.value && !result.value && !displayError.value)

async function handleRun() {
  if (!videoFile.value || !viconFile.value || !intrinsicFile.value) return
  result.value     = null
  fetchError.value = null
  try {
    const markerMapping = buildMarkerMapping()
    const config: Record<string, unknown> = {
      subject_id: settings.subject_id,
      board:  settings.board,
      rotate: settings.extraction.rotate,
      sync:   { mode: effectiveSyncMode.value, video_frame: videoFrame.value, vicon_frame: viconFrame.value },
      solver: {
        use_ransac:                       settings.solver.use_ransac,
        ransac_reprojection_threshold_px: settings.solver.ransac_reprojection_threshold_px,
        ransac_confidence:                settings.solver.ransac_confidence,
        refine_lm:                        settings.solver.refine_lm,
        methods: ['ITERATIVE', 'EPNP', 'SQPNP'],
      },
      rms_warn_px: settings.calibration.rms_accept_threshold * 6,
      rms_bad_px:  settings.calibration.rms_warn_threshold  * 8,
    }
    // outer_corner_mapping triggers the two-step Kabsch calibration.
    // When empty, falls back to board-local mode (result is board-relative, not Vicon-world).
    if (markerMapping) config.outer_corner_mapping = markerMapping
    // Pass Vicon timezone when using XCP timestamp sync so local "ST" times convert to UTC correctly
    if (xcpFile.value) config.vicon_tz_offset_hours = viconTzOffset.value
    const { job_id } = await startExtrinsic(videoFile.value, viconFile.value, intrinsicFile.value, config, xcpFile.value)
    startJob(job_id, 'extrinsic')
  } catch (err: unknown) {
    fetchError.value = `Failed to start calibration: ${err instanceof Error ? err.message : String(err)}`
  }
}

function handleReset() {
  resetJob()
  videoFile.value     = null
  viconFile.value     = null
  intrinsicFile.value = null
  xcpFile.value       = null
  viconTzOffset.value = -(new Date().getTimezoneOffset()) / 60
  result.value        = null
  fetchError.value    = null
}

function openFile(filename: string) {
  if (!jobId.value) return
  ;(window as any).electron?.openExternal(extrinsicFileUrl(jobId.value, filename))
}

const MODE_OPTS = [{ value: 'board', label: 'Board mode' }, { value: 'body', label: 'Body mode' }]
const HOW = [
  { n: '01', t: 'Drop required files', b: 'Phone video, Vicon CSV or C3D, and the intrinsic parameters from module 1.' },
  { n: '02', t: 'Set synchronization', b: 'Enter the video and Vicon frame of the same clap or flash, or add an .xcp file for automatic timestamp sync.' },
  { n: '03', t: 'Run calibration', b: 'solvePnPRansac with Levenberg–Marquardt refinement gives R, t and the 4×4 world→camera transform.' },
]
const checklist = computed(() => [
  { label: 'Phone video', done: !!videoFile.value, note: 'MP4 / MOV / AVI' },
  { label: 'Vicon data', done: !!viconFile.value, note: 'CSV (Nexus export) or C3D' },
  { label: 'Intrinsic parameters', done: !!intrinsicFile.value, note: 'intrinsic_params.json or .npz' },
])
</script>

<template>
  <div class="animate-fade-in">
    <!-- Header -->
    <div class="flex items-end justify-between gap-6 mb-6">
      <div class="flex items-center gap-4">
        <span class="w-10 h-10 grid place-items-center border border-accent rounded text-accent font-mono text-base">02</span>
        <div>
          <h1 class="text-[32px] leading-[1.1] font-bold tracking-tight">Extrinsic Calibration</h1>
          <p class="text-[15px] text-muted mt-1.5">Find where the phone camera sits in the Vicon world.</p>
        </div>
      </div>
      <div class="flex items-center gap-3">
        <Button v-if="extrinsicMode === 'board' && (result || isRunning || displayError)" variant="ghost" @click="handleReset">
          <template #icon><RotateCcw :size="14" :stroke-width="1.5" /></template>Reset
        </Button>
        <SegmentedControl v-model="extrinsicMode" :options="MODE_OPTS" class="min-w-[230px]" />
      </div>
    </div>

    <BodyCalibPanel v-if="extrinsicMode === 'body'" />

    <div v-else class="grid grid-cols-[minmax(340px,400px)_minmax(0,1fr)] gap-6 items-start">
      <!-- ═════ Inputs ═════ -->
      <div class="flex flex-col gap-5">
        <Card class="flex flex-col gap-3.5">
          <h2 class="text-base font-semibold">Input files</h2>
          <DropZone compact label="Phone video" description="MP4 · MOV · AVI"
            accept="video/*" :file="videoFile" :disabled="isRunning" @update:file="videoFile = $event" />
          <DropZone compact label="Vicon data (CSV / C3D)" description="CSV (Nexus export) · C3D"
            accept=".csv,.c3d,.txt" :file="viconFile" :disabled="isRunning" @update:file="viconFile = $event" />
          <DropZone compact label="Intrinsic parameters" description="intrinsic_params.json or .npz"
            accept=".json,.npz" :file="intrinsicFile" :disabled="isRunning" @update:file="intrinsicFile = $event" />
          <DropZone compact label="Vicon session .xcp (optional)" description="Auto-aligns by wall-clock time"
            accept=".xcp" :file="xcpFile" :disabled="isRunning" @update:file="xcpFile = $event" />

          <div class="flex flex-col gap-1.5">
            <span class="text-xs font-medium text-muted">Video rotation</span>
            <SegmentedControl mono :model-value="settings.extraction.rotate" :options="ROTATE_OPTS" :disabled="isRunning"
              @update:model-value="settings.extraction.rotate = $event" />
            <span class="text-xs text-muted"><span class="font-mono text-text">ccw90</span> for iPhone portrait, <span class="font-mono text-text">none</span> for landscape.</span>
          </div>

          <template v-if="xcpFile">
            <AlertBox tone="info"><strong>Timestamp sync active.</strong> The Vicon frame for ChArUco detection is computed from the wall-clock offset.</AlertBox>
            <FormField label="Session timezone offset" type="number" :step="0.5" unit="h" v-model="viconTzOffset"
              :disabled="isRunning" hint="Vicon PC UTC offset, e.g. +9 for KST or JST." />
          </template>
        </Card>

        <Card class="flex flex-col gap-3.5">
          <h2 class="text-base font-semibold">Synchronization</h2>
          <SegmentedControl v-model="syncMode" :options="SYNC_MODES" :disabled="isRunning" class="capitalize" />
          <div class="grid grid-cols-2 gap-3.5">
            <FormField label="Video frame #" type="number" :min="0" v-model="videoFrame" :disabled="isRunning" />
            <FormField label="Vicon frame #" type="number" :min="0" v-model="viconFrame" :disabled="isRunning" />
          </div>
          <p class="text-xs text-muted leading-normal">Set both frame indices to the same clap or flash visible in both the video and the Vicon trajectories.</p>
        </Card>

        <section class="border border-border rounded px-6 py-4">
          <button class="flex w-full justify-between items-center" :aria-expanded="mappingOpen" @click="mappingOpen = !mappingOpen">
            <span class="flex items-center gap-2.5">
              <h2 class="text-base font-semibold">Vicon marker mapping</h2>
              <span v-if="mappingEnabled" class="text-xs font-medium text-accent border border-accent rounded-sm px-1.5 py-px">active</span>
            </span>
            <ChevronDown :size="16" :stroke-width="1.5" :class="['text-muted transition-transform', mappingOpen && 'rotate-180']" />
          </button>
          <div v-if="mappingOpen" class="mt-3.5 flex flex-col gap-3.5">
            <AlertBox tone="info">
              Enter the Vicon marker placed at each physical board corner, using the short name after the colon
              (<span class="font-mono">LEFT 11</span>, not <span class="font-mono">Test1:LEFT 11</span>). The backend uses a Kabsch two-step fit,
              so the bottom corners need not be visible in the video. Set the Vicon frame to one where all four markers have data.
            </AlertBox>
            <div class="border border-border rounded p-3 font-mono text-xs select-none flex flex-col gap-1.5">
              <div class="text-center text-muted font-sans">Board corners (as printed, face-on)</div>
              <div class="flex justify-between px-3 text-accent"><span>top-left</span><span>top-right</span></div>
              <div class="flex justify-between px-3 text-accent"><span>bottom-left</span><span>bottom-right</span></div>
            </div>
            <label v-for="c in CORNER_KEYS" :key="c.key" class="flex flex-col gap-1.5">
              <span class="text-xs font-medium text-muted">{{ c.label }}</span>
              <input v-model="outerMapping[c.key]" type="text" :placeholder="`Vicon marker for ${c.label.toLowerCase()}`" :disabled="isRunning"
                class="h-9 bg-surface border border-border rounded px-2.5 font-mono text-sm outline-none focus:border-accent disabled:opacity-45" />
            </label>
          </div>
        </section>

        <AlertBox v-if="displayError && !showResult" tone="error">{{ displayError }}</AlertBox>

        <Button variant="primary" size="lg" full-width :disabled="!canRun" :loading="isRunning" @click="handleRun">
          <template v-if="!isRunning" #icon><Play :size="18" :stroke-width="1.5" /></template>
          {{ isRunning ? 'Aligning…' : 'Run Extrinsic Calibration' }}
        </Button>
      </div>

      <!-- ═════ Output ═════ -->
      <div class="flex flex-col gap-5 min-w-0">
        <template v-if="showProgress">
          <Card class="!p-6 flex flex-col gap-5">
            <h2 class="text-base font-semibold">Solving pose…</h2>
            <ProgressSteps :job-status="jobStatus" />
          </Card>
          <div class="grid grid-cols-[1.6fr_1fr_1.2fr] gap-5"><div v-for="i in 3" :key="i" class="h-28 rounded shimmer" /></div>
          <div class="h-[300px] rounded shimmer" />
        </template>

        <Card v-else-if="displayError && !result" class="!border-danger !p-7 flex flex-col gap-4">
          <div>
            <div class="text-2xl font-bold text-danger">Calibration failed</div>
            <p class="mt-1.5 text-[15px] leading-normal">{{ displayError }}</p>
          </div>
          <div class="flex gap-2.5">
            <Button variant="primary" :disabled="!canRun" @click="handleRun">Retry</Button>
            <Button variant="secondary" @click="handleReset">Reset</Button>
          </div>
        </Card>

        <ExtrinsicResultView v-else-if="showResult" :result="result" :subject="settings.subject_id"
          :caption="`method ${result!.method}`"
          :downloads="[{ label: 'Reprojection plot', file: 'reprojection_plot.png' }, { label: 'JSON', file: 'extrinsic_params.json' }, { label: 'NPZ', file: 'extrinsic_params.npz' }]"
          :third-metric="{ label: 'PnP method', value: result!.method, subtitle: 'best solver' }"
          :overlay-url="jobId ? extrinsicFileUrl(jobId, 'overlay_frame.jpg') : undefined"
          marker-title="Per-marker error" :warn-at="3" :bad-at="8" @open="openFile">
          <Card v-if="(result as any)?.detected_corner_ids?.length" class="!p-[22px]" title="Detected corners">
            <div class="text-xs text-muted mb-2.5">Detected ChArUco corner IDs in frame <span class="font-mono text-text">{{ (result as any).detected_frame_idx }}</span>. If a mapped ID is missing here, update the marker mapping.</div>
            <div class="flex flex-wrap gap-1.5">
              <span v-for="id in (result as any).detected_corner_ids" :key="id" class="px-2 py-0.5 rounded-sm text-xs font-mono border border-border text-muted">{{ id }}</span>
            </div>
          </Card>
        </ExtrinsicResultView>

        <template v-else-if="showEmpty">
          <Card class="!p-7 flex flex-col gap-5">
            <h2 class="text-base font-semibold">Output · How it works</h2>
            <div class="grid grid-cols-3 border border-border rounded overflow-hidden">
              <div v-for="(h, i) in HOW" :key="h.n" :class="['p-5 flex flex-col gap-2.5', i < 2 && 'border-r border-border']">
                <span class="font-mono text-[13px] text-accent">{{ h.n }}</span>
                <div class="text-lg font-semibold">{{ h.t }}</div>
                <div class="text-sm text-muted leading-normal">{{ h.b }}</div>
              </div>
            </div>
          </Card>
          <Card title="Required files">
            <ul class="flex flex-col gap-2">
              <li v-for="item in checklist" :key="item.label" class="flex items-center gap-2.5 text-sm">
                <Check v-if="item.done" :size="16" :stroke-width="2" class="text-success" />
                <Circle v-else :size="16" :stroke-width="1.5" class="text-dim" />
                <span :class="item.done ? 'font-medium' : 'text-muted'">{{ item.label }}</span>
                <span class="text-xs text-dim">{{ item.note }}</span>
              </li>
            </ul>
          </Card>
        </template>
      </div>
    </div>
  </div>
</template>
