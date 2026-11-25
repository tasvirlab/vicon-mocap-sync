<script setup lang="ts">
import { ref, watch, computed } from 'vue'
import { Play, RotateCcw, Download, FileText } from '@lucide/vue'
import Button            from '@renderer/calib/components/ui/Button.vue'
import Card              from '@renderer/calib/components/ui/Card.vue'
import AlertBox          from '@renderer/calib/components/ui/AlertBox.vue'
import DropZone          from '@renderer/calib/components/ui/DropZone.vue'
import FormField         from '@renderer/calib/components/ui/FormField.vue'
import MetricsCard       from '@renderer/calib/components/ui/MetricsCard.vue'
import MetricsGrid       from '@renderer/calib/components/ui/MetricsGrid.vue'
import ProgressSteps     from '@renderer/calib/components/ui/ProgressSteps.vue'
import QualityBadge      from '@renderer/calib/components/ui/QualityBadge.vue'
import SegmentedControl  from '@renderer/calib/components/ui/SegmentedControl.vue'
import ReprojectionChart from '@renderer/calib/components/visualization/ReprojectionChart.vue'
import { useCalibrationJob } from '@renderer/calib/composables/useCalibrationJob'
import { useSettings }       from '@renderer/calib/composables/useSettings'
import { startIntrinsic, startIntrinsicFromImages, getIntrinsicResult, intrinsicFileUrl } from '@renderer/calib/lib/api'
import { qualityColor, qualityLevel } from '@renderer/calib/lib/types'
import type { IntrinsicResult } from '@renderer/calib/lib/types'

type InputMode = 'video' | 'images'

const ROTATIONS = ['none', 'ccw90', 'cw90', '180']
const MODES = [{ value: 'video', label: 'Video' }, { value: 'images', label: 'Images' }]
const QUALITY_MSG: Record<string, string> = {
  excellent:  'Excellent — a precise lens model.',
  good:       'Good — suitable for most work.',
  acceptable: 'Acceptable — usable; consider recording more angles.',
  poor:       'Poor — recalibrate before using these parameters.',
}
const HOW = [
  { n: '01', t: 'Drop your video',      b: 'Record the ChArUco board from many angles and distances, as one video or a set of images.' },
  { n: '02', t: 'Set board parameters', b: 'Match the ArUco dictionary, grid size and the physical square and marker sizes of your print.' },
  { n: '03', t: 'Run calibration',      b: 'Frames are extracted, corners detected, and cv2.calibrateCamera estimates K and distortion.' },
]
const TIPS = [
  'Film 30–50 seconds, varying tilt and distance',
  'Even lighting, no glare on the board',
  'Aim for RMS below 0.5 px (Excellent)',
  'iPhone portrait video: set rotation to ccw90',
]

const DICT_OPTIONS = [
  'DICT_4X4_50', 'DICT_4X4_100', 'DICT_5X5_50',
  'DICT_6X6_250', 'DICT_7X7_250', 'DICT_ARUCO_ORIGINAL',
]

const { settings } = useSettings()
const { jobId, jobStatus, isRunning, error, startJob, reset: resetJob } = useCalibrationJob()

const inputMode   = ref<InputMode>('video')
const videoFile   = ref<File | null>(null)
const imageFiles  = ref<File[]>([])
const result      = ref<IntrinsicResult | null>(null)
const fetchError  = ref<string | null>(null)
const imgInputRef = ref<HTMLInputElement | null>(null)
const imgDrag     = ref(false)

watch(() => jobStatus.value?.status, (status) => {
  if (status === 'completed' && jobId.value) {
    fetchError.value = null
    getIntrinsicResult(jobId.value)
      .then(r => { result.value = r })
      .catch(err => { fetchError.value = `Failed to load result: ${err?.message ?? 'unknown error'}` })
  }
})

const canRun = computed(() =>
  inputMode.value === 'video'
    ? !!videoFile.value && !isRunning.value
    : imageFiles.value.length > 0 && !isRunning.value
)
const displayError = computed(() => error.value || fetchError.value)
const showProgress = computed(() => isRunning.value && !!jobStatus.value)
const showResult   = computed(() => !isRunning.value && !!result.value)
const showEmpty    = computed(() => !isRunning.value && !result.value && !displayError.value)

async function handleRun() {
  result.value     = null
  fetchError.value = null
  try {
    const config = {
      subject_id: settings.subject_id,
      board:      settings.board,
      extraction: settings.extraction,
      calibration: settings.calibration,
      rotate:     settings.extraction.rotate ?? 'ccw90',
    }
    if (inputMode.value === 'video') {
      if (!videoFile.value) return
      const { job_id } = await startIntrinsic(videoFile.value, config)
      startJob(job_id, 'intrinsic')
    } else {
      if (!imageFiles.value.length) return
      const { job_id } = await startIntrinsicFromImages(imageFiles.value, config)
      startJob(job_id, 'intrinsic')
    }
  } catch (err: unknown) {
    fetchError.value = `Failed to start calibration: ${err instanceof Error ? err.message : String(err)}`
  }
}

function handleReset() {
  resetJob()
  videoFile.value  = null
  imageFiles.value = []
  result.value     = null
  fetchError.value = null
}

function switchMode(m: InputMode) {
  inputMode.value  = m
  videoFile.value  = null
  imageFiles.value = []
}

function handleImagePick(e: Event) {
  const files = (e.target as HTMLInputElement).files
  if (files) imageFiles.value = Array.from(files)
}

function handleImageDrop(e: DragEvent) {
  e.preventDefault()
  imgDrag.value = false
  const files = Array.from(e.dataTransfer?.files ?? []).filter(f => /\.(jpe?g|png|bmp|tiff?)$/i.test(f.name))
  if (files.length) imageFiles.value = [...imageFiles.value, ...files]
}

function openFile(filename: string) {
  if (!jobId.value) return
  ;(window as any).electron?.openExternal(intrinsicFileUrl(jobId.value, filename))
}
</script>

<template>
  <div class="animate-fade-in">
    <!-- Header -->
    <div class="flex items-end justify-between gap-6 mb-6">
      <div class="flex items-center gap-4">
        <span class="w-10 h-10 grid place-items-center border border-accent rounded text-accent font-mono text-base">01</span>
        <div>
          <h1 class="text-[32px] leading-[1.1] font-bold tracking-tight">Intrinsic Calibration</h1>
          <p class="text-[15px] text-muted mt-1.5">Estimate camera matrix and lens distortion from a ChArUco board recording.</p>
        </div>
      </div>
      <Button v-if="result || isRunning || displayError" variant="ghost" @click="handleReset">
        <template #icon><RotateCcw :size="14" :stroke-width="1.5" /></template>
        Reset
      </Button>
    </div>

    <div class="grid grid-cols-[minmax(340px,400px)_minmax(0,1fr)] gap-6 items-start">
      <!-- ═════ Inputs ═════ -->
      <Card class="flex flex-col gap-[22px]">
        <h2 class="text-base font-semibold -mb-1">Inputs</h2>

        <FormField label="Subject ID" v-model="settings.subject_id" placeholder="SUB01"
          :disabled="isRunning" hint="Leave blank to save at the exports root." />

        <div class="flex flex-col gap-1.5">
          <span class="text-xs font-medium text-muted">Source</span>
          <SegmentedControl :model-value="inputMode" :options="MODES" :disabled="isRunning"
            @update:model-value="switchMode($event as InputMode)" />
        </div>

        <DropZone
          v-if="inputMode === 'video'"
          label="Drag ChArUco video here"
          description="MP4 · MOV · AVI · MKV"
          accept="video/*,.mp4,.mov,.avi,.mkv"
          :file="videoFile" :disabled="isRunning"
          @update:file="videoFile = $event"
        />
        <div v-else class="flex flex-col gap-2.5">
          <div
            :class="['flex flex-col items-center gap-2 text-center px-4 py-5 border rounded cursor-pointer transition-colors',
              imgDrag ? 'drop-zone-active border-dashed' : imageFiles.length ? 'border-solid border-accent' : 'border-dashed border-dim hover:border-muted']"
            @dragover.prevent="imgDrag = true" @dragleave="imgDrag = false" @drop="handleImageDrop" @click="imgInputRef?.click()"
          >
            <input ref="imgInputRef" type="file" multiple accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff" class="sr-only" :disabled="isRunning" @change="handleImagePick" />
            <template v-if="imageFiles.length === 0">
              <div class="text-sm font-medium">Drag images here or click to browse</div>
              <div class="text-xs text-muted">JPG · PNG · BMP · TIFF — select multiple</div>
            </template>
            <template v-else>
              <div class="text-sm font-medium">{{ imageFiles.length }} image{{ imageFiles.length !== 1 ? 's' : '' }} loaded</div>
              <div class="text-xs text-muted font-mono max-h-20 overflow-y-auto w-full space-y-0.5">
                <div v-for="f in imageFiles.slice(0, 6)" :key="f.name" class="truncate">{{ f.name }}</div>
                <div v-if="imageFiles.length > 6">…and {{ imageFiles.length - 6 }} more</div>
              </div>
              <button type="button" class="text-xs text-muted hover:text-danger" @click.stop="imageFiles = []">Clear all</button>
            </template>
          </div>
          <AlertBox v-if="imageFiles.length > 0 && imageFiles.length < 15" tone="warning">
            Only {{ imageFiles.length }} images. At least 15 from different board angles are recommended.
          </AlertBox>
        </div>

        <div v-if="inputMode === 'video'" class="flex flex-col gap-1.5">
          <span class="text-xs font-medium text-muted">Video rotation</span>
          <SegmentedControl mono :model-value="settings.extraction.rotate ?? 'ccw90'" :options="ROTATIONS" :disabled="isRunning"
            @update:model-value="settings.extraction.rotate = $event" />
          <span class="text-xs text-muted">Use <span class="font-mono text-text">ccw90</span> for iPhone portrait MOV (default); <span class="font-mono text-text">none</span> for landscape clips.</span>
        </div>

        <div class="flex flex-col gap-3.5">
          <h2 class="text-base font-semibold">Board parameters</h2>
          <label class="flex flex-col gap-1.5">
            <span class="text-xs font-medium text-muted">ArUco dictionary</span>
            <select v-model="settings.board.dictionary" :disabled="isRunning"
              class="h-9 bg-surface border border-border rounded px-2.5 font-mono text-sm outline-none focus:border-accent disabled:opacity-45">
              <option v-for="d in DICT_OPTIONS" :key="d">{{ d }}</option>
            </select>
          </label>
          <div class="grid grid-cols-2 gap-3.5">
            <FormField label="Columns" type="number" :min="3" :step="1" v-model="settings.board.cols" :disabled="isRunning" />
            <FormField label="Rows" type="number" :min="3" :step="1" v-model="settings.board.rows" :disabled="isRunning" />
            <FormField label="Square" type="number" :min="1" :step="0.5" unit="mm" v-model="settings.board.square_length_mm" :disabled="isRunning" />
            <FormField label="Marker" type="number" :min="1" :step="0.5" unit="mm" v-model="settings.board.marker_length_mm" :disabled="isRunning" />
          </div>
        </div>

        <AlertBox v-if="displayError && !showResult" tone="error">{{ displayError }}</AlertBox>

        <Button variant="primary" size="lg" full-width :disabled="!canRun" :loading="isRunning" @click="handleRun">
          <template v-if="!isRunning" #icon><Play :size="18" :stroke-width="1.5" /></template>
          {{ isRunning ? 'Calibrating…' : 'Run Calibration' }}
        </Button>
      </Card>

      <!-- ═════ Output ═════ -->
      <div class="flex flex-col gap-5 min-w-0">
        <!-- Running -->
        <template v-if="showProgress">
          <Card class="!p-7 flex flex-col gap-6">
            <div class="flex justify-between items-center">
              <h2 class="text-base font-semibold">Calibrating<span v-if="settings.subject_id"> · {{ settings.subject_id }}</span></h2>
              <Button variant="secondary" @click="handleReset">
                <template #icon><RotateCcw :size="14" :stroke-width="1.5" /></template>Cancel &amp; reset
              </Button>
            </div>
            <ProgressSteps :job-status="jobStatus" />
          </Card>
          <div class="grid grid-cols-[1.6fr_1fr_1.2fr] gap-5"><div v-for="i in 3" :key="i" class="h-28 rounded shimmer" /></div>
          <div class="h-[180px] rounded shimmer" />
        </template>

        <!-- Error -->
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

        <!-- Result -->
        <template v-else-if="showResult">
          <div class="rise border rounded px-6 py-5 flex items-center justify-between gap-5 flex-wrap"
            :style="{ borderColor: qualityColor(result!.quality), background: `color-mix(in srgb, ${qualityColor(result!.quality)} 10%, transparent)` }">
            <div class="flex items-center gap-4">
              <QualityBadge :quality="result!.quality" />
              <div>
                <div class="text-[15px] font-medium">{{ QUALITY_MSG[qualityLevel(result!.quality)] }}</div>
                <div class="text-xs text-muted mt-0.5 font-mono">{{ settings.subject_id || 'no subject' }} · {{ result!.quality }}</div>
              </div>
            </div>
            <div class="flex gap-2 flex-wrap">
              <Button @click="openFile('reprojection_error_plot.png')"><template #icon><Download :size="14" :stroke-width="1.5" /></template>Error plot</Button>
              <Button @click="openFile('intrinsic.json')"><template #icon><Download :size="14" :stroke-width="1.5" /></template>JSON</Button>
              <Button @click="openFile('intrinsic.yaml')"><template #icon><Download :size="14" :stroke-width="1.5" /></template>YAML</Button>
              <Button @click="openFile('intrinsic_params.npz')"><template #icon><Download :size="14" :stroke-width="1.5" /></template>NPZ</Button>
            </div>
          </div>

          <MetricsGrid weights="1.6fr 1fr 1.2fr">
            <MetricsCard label="RMS error" :value="result!.rms_px.toFixed(4)" unit="px" :size="48" :color="qualityColor(result!.quality)" :delay="0.08" />
            <MetricsCard label="Frames used" :value="result!.num_frames" subtitle="valid ChArUco detections" :size="32" :delay="0.12" />
            <MetricsCard label="Resolution" :value="`${result!.image_width}×${result!.image_height}`" unit="px" :size="26" :delay="0.16" />
          </MetricsGrid>

          <div class="grid grid-cols-2 gap-5 rise" style="animation-delay:.16s">
            <Card class="!p-[22px]" title="Camera matrix K">
              <div class="border border-border rounded">
                <div v-for="(e, i) in [
                  { k: 'fx', v: result!.camera_matrix.fx }, { k: 'fy', v: result!.camera_matrix.fy },
                  { k: 'cx', v: result!.camera_matrix.cx }, { k: 'cy', v: result!.camera_matrix.cy }]" :key="e.k"
                  :class="['flex justify-between px-3.5 py-[9px] font-mono text-sm', i < 3 && 'border-b border-border']">
                  <span class="text-muted">{{ e.k }}</span><span>{{ e.v.toFixed(2) }}<span class="text-muted"> px</span></span>
                </div>
              </div>
            </Card>
            <Card class="!p-[22px]" title="Distortion vector">
              <div class="border border-border rounded">
                <div v-for="(n, i) in ['k1', 'k2', 'p1', 'p2', 'k3']" :key="n"
                  :class="['flex justify-between px-3.5 py-[9px] font-mono text-sm', i < 4 && 'border-b border-border']">
                  <span class="text-muted">{{ n }}</span><span>{{ result!.distortion_coefficients.vector[i]?.toFixed(6) }}</span>
                </div>
              </div>
            </Card>
          </div>

          <ReprojectionChart v-if="result!.per_frame_rms.length > 0" :data="result!.per_frame_rms" title="Per-frame RMS"
            :warn-at="settings.calibration.rms_accept_threshold" :bad-at="settings.calibration.rms_warn_threshold" />

          <Card v-if="jobId" class="!p-[22px] rise" style="animation-delay:.32s">
            <template #header>
              <h2 class="text-base font-semibold">Reprojection check</h2>
              <div class="flex gap-4 text-xs text-muted">
                <span class="flex items-center gap-1.5"><i class="w-2 h-2 bg-success" />Detected corners</span>
                <span class="flex items-center gap-1.5"><span class="text-danger text-sm leading-none">×</span>Reprojected</span>
              </div>
            </template>
            <div class="border border-border rounded overflow-hidden">
              <div class="hatch">
                <img :src="intrinsicFileUrl(jobId, 'reprojection_error_plot.png')" alt="Reprojection error plot" class="w-full object-contain"
                  @error="($event.target as HTMLImageElement).style.display = 'none'" />
              </div>
              <div class="px-2.5 py-2 text-xs font-mono text-muted">
                Error plot · reprojection error vs. frame. Check images are in <span class="text-text">reprojection_check/</span> of the output folder.
              </div>
            </div>
          </Card>

          <div class="flex gap-6 text-sm">
            <button class="flex items-center gap-1.5 text-accent hover:text-text" @click="openFile('calibration_report.txt')"><FileText :size="16" :stroke-width="1.5" />Open report</button>
            <button class="flex items-center gap-1.5 text-accent hover:text-text" @click="openFile('distortion.json')"><FileText :size="16" :stroke-width="1.5" />distortion.json</button>
          </div>
        </template>

        <!-- Empty: how it works -->
        <Card v-else-if="showEmpty" class="!p-7 flex flex-col gap-6">
          <h2 class="text-base font-semibold">Output · How it works</h2>
          <div class="grid grid-cols-3 border border-border rounded overflow-hidden">
            <div v-for="(h, i) in HOW" :key="h.n" :class="['p-5 flex flex-col gap-2.5', i < 2 && 'border-r border-border']">
              <span class="font-mono text-[13px] text-accent">{{ h.n }}</span>
              <div class="text-xl font-semibold">{{ h.t }}</div>
              <div class="text-sm text-muted leading-normal">{{ h.b }}</div>
            </div>
          </div>
          <div>
            <div class="text-xs font-medium text-muted mb-2.5">Recording tips</div>
            <ul class="flex flex-col gap-2">
              <li v-for="t in TIPS" :key="t" class="flex gap-2.5 items-baseline text-sm">
                <span class="w-1.5 h-1.5 bg-accent flex-none -translate-y-px" />{{ t }}
              </li>
            </ul>
          </div>
        </Card>
      </div>
    </div>
  </div>
</template>
