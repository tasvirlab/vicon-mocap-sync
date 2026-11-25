<script setup lang="ts">
import { ref, watch, computed } from 'vue'
import { Check, ExternalLink, RotateCcw } from '@lucide/vue'
import Button        from '@renderer/calib/components/ui/Button.vue'
import FormField    from '@renderer/calib/components/ui/FormField.vue'
import ToggleSwitch from '@renderer/calib/components/ui/ToggleSwitch.vue'
import { useSettings } from '@renderer/calib/composables/useSettings'

// Bound straight to the shared store: every edit persists automatically.
const { settings: cfg, resetToDefaults } = useSettings()

const DICT_OPTIONS = [
  'DICT_4X4_50', 'DICT_4X4_100', 'DICT_5X5_50',
  'DICT_6X6_250', 'DICT_7X7_250', 'DICT_ARUCO_ORIGINAL',
]

const saving = ref(false)
let timer: ReturnType<typeof setTimeout> | null = null
watch(cfg, () => {
  saving.value = true
  if (timer) clearTimeout(timer)
  timer = setTimeout(() => { saving.value = false }, 600)
}, { deep: true })
const savedLabel = computed(() => saving.value ? 'Saving…' : 'All changes saved')

function openDocs() { window.open('http://127.0.0.1:8765/docs', '_blank') }
</script>

<template>
  <div class="animate-fade-in">
    <div class="flex items-end justify-between gap-6 mb-6">
      <div>
        <h1 class="text-[32px] leading-[1.1] font-bold tracking-tight">Settings</h1>
        <p class="text-[15px] text-muted mt-1.5">Changes are saved automatically.</p>
      </div>
      <div class="flex items-center gap-3.5">
        <span class="flex items-center gap-2 text-[13px] text-muted" role="status">
          <Check :size="14" :stroke-width="1.5" class="text-success" />{{ savedLabel }}
        </span>
        <Button variant="secondary" @click="resetToDefaults">
          <template #icon><RotateCcw :size="14" :stroke-width="1.5" /></template>Reset to defaults
        </Button>
      </div>
    </div>

    <div class="flex flex-col gap-5">
      <!-- ChArUco board -->
      <section class="border border-border rounded p-6 grid grid-cols-[220px_minmax(0,1fr)] gap-6">
        <div>
          <h2 class="text-base font-semibold">ChArUco board</h2>
          <p class="text-[13px] text-muted mt-1.5 leading-normal">Must match the printed board.</p>
        </div>
        <div class="grid grid-cols-2 gap-4">
          <label class="flex flex-col gap-1.5">
            <span class="text-xs font-medium text-muted">ArUco dictionary</span>
            <select v-model="cfg.board.dictionary" class="h-9 bg-surface border border-border rounded px-2.5 font-mono text-sm outline-none focus:border-accent">
              <option v-for="d in DICT_OPTIONS" :key="d">{{ d }}</option>
            </select>
            <span class="text-xs text-muted">Marker dictionary of the print.</span>
          </label>
          <span />
          <FormField label="Columns" type="number" :min="3" :step="1" v-model="cfg.board.cols" hint="Chessboard squares in X." />
          <FormField label="Rows" type="number" :min="3" :step="1" v-model="cfg.board.rows" hint="Chessboard squares in Y." />
          <FormField label="Square size" type="number" :min="1" :step="0.5" unit="mm" v-model="cfg.board.square_length_mm" hint="Side length of each square." />
          <FormField label="Marker size" type="number" :min="1" :step="0.5" unit="mm" v-model="cfg.board.marker_length_mm" hint="ArUco marker side (≤ 0.8 × square)." />
        </div>
      </section>

      <!-- Frame extraction -->
      <section class="border border-border rounded p-6 grid grid-cols-[220px_minmax(0,1fr)] gap-6">
        <div>
          <h2 class="text-base font-semibold">Frame extraction</h2>
          <p class="text-[13px] text-muted mt-1.5 leading-normal">How frames are picked from a video.</p>
        </div>
        <div class="grid grid-cols-2 gap-4">
          <FormField label="Sample interval" type="number" :min="1" unit="frames" v-model="cfg.extraction.sample_interval_frames" hint="Process every Nth frame." />
          <FormField label="Blur threshold" type="number" :min="0" :step="5" v-model="cfg.extraction.blur_threshold" hint="Minimum Laplacian variance (higher = stricter)." />
          <FormField label="Target frames" type="number" :min="15" v-model="cfg.extraction.target_accepted_frames" hint="Desired number of accepted frames." />
        </div>
      </section>

      <!-- Quality thresholds -->
      <section class="border border-border rounded p-6 grid grid-cols-[220px_minmax(0,1fr)] gap-6">
        <div>
          <h2 class="text-base font-semibold">Quality thresholds</h2>
          <p class="text-[13px] text-muted mt-1.5 leading-normal">RMS limits for each rating.</p>
        </div>
        <div class="grid grid-cols-2 gap-4">
          <FormField label="Excellent below" type="number" :min="0.1" :step="0.05" unit="px" v-model="cfg.calibration.rms_accept_threshold" hint="RMS below this is rated Excellent." />
          <FormField label="Good below" type="number" :min="0.1" :step="0.1" unit="px" v-model="cfg.calibration.rms_warn_threshold" hint="RMS below this is rated Good." />
        </div>
      </section>

      <!-- PnP solver -->
      <section class="border border-border rounded p-6 grid grid-cols-[220px_minmax(0,1fr)] gap-6">
        <div>
          <h2 class="text-base font-semibold">PnP solver</h2>
          <p class="text-[13px] text-muted mt-1.5 leading-normal">Used by Extrinsic calibration.</p>
        </div>
        <div class="grid grid-cols-2 gap-4">
          <div class="flex flex-col gap-1.5">
            <span class="text-xs font-medium text-muted">Use RANSAC</span>
            <div class="h-9 flex items-center"><ToggleSwitch v-model="cfg.solver.use_ransac" :label="cfg.solver.use_ransac ? 'On' : 'Off'" /></div>
            <span class="text-xs text-muted">Run solvePnPRansac before deterministic methods.</span>
          </div>
          <FormField label="RANSAC threshold" type="number" :min="1" :step="0.5" unit="px" v-model="cfg.solver.ransac_reprojection_threshold_px" hint="Reprojection error limit for inliers." />
          <div class="flex flex-col gap-1.5">
            <span class="text-xs font-medium text-muted">LM refinement</span>
            <div class="h-9 flex items-center"><ToggleSwitch v-model="cfg.solver.refine_lm" :label="cfg.solver.refine_lm ? 'On' : 'Off'" /></div>
            <span class="text-xs text-muted">Refine the PnP solution with Levenberg–Marquardt.</span>
          </div>
        </div>
      </section>

      <!-- Backend -->
      <section class="border border-border rounded p-6 grid grid-cols-[220px_minmax(0,1fr)] gap-6">
        <div>
          <h2 class="text-base font-semibold">Backend</h2>
          <p class="text-[13px] text-muted mt-1.5 leading-normal">Local calibration service.</p>
        </div>
        <div class="grid grid-cols-2 gap-4">
          <div class="flex flex-col gap-1.5">
            <span class="text-xs font-medium text-muted">API endpoint</span>
            <div class="flex items-center gap-2">
              <code class="flex-1 h-9 flex items-center bg-surface border border-border rounded px-2.5 font-mono text-sm">http://127.0.0.1:8765</code>
              <Button variant="secondary" size="md" title="Open API docs" @click="openDocs"><ExternalLink :size="14" :stroke-width="1.5" /></Button>
            </div>
          </div>
          <div class="flex flex-col gap-1.5">
            <span class="text-xs font-medium text-muted">Conda environment</span>
            <code class="h-9 flex items-center bg-surface border border-border rounded px-2.5 font-mono text-sm">calib_app</code>
          </div>
        </div>
      </section>
    </div>
  </div>
</template>
