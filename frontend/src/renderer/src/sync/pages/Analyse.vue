<script setup lang="ts">
/**
 * Analyse — Channel 2 (posture duration) + Channel 3 (trunk dynamics) figures.
 *
 * Read-only inspection surface: upload one CSV, see the two figures and the
 * numbers behind them. Nothing is written to outputs/ and no trial is modified,
 * so this can be used freely during QC or when preparing a figure panel.
 */
import { ref, computed } from 'vue'
import {
  BarChart3, Play, RotateCcw, AlertTriangle, Download, Info, Timer, Activity, Gauge,
} from '@lucide/vue'
import Button from '@renderer/sync/components/ui/Button.vue'
import PageHeader from '@renderer/sync/components/ui/PageHeader.vue'
import DropZone from '@renderer/sync/components/ui/DropZone.vue'
import { analyseFigures } from '@renderer/sync/lib/api'
import type { AnalyseResult } from '@renderer/sync/lib/types'

const csvFile  = ref<File | null>(null)
const jsonFile = ref<File | null>(null)
const taskId   = ref('')
const fpsOverride = ref<number | null>(null)
const enterS   = ref(0.5)
const exitS    = ref(0.5)
// Channel 1 task metadata — entered per trial, never inferred from pose.
const load       = ref(0)
const coupling   = ref(0)
const activity   = ref(1)
const unilateral = ref(false)

const running = ref(false)
const error   = ref<string | null>(null)
const result  = ref<AnalyseResult | null>(null)

const SEGMENT_LABELS: Record<string, string> = {
  trunk: 'Trunk', neck: 'Neck', upper_arm_r: 'Upper arm (R)', upper_arm_l: 'Upper arm (L)',
}

const KIND_LABELS: Record<string, string> = {
  angles:   'angles.csv — signed angle channels (channels recomputed here)',
  joints:   'synced_60hz.csv — 3D joints (angles computed, then channels)',
  dynamics: 'trunk_dynamics.csv — pre-computed Channel 3 only',
}

const segments = computed(() =>
  Object.entries(result.value?.posture_duration?.segments ?? {}))

const predictors = computed(() => {
  const p = result.value?.trunk_dynamics?.marras_predictors ?? {}
  const LABELS: Record<string, string> = {
    peak_trunk_sagittal_deg: 'Peak sagittal flexion (°)',
    peak_lateral_vel_deg_s:  'Peak lateral velocity (°/s)',
    mean_lateral_vel_deg_s:  'Mean lateral velocity (°/s)',
    peak_twist_vel_deg_s:    'Peak twisting velocity (°/s)',
    mean_twist_vel_deg_s:    'Mean twisting velocity (°/s)',
  }
  return Object.entries(LABELS)
    .map(([k, label]) => ({ label, value: p[k] }))
    .filter(e => e.value != null)
})

async function run() {
  if (!csvFile.value) return
  running.value = true
  error.value = null
  result.value = null
  try {
    result.value = await analyseFigures(csvFile.value, {
      dynamicsJson: jsonFile.value,
      fps: fpsOverride.value,
      enterS: enterS.value,
      exitS: exitS.value,
      taskId: taskId.value,
      load: load.value,
      coupling: coupling.value,
      activity: activity.value,
      unilateral: unilateral.value,
    })
  } catch (e: any) {
    error.value = e?.response?.data?.detail ?? e?.message ?? 'Analysis failed'
  } finally {
    running.value = false
  }
}

/** Format a numeric field defensively.
 *
 *  A missing field must render as "—", never throw. Calling `.toFixed()` on a
 *  renamed/absent field raises inside the template, which fails the whole
 *  component render and leaves the page BLANK — one backend field rename
 *  (total_above_threshold_s -> total_high_risk_s) took down the entire Analyse
 *  page this way. A results table is not worth losing the page over. */
function num(v: unknown, digits = 2): string {
  return typeof v === 'number' && Number.isFinite(v) ? v.toFixed(digits) : '—'
}

function reset() {
  csvFile.value = null; jsonFile.value = null; result.value = null
  error.value = null; taskId.value = ''; fpsOverride.value = null
}

function download(key: 'reba' | 'posture_duration' | 'trunk_dynamics') {
  const b64 = result.value?.figures?.[key]
  if (!b64) return
  const a = document.createElement('a')
  a.href = `data:image/png;base64,${b64}`
  a.download = `${result.value?.task_id || 'trial'}_${key}.png`
  a.click()
}
</script>

<template>
  <div class="p-6 space-y-4 animate-fade-in">
    <PageHeader
      title="Analyse"
      subtitle="Visualise all three ergonomic channels — REBA severity, posture duration, trunk dynamics"
      :icon="BarChart3"
      color="#8B5CF6"
    >
      <template #actions>
        <Button v-if="result" variant="ghost" size="sm" @click="reset">
          <template #icon><RotateCcw :size="13" /></template>
          Clear
        </Button>
      </template>
    </PageHeader>

    <div class="grid grid-cols-5 gap-4">

      <!-- ══ Left: input ══ -->
      <div class="col-span-2 space-y-4">
        <div class="rounded-2xl border border-border p-4 space-y-3">
          <p class="text-xs font-semibold text-text">Input file</p>

          <DropZone
            label="Drop a trial CSV"
            description="<strong>reba/angles.csv</strong> · <strong>synced_60hz.csv</strong> · <strong>reba/trunk_dynamics.csv</strong>"
            accept=".csv"
            :file="csvFile"
            :disabled="running"
            @update:file="csvFile = $event"
          />

          <DropZone
            label="trunk_dynamics.json (optional)"
            description="Only needed when uploading trunk_dynamics.csv — supplies the Marras panel"
            accept=".json"
            :file="jsonFile"
            :disabled="running"
            @update:file="jsonFile = $event"
          />

          <div class="grid grid-cols-2 gap-2">
            <label class="block">
              <span class="text-xs text-muted">Trial label (figure title)</span>
              <input v-model="taskId" placeholder="SUB01_lifting_T01"
                class="w-full mt-1 px-2 py-1.5 text-xs rounded-lg bg-surface border border-border text-text focus:border-accent/50 outline-none" />
            </label>
            <label class="block">
              <span class="text-xs text-muted">fps override (blank = from time_s)</span>
              <input v-model.number="fpsOverride" type="number" step="0.001" placeholder="auto"
                class="w-full mt-1 px-2 py-1.5 text-xs rounded-lg bg-surface border border-border text-text focus:border-accent/50 outline-none" />
            </label>
            <label class="block">
              <span class="text-xs text-muted">Hysteresis enter (s)</span>
              <input v-model.number="enterS" type="number" step="0.1" min="0.1"
                class="w-full mt-1 px-2 py-1.5 text-xs rounded-lg bg-surface border border-border text-text focus:border-accent/50 outline-none" />
            </label>
            <label class="block">
              <span class="text-xs text-muted">Hysteresis exit (s)</span>
              <input v-model.number="exitS" type="number" step="0.1" min="0.1"
                class="w-full mt-1 px-2 py-1.5 text-xs rounded-lg bg-surface border border-border text-text focus:border-accent/50 outline-none" />
            </label>
          </div>

          <div class="pt-2 border-t border-border space-y-2">
            <p class="text-xs font-semibold text-text">Channel 1 — REBA task metadata</p>
            <p class="text-xs text-muted">Entered per trial, never inferred from pose. Leaving load and coupling at 0 scores posture only.</p>
            <div class="grid grid-cols-2 gap-2">
              <label class="block">
                <span class="text-xs text-muted">Load / force (0-3)</span>
                <input v-model.number="load" type="number" min="0" max="3" step="1"
                  class="w-full mt-1 px-2 py-1.5 text-xs rounded-lg bg-surface border border-border text-text focus:border-accent/50 outline-none" />
              </label>
              <label class="block">
                <span class="text-xs text-muted">Coupling (0-3)</span>
                <input v-model.number="coupling" type="number" min="0" max="3" step="1"
                  class="w-full mt-1 px-2 py-1.5 text-xs rounded-lg bg-surface border border-border text-text focus:border-accent/50 outline-none" />
              </label>
              <label class="block">
                <span class="text-xs text-muted">Activity (0-3)</span>
                <input v-model.number="activity" type="number" min="0" max="3" step="1"
                  class="w-full mt-1 px-2 py-1.5 text-xs rounded-lg bg-surface border border-border text-text focus:border-accent/50 outline-none" />
              </label>
              <label class="flex items-center gap-2 mt-4">
                <input v-model="unilateral" type="checkbox"
                  class="rounded border-border bg-surface accent-accent" />
                <span class="text-xs text-muted">Unilateral leg load</span>
              </label>
            </div>
          </div>

          <Button variant="primary" :full-width="true" :disabled="!csvFile" :loading="running" @click="run">
            <template v-if="!running" #icon><Play :size="15" /></template>
            {{ running ? 'Rendering figures…' : 'Render figures' }}
          </Button>

          <p v-if="error" class="text-xs text-danger flex items-start gap-1.5">
            <AlertTriangle :size="12" class="flex-shrink-0 mt-0.5" /> {{ error }}
          </p>
        </div>

        <!-- What this section shows -->
        <div class="bg-surface rounded-2xl border border-border p-4 text-xs text-muted space-y-1.5">
          <p class="font-semibold text-text text-xs mb-1 flex items-center gap-1.5">
            <Info :size="12" /> What the three channels answer
          </p>
          <p>· <strong class="text-text">Channel 1 — severity:</strong> how awkward the posture is, as a frame-wise REBA score and action level. Load, coupling and activity are <strong class="text-text">entered task metadata</strong>, never inferred from pose.</p>
          <p>· <strong class="text-text">Channel 2 — duration:</strong> how long and how often a posture stays above its angle threshold. Hysteresis gating turns noisy frame-wise state into stable bouts with back-dated onset and offset. Bouts include a movement's rising and falling phases, so they are <strong class="text-text">not</strong> static-posture durations.</p>
          <p>· <strong class="text-text">Channel 3 — dynamics:</strong> how rapidly or abruptly the trunk moves, in all three anatomical planes, as signed angle → angular velocity → angular acceleration.</p>
          <p>· The channels stay <strong class="text-text">separate</strong> — they are not fused into a composite risk score.</p>
          <p>· Read-only: nothing is written to outputs/ and no trial is modified.</p>
        </div>

        <!-- Result meta -->
        <div v-if="result" class="rounded-2xl border border-border p-4 space-y-2">
          <p class="text-xs font-semibold text-text">Input recognised</p>
          <p class="text-xs text-muted">{{ KIND_LABELS[result.input_kind] ?? result.input_kind }}</p>
          <div class="grid grid-cols-2 gap-2 pt-1 border-t border-border">
            <div>
              <p class="text-sm font-bold font-mono text-text">{{ result.n_frames }}</p>
              <p class="text-xs text-muted">frames</p>
            </div>
            <div>
              <p class="text-sm font-bold font-mono text-text">{{ result.fps }}</p>
              <p class="text-xs text-muted">Hz</p>
            </div>
          </div>
          <p v-for="(w, i) in result.warnings" :key="i"
            class="text-xs text-warning flex items-start gap-1.5">
            <AlertTriangle :size="11" class="flex-shrink-0 mt-0.5" /> {{ w }}
          </p>
        </div>
      </div>

      <!-- ══ Right: figures ══ -->
      <div class="col-span-3 space-y-4">
        <div v-if="!result" class="rounded-2xl border border-border p-10 flex flex-col items-center justify-center text-muted space-y-2">
          <BarChart3 :size="34" class="opacity-20" />
          <p class="text-xs">Upload a trial CSV to render all three channel figures.</p>
        </div>

        <template v-else>
          <!-- Channel 1 -->
          <div v-if="result.figures.reba" class="rounded-2xl border border-border p-4 space-y-3">
            <div class="flex items-center justify-between">
              <div class="flex items-center gap-2">
                <Gauge :size="14" class="text-accent" />
                <span class="text-xs font-semibold text-text">Channel 1 — REBA posture severity</span>
              </div>
              <Button variant="ghost" size="sm" @click="download('reba')">
                <template #icon><Download :size="12" /></template>
                PNG
              </Button>
            </div>
            <img :src="`data:image/png;base64,${result.figures.reba}`"
              alt="REBA severity figure" class="w-full rounded-lg bg-white" />
            <div v-if="result.reba" class="grid grid-cols-5 gap-2">
              <div v-for="m in [
                    { k: 'mean',   label: 'mean REBA' },
                    { k: 'median', label: 'median' },
                    { k: 'max',    label: 'peak' },
                    { k: 'n_valid_frames',   label: 'valid frames' },
                    { k: 'n_dropped_frames', label: 'invalid' },
                  ]" :key="m.k"
                class="rounded-lg border border-border bg-surface p-2 text-center">
                <p class="text-sm font-bold font-mono text-text">
                  {{ typeof result.reba.summary[m.k] === 'number'
                      ? Number(result.reba.summary[m.k]).toFixed(m.k === 'mean' || m.k === 'median' ? 1 : 0)
                      : result.reba.summary[m.k] }}
                </p>
                <p class="text-xs text-muted leading-tight mt-0.5">{{ m.label }}</p>
              </div>
            </div>
            <p v-if="result.reba" class="text-xs text-muted">
              Scored with load {{ result.reba.params.load }}, coupling {{ result.reba.params.coupling }},
              activity {{ result.reba.params.activity }}<span v-if="result.reba.params.unilateral">, unilateral</span>
              — entered task metadata. Vicon-derived reference, not expert-validated REBA.
            </p>
          </div>

          <!-- Channel 2 -->
          <div v-if="result.figures.posture_duration" class="rounded-2xl border border-border p-4 space-y-3">
            <div class="flex items-center justify-between">
              <div class="flex items-center gap-2">
                <Timer :size="14" class="text-accent" />
                <span class="text-xs font-semibold text-text">Channel 2 — within-task posture duration</span>
              </div>
              <Button variant="ghost" size="sm" @click="download('posture_duration')">
                <template #icon><Download :size="12" /></template>
                PNG
              </Button>
            </div>
            <img :src="`data:image/png;base64,${result.figures.posture_duration}`"
              alt="Posture duration figure" class="w-full rounded-lg bg-white" />

            <div v-if="segments.length" class="overflow-x-auto">
              <table class="w-full text-xs">
                <thead class="text-muted border-b border-border">
                  <tr>
                    <th class="text-left font-medium py-1.5">Segment</th>
                    <th class="text-right font-medium">Bouts</th>
                    <th class="text-right font-medium">Total high-risk (s)</th>
                    <th class="text-right font-medium">Longest (s)</th>
                    <th class="text-right font-medium">% of task</th>
                    <th class="text-right font-medium">Entry (°)</th>
                    <th class="text-right font-medium">Static holds</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="[key, s] in segments" :key="key" class="border-b border-border/50">
                    <td class="py-1.5 text-text">{{ SEGMENT_LABELS[key] ?? key }}</td>
                    <td class="text-right font-mono text-text">{{ num(s.n_bouts, 0) }}</td>
                    <td class="text-right font-mono text-text">{{ num(s.total_high_risk_s) }}</td>
                    <td class="text-right font-mono text-text">{{ num(s.longest_bout_s) }}</td>
                    <td class="text-right font-mono text-text">{{ num(s.percent_task_time, 0) }}%</td>
                    <td class="text-right font-mono text-muted">{{ num(s.entry_threshold_deg, 0) }}</td>
                    <td class="text-right font-mono"
                      :class="s.static_evaluated ? 'text-text' : 'text-muted'">
                      {{ s.static_evaluated ? num(s.n_static_holds, 0) : 'not eval.' }}
                    </td>
                  </tr>
                </tbody>
              </table>
              <p class="text-xs text-muted mt-2 flex items-start gap-1.5">
                <AlertTriangle :size="11" class="flex-shrink-0 mt-0.5" />
                <span>
                  Bouts include a movement's rising and falling phases, so they are
                  <strong>not</strong> static-posture durations. Static classification is decided
                  <strong>per bout</strong> (duration &gt; 4 s, low angular velocity, low
                  variability) and
                  <template v-if="result.posture_duration?.static_criteria?.evaluated">
                    is active for this trial.
                  </template>
                  <template v-else>
                    is <strong>not evaluated</strong> — the low-motion thresholds have not been
                    derived and frozen, so no bout is claimed to be a static hold.
                  </template>
                  ISO 11226 holding-time comparison applies only to a static bout under minimal
                  external force; its anchors remain unverified against the standard.
                </span>
              </p>
            </div>
          </div>

          <!-- Channel 3 -->
          <div v-if="result.figures.trunk_dynamics" class="rounded-2xl border border-border p-4 space-y-3">
            <div class="flex items-center justify-between">
              <div class="flex items-center gap-2">
                <Activity :size="14" class="text-accent" />
                <span class="text-xs font-semibold text-text">Channel 3 — trunk-motion dynamics</span>
              </div>
              <Button variant="ghost" size="sm" @click="download('trunk_dynamics')">
                <template #icon><Download :size="12" /></template>
                PNG
              </Button>
            </div>
            <img :src="`data:image/png;base64,${result.figures.trunk_dynamics}`"
              alt="Trunk dynamics figure" class="w-full rounded-lg bg-white" />

            <div v-if="predictors.length" class="grid grid-cols-5 gap-2">
              <div v-for="p in predictors" :key="p.label"
                class="rounded-lg border border-border bg-surface p-2 text-center">
                <p class="text-sm font-bold font-mono text-text">{{ Number(p.value).toFixed(1) }}</p>
                <p class="text-xs text-muted leading-tight mt-0.5">{{ p.label }}</p>
              </div>
            </div>

            <p v-if="result.trunk_dynamics" class="text-xs"
              :class="result.trunk_dynamics.marras_probability.applicable ? 'text-muted' : 'text-warning'">
              <template v-if="result.trunk_dynamics.marras_probability.applicable">
                Marras high-risk-<strong>group</strong> probability:
                {{ result.trunk_dynamics.marras_probability.probability?.toFixed(3) }}
                — a group-level probability, never an individual injury probability.
              </template>
              <template v-else>
                Marras probability not applicable — {{ result.trunk_dynamics.marras_probability.reason }}
              </template>
            </p>
          </div>
        </template>
      </div>
    </div>
  </div>
</template>
