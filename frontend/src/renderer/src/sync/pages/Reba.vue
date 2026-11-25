<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import {
  Activity, Play, RotateCcw, AlertCircle, Info,
  CheckCircle, XCircle, FolderOpen, ChevronDown,
} from '@lucide/vue'
import Button     from '@renderer/sync/components/ui/Button.vue'
import PageHeader  from '@renderer/sync/components/ui/PageHeader.vue'
import StatusBadge from '@renderer/sync/components/ui/StatusBadge.vue'
import ProgressSteps from '@renderer/sync/components/ui/ProgressSteps.vue'
import { useSyncJob } from '@renderer/sync/composables/useSyncJob'
import {
  listProjects, startReba, getRebaResult, getSavedReba, openProjectFolder,
  runJobToCompletion,
} from '@renderer/sync/lib/api'
import type { ProjectEntry, RebaResult, StatSummary } from '@renderer/sync/lib/types'

// ── Param option lists ──────────────────────────────────────────────────────────
const LOAD_OPTIONS     = [{ v: 0, label: '0 — < 5 kg (light)' }, { v: 1, label: '1 — 5–10 kg' }, { v: 2, label: '2 — > 10 kg' }]
const COUPLING_OPTIONS = [{ v: 0, label: '0 — Good grip' }, { v: 1, label: '1 — Fair grip' }, { v: 2, label: '2 — Poor / none' }]
const ACTIVITY_OPTIONS = [{ v: 0, label: '0 — Static posture' }, { v: 1, label: '1 — Repeated / dynamic' }]

const RISK_COLOR: Record<string, string> = {
  negligible: 'text-success bg-success/10 border-success/30',
  low:        'text-yellow-400 bg-yellow-400/10 border-yellow-400/30',
  medium:     'text-orange-400 bg-orange-400/10 border-orange-400/30',
  high:       'text-danger bg-danger/10 border-danger/30',
  very_high:  'text-purple-400 bg-purple-400/10 border-purple-400/30',
}

const ANGLE_LABELS: Record<string, string> = {
  trunk_total_deg:      'Trunk — total inclination',
  trunk_sagittal_deg:   'Trunk — sagittal (fwd+/bwd−)',
  trunk_lateral_deg:    'Trunk — lateral bend',
  trunk_twist_deg:      'Trunk — twist (sho vs hip)',
  neck_flex_deg:        'Neck — flexion from trunk',
  neck_lateral_deg:     'Neck — lateral bend',
  upper_arm_r_deg:      'Upper arm R — elevation',
  upper_arm_r_abd_deg:  'Upper arm R — abduction',
  lower_arm_r_deg:      'Lower arm R — elbow flexion',
  wrist_r_deg:          'Wrist R — deviation',
  upper_arm_l_deg:      'Upper arm L — elevation',
  upper_arm_l_abd_deg:  'Upper arm L — abduction',
  lower_arm_l_deg:      'Lower arm L — elbow flexion',
  wrist_l_deg:          'Wrist L — deviation',
  knee_r_deg:           'Knee R — flexion',
  knee_l_deg:           'Knee L — flexion',
  knee_max_deg:         'Knee — max (both sides)',
  hip_r_deg:            'Hip R — angle',
  hip_l_deg:            'Hip L — angle',
}

const COMPONENT_LABELS: Record<string, string> = {
  trunk_score:    'Trunk',
  neck_score:     'Neck',
  legs_score:     'Legs',
  upper_arm_adj:  'Upper Arm (adjusted)',
  lower_arm_score:'Lower Arm',
  wrist_adj:      'Wrist (adjusted)',
  score_a:        'Score A (Group A + activity)',
  score_b:        'Score B (Group B + coupling)',
  reba_total:     'REBA Total',
}

const COMPONENT_MAX: Record<string, number> = {
  trunk_score: 5, neck_score: 3, legs_score: 4,
  upper_arm_adj: 6, lower_arm_score: 2, wrist_adj: 3,
  score_a: 12, score_b: 12, reba_total: 15,
}

// ── State ────────────────────────────────────────────────────────────────────────
const { jobId, jobStatus, isRunning, error, startJob, reset: resetJob } = useSyncJob()

const tasks        = ref<ProjectEntry[]>([])
const selected     = ref<Set<string>>(new Set())   // batch-run checkboxes
const viewTask     = ref('')                        // dropdown: view one saved result
const loadingTasks = ref(false)

const paramLoad      = ref(0)
const paramCoupling  = ref(0)
const paramActivity  = ref(1)
const paramUnilateral = ref(false)

const result     = ref<RebaResult | null>(null)
const fetchError = ref<string | null>(null)

// ── Batch run state (2+ trials selected) ──────────────────────────────────────────
const batchRunning  = ref(false)
const batchResults  = ref<{ task_name: string; status: 'ok' | 'error'; error?: string }[]>([])
const batchProgress = ref({ current: 0, total: 0 })

// ── Load task list ────────────────────────────────────────────────────────────────
async function loadTasks() {
  loadingTasks.value = true
  try {
    const all = await listProjects()
    // Only show tasks that have video_synced.csv (has_synced_csv flag)
    tasks.value = all.filter(t => t.has_synced_csv)
    if (!viewTask.value && tasks.value.length > 0) {
      viewTask.value = tasks.value[0].id
    }
  } catch { /* offline */ }
  finally { loadingTasks.value = false }
}

function toggle(name: string) {
  const s = new Set(selected.value)
  s.has(name) ? s.delete(name) : s.add(name)
  selected.value = s
}
function selectAll() { selected.value = new Set(tasks.value.map(t => t.id)) }
function selectNone() { selected.value = new Set() }

// ── View saved result — independent of the batch-run checkboxes ──────────────────
watch(viewTask, async (name) => {
  result.value     = null
  fetchError.value = null
  if (!name) return
  try {
    result.value = await getSavedReba(name)
  } catch { /* no saved result yet */ }
})

onMounted(loadTasks)

// ── Fetch result after a single-trial job completes ───────────────────────────────
watch(() => jobStatus.value?.status, (status) => {
  if (status === 'completed' && jobId.value) {
    fetchError.value = null
    getRebaResult(jobId.value)
      .then(r => { result.value = r; viewTask.value = r.task_name; loadTasks() })
      .catch(e => { fetchError.value = `Failed to load result: ${e?.message}` })
  }
})

// ── Run — one trial shows full detail directly; 2+ run sequentially as a batch ────
async function handleRun() {
  const names = [...selected.value]
  if (names.length === 0) return
  result.value       = null
  fetchError.value   = null
  batchResults.value = []

  if (names.length === 1) {
    try {
      const { job_id } = await startReba(
        names[0], paramLoad.value, paramCoupling.value, paramActivity.value, paramUnilateral.value,
      )
      startJob(job_id, 'reba')
    } catch (err: unknown) {
      fetchError.value = `Failed to start: ${err instanceof Error ? err.message : String(err)}`
    }
    return
  }

  batchRunning.value = true
  batchProgress.value = { current: 0, total: names.length }
  for (const name of names) {
    batchProgress.value = { current: batchProgress.value.current + 1, total: names.length }
    try {
      const { job_id } = await startReba(
        name, paramLoad.value, paramCoupling.value, paramActivity.value, paramUnilateral.value,
      )
      await runJobToCompletion('reba', job_id)
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

function handleReset() {
  resetJob()
  result.value = fetchError.value = null
  batchResults.value = []
}

// ── Computed helpers ─────────────────────────────────────────────────────────────
const canRun          = computed(() => selected.value.size > 0 && !isRunning.value && !batchRunning.value)
const displayError    = computed(() => error.value || fetchError.value)
const showProgress    = computed(() => (isRunning.value && !!jobStatus.value) || batchRunning.value)
const showBatchResult = computed(() => !batchRunning.value && batchResults.value.length > 0)
const showResult      = computed(() => !isRunning.value && !batchRunning.value && !showBatchResult.value && !!result.value)
const showEmpty       = computed(() => !isRunning.value && !batchRunning.value && !showBatchResult.value && !result.value)

function riskClass(risk: string) {
  return RISK_COLOR[risk] ?? 'text-muted bg-surface border-border'
}

function riskLabel(key: string) {
  return key.replace('_', ' ').replace(/^\w/, c => c.toUpperCase())
}

function pct(count: number) {
  return result.value ? ((count / result.value.n_frames) * 100).toFixed(1) : '0'
}

function statCell(s: StatSummary, key: 'mean' | 'min' | 'max' | 'median' | 'std', decimals = 1) {
  const v = s[key]
  return typeof v === 'number' ? v.toFixed(decimals) : String(v)
}

// Ordered angle keys for table display
const angleKeys = computed(() =>
  result.value ? Object.keys(result.value.angle_summary).filter(k => k in ANGLE_LABELS) : []
)

const componentKeys = computed(() =>
  result.value ? Object.keys(COMPONENT_LABELS).filter(k => k in (result.value!.component_stats)) : []
)
</script>

<template>
  <div class="p-6 animate-fade-in flex flex-col h-screen gap-5">

    <!-- Header -->
    <PageHeader title="REBA Score" :icon="Activity" color="#F59E0B" :step="4"
      subtitle="Compute ergonomic risk scores (REBA 1–15) from the synced 3D Vicon joints — the ground truth for validation.">
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

        <!-- Trial selection (batch) -->
        <div class="rounded-xl border border-border p-4 space-y-3">
          <div class="flex items-center justify-between">
            <p class="text-xs font-semibold text-muted ">Select Trials</p>
            <button class="text-xs text-accent hover:text-accent/80"
              :disabled="loadingTasks" @click="loadTasks">
              {{ loadingTasks ? 'Loading…' : 'Refresh' }}
            </button>
          </div>

          <div v-if="tasks.length === 0 && !loadingTasks"
            class="flex items-start gap-2 bg-surface rounded-lg p-2.5">
            <Info :size="12" class="text-muted mt-0.5 flex-shrink-0" />
            <p class="text-xs text-muted">No tasks with video_synced.csv found. Run Overlay first.</p>
          </div>

          <template v-else>
            <div class="flex gap-2 text-xs">
              <button class="text-accent hover:text-accent/80" @click="selectAll">Select all</button>
              <span class="text-muted">·</span>
              <button class="text-accent hover:text-accent/80" @click="selectNone">Clear</button>
              <span class="text-muted ml-auto">{{ selected.size }} / {{ tasks.length }} selected</span>
            </div>
            <div class="space-y-1.5 max-h-64 overflow-y-auto">
              <label v-for="task in tasks" :key="task.id"
                class="flex items-center gap-2.5 p-2.5 rounded-lg border cursor-pointer transition-colors"
                :class="selected.has(task.id)
                  ? 'border-accent bg-tint'
                  : 'border-border hover:border-muted'">
                <input type="checkbox" class="w-3.5 h-3.5 accent-accent"
                  :checked="selected.has(task.id)" @change="toggle(task.id)" />
                <span class="text-xs font-medium text-text truncate flex-1">{{ task.name }}</span>
              </label>
            </div>
          </template>
        </div>

        <!-- REBA Parameters -->
        <div class="rounded-xl border border-border p-4 space-y-3">
          <p class="text-xs font-semibold text-muted ">REBA Parameters</p>
          <p class="text-xs text-muted">These add to the base posture scores. Load and coupling are task-specific.</p>

          <!-- Load -->
          <div class="space-y-1">
            <label class="text-xs font-medium text-muted">Load / Force</label>
            <div class="relative">
              <select v-model="paramLoad" :disabled="isRunning"
                class="w-full bg-surface border border-border rounded-lg px-2 py-1.5 text-xs text-text appearance-none focus:outline-none focus:border-accent disabled:opacity-50">
                <option v-for="o in LOAD_OPTIONS" :key="o.v" :value="o.v">{{ o.label }}</option>
              </select>
              <ChevronDown :size="12" class="absolute right-2 top-1/2 -translate-y-1/2 text-muted pointer-events-none" />
            </div>
          </div>

          <!-- Coupling -->
          <div class="space-y-1">
            <label class="text-xs font-medium text-muted">Coupling (grip quality)</label>
            <div class="relative">
              <select v-model="paramCoupling" :disabled="isRunning"
                class="w-full bg-surface border border-border rounded-lg px-2 py-1.5 text-xs text-text appearance-none focus:outline-none focus:border-accent disabled:opacity-50">
                <option v-for="o in COUPLING_OPTIONS" :key="o.v" :value="o.v">{{ o.label }}</option>
              </select>
              <ChevronDown :size="12" class="absolute right-2 top-1/2 -translate-y-1/2 text-muted pointer-events-none" />
            </div>
          </div>

          <!-- Activity -->
          <div class="space-y-1">
            <label class="text-xs font-medium text-muted">Activity modifier</label>
            <div class="relative">
              <select v-model="paramActivity" :disabled="isRunning"
                class="w-full bg-surface border border-border rounded-lg px-2 py-1.5 text-xs text-text appearance-none focus:outline-none focus:border-accent disabled:opacity-50">
                <option v-for="o in ACTIVITY_OPTIONS" :key="o.v" :value="o.v">{{ o.label }}</option>
              </select>
              <ChevronDown :size="12" class="absolute right-2 top-1/2 -translate-y-1/2 text-muted pointer-events-none" />
            </div>
          </div>

          <!-- Unilateral -->
          <label class="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" v-model="paramUnilateral" :disabled="isRunning"
              class="w-3.5 h-3.5 rounded border-border accent-green-500 cursor-pointer" />
            <span class="text-xs text-text">Unilateral weight bearing (one leg)</span>
          </label>
          <p v-if="selected.size > 1" class="text-xs text-warning leading-relaxed">
            Load/coupling are applied identically to all {{ selected.size }} selected trials — best for
            repeat trials of the same task. Mixed task types may need different values per trial.
          </p>
        </div>

        <!-- Run button -->
        <Button variant="primary" size="lg" :full-width="true"
          :disabled="!canRun" :loading="isRunning || batchRunning" @click="handleRun">
          <template v-if="!isRunning && !batchRunning" #icon><Play :size="15" /></template>
          {{ batchRunning ? `Computing ${batchProgress.current}/${batchProgress.total}…`
            : isRunning ? 'Computing REBA…'
            : `Compute REBA for ${selected.size} trial(s)` }}
        </Button>

        <div v-if="displayError" class="flex items-start gap-2 bg-danger/10 border border-danger/30 rounded-xl p-3">
          <AlertCircle :size="14" class="text-danger flex-shrink-0 mt-0.5" />
          <p class="text-xs text-danger leading-relaxed whitespace-pre-wrap">{{ displayError }}</p>
        </div>

        <!-- View a previously computed result, independent of the batch checkboxes -->
        <div class="rounded-xl border border-border p-4 space-y-2">
          <p class="text-xs font-semibold text-muted ">View Saved Result</p>
          <select v-model="viewTask"
            class="w-full bg-surface border border-border rounded-lg px-2 py-1.5 text-xs text-text focus:outline-none focus:border-accent">
            <option value="">Select a trial…</option>
            <option v-for="t in tasks" :key="t.id" :value="t.id">{{ t.name }}</option>
          </select>
        </div>

        <!-- Params reminder when result is shown -->
        <div v-if="result" class="bg-surface rounded-xl border border-border p-3 space-y-1">
          <p class="text-xs font-semibold text-muted  mb-1.5">Computed with</p>
          <div class="flex flex-wrap gap-1.5">
            <span class="text-xs px-2 py-0.5 rounded-full bg-card border border-border text-muted font-mono">
              load={{ result.params.load }}
            </span>
            <span class="text-xs px-2 py-0.5 rounded-full bg-card border border-border text-muted font-mono">
              coupling={{ result.params.coupling }}
            </span>
            <span class="text-xs px-2 py-0.5 rounded-full bg-card border border-border text-muted font-mono">
              activity={{ result.params.activity }}
            </span>
            <span class="text-xs px-2 py-0.5 rounded-full bg-card border border-border text-muted font-mono">
              unilateral={{ result.params.unilateral }}
            </span>
          </div>
          <p class="text-xs text-muted mt-1">{{ result.n_frames }} frames · {{ result.processing_time_s }}s</p>
        </div>
      </div>

      <!-- ── Right panel ─────────────────────────────────────────────────────── -->
      <div class="col-span-3 h-full overflow-y-auto">
        <Transition name="pf" mode="out-in">

          <!-- Progress — single trial (full step detail) -->
          <div v-if="showProgress && !batchRunning" key="prog"
            class="rounded-xl border border-border p-5 space-y-4">
            <div class="flex items-center justify-between">
              <span class="text-xs font-semibold text-muted ">Computing REBA</span>
              <StatusBadge status="running" />
            </div>
            <ProgressSteps :job-status="jobStatus" />
          </div>

          <!-- Progress — batch (N trials, one at a time) -->
          <div v-else-if="batchRunning" key="batch-prog"
            class="rounded-xl border border-border p-5 space-y-4">
            <div class="flex items-center justify-between">
              <span class="text-xs font-semibold text-muted ">
                Computing REBA — trial {{ batchProgress.current }}/{{ batchProgress.total }}
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
              <p class="text-xs text-muted">Use "View Saved Result" (left panel) to see full detail for any one trial.</p>
            </div>
          </div>

          <!-- Results -->
          <div v-else-if="showResult" key="res" class="space-y-4">

            <!-- Success + open folder -->
            <div class="bg-success/10 border border-success/30 rounded-xl p-4 flex items-center justify-between">
              <div>
                <div class="flex items-center gap-2 mb-0.5">
                  <CheckCircle :size="13" class="text-success" />
                  <p class="text-sm font-semibold text-success">REBA scores computed</p>
                </div>
                <p class="text-xs text-muted">{{ result!.task_name }}</p>
              </div>
              <Button variant="ghost" size="sm" @click="openProjectFolder(result!.task_name + '/reba')">
                <template #icon><FolderOpen :size="12" /></template>
                Open output
              </Button>
            </div>

            <!-- ── REBA Score Summary ─────────────────────────────────────────── -->
            <div class="rounded-xl border border-border p-4 space-y-3">
              <p class="text-xs font-semibold text-muted ">REBA Total Score — Summary</p>

              <div class="grid grid-cols-5 gap-2">
                <div v-for="(label, key) in { mean: 'Mean', median: 'Median', max: 'Max', min: 'Min', std: 'Std Dev' }"
                  :key="key" class="bg-surface rounded-lg p-2.5 text-center">
                  <p class="text-xs text-muted  mb-0.5">{{ label }}</p>
                  <p class="text-lg font-bold text-text font-mono">
                    {{ (result!.reba_summary as any)[key]?.toFixed ? (result!.reba_summary as any)[key].toFixed(2) : (result!.reba_summary as any)[key] }}
                  </p>
                  <p class="text-xs text-muted">/ 15</p>
                </div>
              </div>

              <!-- Risk distribution -->
              <div class="space-y-1.5">
                <p class="text-xs text-muted  font-semibold">Risk distribution</p>
                <div class="flex flex-wrap gap-1.5">
                  <span
                    v-for="(count, risk) in result!.reba_summary.risk_distribution"
                    :key="risk"
                    class="flex items-center gap-1 px-2 py-1 rounded-lg border text-xs font-medium"
                    :class="riskClass(String(risk))">
                    {{ riskLabel(String(risk)) }}: {{ count }} frames ({{ pct(count) }}%)
                  </span>
                </div>
              </div>

              <!-- Peak frame -->
              <div class="bg-surface rounded-lg p-2.5">
                <p class="text-xs text-muted  mb-1">Peak frame (frame {{ result!.reba_summary.peak_frame }})</p>
                <div class="flex flex-wrap gap-1.5">
                  <span v-for="(label, key) in {
                    peak_trunk: 'Trunk', peak_neck: 'Neck', peak_legs: 'Legs',
                    peak_upper_arm: 'Upper Arm', peak_lower_arm: 'Lower Arm', peak_wrist: 'Wrist',
                    peak_score_a: 'Score A', peak_score_b: 'Score B',
                  }" :key="key"
                    class="text-xs px-2 py-0.5 rounded-full bg-card border border-border text-text font-mono">
                    {{ label }}: {{ (result!.reba_summary as any)[key] }}
                  </span>
                </div>
              </div>
            </div>

            <!-- ── REBA Component Statistics ──────────────────────────────────── -->
            <div class="rounded-xl border border-border p-4 space-y-3">
              <p class="text-xs font-semibold text-muted ">REBA Component Scores — Statistics (all frames)</p>
              <div class="overflow-x-auto">
                <table class="w-full text-xs">
                  <thead>
                    <tr class="border-b border-border">
                      <th class="text-left text-muted font-semibold pb-2 pr-3">Component</th>
                      <th class="text-right text-muted font-semibold pb-2 px-2">Max</th>
                      <th class="text-right text-muted font-semibold pb-2 px-2">Mean</th>
                      <th class="text-right text-muted font-semibold pb-2 px-2">Median</th>
                      <th class="text-right text-muted font-semibold pb-2 px-2">Min</th>
                      <th class="text-right text-muted font-semibold pb-2 pl-2">Std</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="key in componentKeys" :key="key"
                      class="border-b border-border/40 hover:bg-surface/50 transition-colors"
                      :class="key === 'reba_total' ? 'font-bold text-text' : 'text-muted'">
                      <td class="py-1.5 pr-3 text-left">
                        {{ COMPONENT_LABELS[key] }}
                        <span class="ml-1 text-xs text-muted/60 font-normal">/ {{ COMPONENT_MAX[key] }}</span>
                      </td>
                      <td class="py-1.5 px-2 text-right font-mono">{{ result!.component_stats[key]?.max }}</td>
                      <td class="py-1.5 px-2 text-right font-mono">{{ result!.component_stats[key]?.mean.toFixed(2) }}</td>
                      <td class="py-1.5 px-2 text-right font-mono">{{ result!.component_stats[key]?.median.toFixed(2) }}</td>
                      <td class="py-1.5 px-2 text-right font-mono">{{ result!.component_stats[key]?.min }}</td>
                      <td class="py-1.5 pl-2 text-right font-mono">{{ result!.component_stats[key]?.std.toFixed(2) }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <!-- ── Joint Angle Statistics ─────────────────────────────────────── -->
            <div class="rounded-xl border border-border p-4 space-y-3">
              <p class="text-xs font-semibold text-muted ">Joint Angles — Statistics (all frames, degrees°)</p>
              <div class="overflow-x-auto">
                <table class="w-full text-xs">
                  <thead>
                    <tr class="border-b border-border">
                      <th class="text-left text-muted font-semibold pb-2 pr-3">Angle</th>
                      <th class="text-right text-muted font-semibold pb-2 px-2">Max °</th>
                      <th class="text-right text-muted font-semibold pb-2 px-2">Mean °</th>
                      <th class="text-right text-muted font-semibold pb-2 px-2">Median °</th>
                      <th class="text-right text-muted font-semibold pb-2 px-2">Min °</th>
                      <th class="text-right text-muted font-semibold pb-2 pl-2">Std °</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="key in angleKeys" :key="key"
                      class="border-b border-border/40 hover:bg-surface/50 transition-colors text-muted">
                      <td class="py-1.5 pr-3 text-left text-text">{{ ANGLE_LABELS[key] ?? key }}</td>
                      <td class="py-1.5 px-2 text-right font-mono">{{ statCell(result!.angle_summary[key], 'max') }}</td>
                      <td class="py-1.5 px-2 text-right font-mono">{{ statCell(result!.angle_summary[key], 'mean') }}</td>
                      <td class="py-1.5 px-2 text-right font-mono">{{ statCell(result!.angle_summary[key], 'median') }}</td>
                      <td class="py-1.5 px-2 text-right font-mono">{{ statCell(result!.angle_summary[key], 'min') }}</td>
                      <td class="py-1.5 pl-2 text-right font-mono">{{ statCell(result!.angle_summary[key], 'std') }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <!-- Output files note -->
            <div class="bg-surface rounded-xl border border-border p-3 flex items-start gap-2">
              <Info :size="12" class="text-accent mt-0.5 flex-shrink-0" />
              <div>
                <p class="text-xs text-muted">
                  Per-frame data saved to <span class="font-mono text-text">outputs/{{ result!.task_name }}/reba/</span>
                </p>
                <p class="text-xs text-muted mt-0.5">angles.csv — all 19 angles per frame · reba_scores.csv — all component scores per frame</p>
              </div>
            </div>

          </div>

          <!-- Empty -->
          <div v-else key="empty" class="flex flex-col items-center justify-center h-full text-muted space-y-3">
            <Activity :size="40" class="opacity-20" />
            <div class="text-center">
              <p class="text-sm font-medium">No REBA data yet</p>
              <p class="text-xs mt-1">Select a task with a synced CSV and click Compute</p>
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
