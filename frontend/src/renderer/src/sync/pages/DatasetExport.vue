<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import {
  Boxes, Database, Play, RotateCcw, AlertCircle, Info, CheckCircle, Download,
} from '@lucide/vue'
import Button from '@renderer/sync/components/ui/Button.vue'
import PageHeader from '@renderer/sync/components/ui/PageHeader.vue'
import StatusBadge from '@renderer/sync/components/ui/StatusBadge.vue'
import ProgressSteps from '@renderer/sync/components/ui/ProgressSteps.vue'
import MetricsCard from '@renderer/sync/components/ui/MetricsCard.vue'
import MetricsGrid from '@renderer/sync/components/ui/MetricsGrid.vue'
import { useSyncJob } from '@renderer/sync/composables/useSyncJob'
import {
  listDatasetTasks, listDatasetExports, startDatasetExport, getDatasetResult, datasetDownloadUrl,
} from '@renderer/sync/lib/api'
import type { TrialListEntry, DatasetExportResult, DatasetExportEntry } from '@renderer/sync/lib/types'

const { jobId, jobStatus, isRunning, error, startJob, reset: resetJob } = useSyncJob()

// ── Trial selection ──────────────────────────────────────────────────────────────
const tasks = ref<TrialListEntry[]>([])
const selected = ref<Set<string>>(new Set())
const loadingTasks = ref(false)

// ── Config ───────────────────────────────────────────────────────────────────────
const windowSize = ref(243)
const stepSize = ref(81)
const testSplitMode = ref<'auto' | 'manual'>('auto')
const testFraction = ref(0.2)
const testSubjectsInput = ref('')   // comma-separated SUBIDs for manual mode

// ── Past exports ─────────────────────────────────────────────────────────────────
const pastExports = ref<DatasetExportEntry[]>([])

// ── Results ──────────────────────────────────────────────────────────────────────
const result = ref<DatasetExportResult | null>(null)
const fetchError = ref<string | null>(null)

async function loadTasks() {
  loadingTasks.value = true
  try {
    tasks.value = await listDatasetTasks()
    pastExports.value = await listDatasetExports()
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

const subjectsPreview = computed(() => {
  const ids = new Set(
    [...selected.value]
      .map(name => tasks.value.find(t => t.task_name === name)?.subject_id)
      .filter((s): s is string => !!s),
  )
  return [...ids].sort()
})

async function handleRun() {
  result.value = null
  fetchError.value = null
  try {
    const taskNames = [...selected.value]
    const manualSubjects = testSplitMode.value === 'manual'
      ? testSubjectsInput.value.split(',').map(s => s.trim()).filter(Boolean)
      : null
    const { job_id } = await startDatasetExport(
      taskNames, windowSize.value, stepSize.value, manualSubjects, testFraction.value,
    )
    startJob(job_id, 'dataset')
  } catch (err: unknown) {
    fetchError.value = `Failed to start: ${err instanceof Error ? err.message : String(err)}`
  }
}

function handleReset() {
  resetJob()
  result.value = fetchError.value = null
}

watch(() => jobStatus.value?.status, (status) => {
  if (status === 'completed' && jobId.value) {
    getDatasetResult(jobId.value)
      .then(r => { result.value = r; loadTasks() })
      .catch(e => { fetchError.value = `Failed to load result: ${e?.message}` })
  }
})

const canRun = computed(() => !isRunning.value && selected.value.size > 0)
const displayError = computed(() => error.value || fetchError.value)
const showProgress = computed(() => isRunning.value && !!jobStatus.value)
const showResult = computed(() => !isRunning.value && !!result.value)
const showEmpty = computed(() => !isRunning.value && !result.value)
</script>

<template>
  <div class="p-6 animate-fade-in flex flex-col h-screen gap-5">

    <PageHeader title="Dataset Export" :icon="Boxes" color="#EC4899" :step="6"
      subtitle="Package synced.csv + pose2d.csv into windowed train/test .npz for 3D pose-lifting models — subject-wise split, train-only spine calibration.">
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
            <p class="text-xs text-muted">No trials with pose2d.csv found. Run 2D Pose Extraction first.</p>
          </div>

          <template v-else>
            <div class="flex gap-2 text-xs">
              <button class="text-accent hover:text-accent/80" @click="selectAll">Select all</button>
              <span class="text-muted">·</span>
              <button class="text-accent hover:text-accent/80" @click="selectNone">Clear</button>
              <span class="text-muted ml-auto">{{ selected.size }} / {{ tasks.length }} selected</span>
            </div>
            <div class="space-y-1.5 max-h-56 overflow-y-auto">
              <label v-for="task in tasks" :key="task.task_name"
                class="flex items-center gap-2.5 p-2.5 rounded-lg border cursor-pointer transition-colors"
                :class="selected.has(task.task_name) ? 'border-accent3/50 bg-accent3/10' : 'border-border hover:border-accent3/30 bg-surface'">
                <input type="checkbox" class="w-3.5 h-3.5 accent-accent3"
                  :checked="selected.has(task.task_name)" @change="toggle(task.task_name)" />
                <span class="text-xs font-medium text-text truncate flex-1">{{ task.task_name }}</span>
                <span class="text-xs px-1.5 py-0.5 rounded-full bg-card border border-border text-muted font-mono">{{ task.subject_id }}</span>
              </label>
            </div>
            <p v-if="selected.size > 0" class="text-xs text-muted">
              Subjects in selection: {{ subjectsPreview.join(', ') }}
            </p>
          </template>
        </div>

        <!-- Window config -->
        <div class="rounded-xl border border-border p-4 space-y-3">
          <p class="text-xs font-semibold text-muted ">Window Config</p>
          <p class="text-xs text-muted">243/81 is the common VideoPose3D-style window/stride; confirm it matches your model's configuration.</p>
          <div class="grid grid-cols-2 gap-2">
            <label>
              <span class="text-xs text-muted">Window size (frames)</span>
              <input type="number" v-model.number="windowSize" min="9" :disabled="isRunning"
                class="mt-1 w-full bg-surface border border-border rounded-lg px-2 py-1.5 text-xs text-text focus:outline-none focus:border-accent disabled:opacity-50" />
            </label>
            <label>
              <span class="text-xs text-muted">Step size (frames)</span>
              <input type="number" v-model.number="stepSize" min="1" :disabled="isRunning"
                class="mt-1 w-full bg-surface border border-border rounded-lg px-2 py-1.5 text-xs text-text focus:outline-none focus:border-accent disabled:opacity-50" />
            </label>
          </div>
        </div>

        <!-- Subject-wise split -->
        <div class="rounded-xl border border-border p-4 space-y-3">
          <p class="text-xs font-semibold text-muted ">Subject-Wise Split</p>
          <div class="flex gap-2">
            <button class="flex-1 text-xs py-1.5 rounded-lg border"
              :class="testSplitMode === 'auto' ? 'border-accent3/50 bg-accent3/10 text-text' : 'border-border text-muted'"
              @click="testSplitMode = 'auto'">Auto (fraction)</button>
            <button class="flex-1 text-xs py-1.5 rounded-lg border"
              :class="testSplitMode === 'manual' ? 'border-accent3/50 bg-accent3/10 text-text' : 'border-border text-muted'"
              @click="testSplitMode = 'manual'">Manual SUBIDs</button>
          </div>

          <label v-if="testSplitMode === 'auto'">
            <div class="flex justify-between text-xs text-muted mb-1">
              <span>Test fraction (held-out subjects)</span>
              <span class="font-mono text-text">{{ (testFraction * 100).toFixed(0) }}%</span>
            </div>
            <input type="range" v-model.number="testFraction" min="0.1" max="0.5" step="0.05" :disabled="isRunning" class="w-full" />
          </label>
          <label v-else>
            <span class="text-xs text-muted">Test SUBIDs (comma-separated)</span>
            <input v-model="testSubjectsInput" placeholder="SUB08, SUB09, SUB10" :disabled="isRunning"
              class="mt-1 w-full bg-surface border border-border rounded-lg px-2 py-1.5 text-xs text-text focus:outline-none focus:border-accent disabled:opacity-50" />
          </label>
        </div>

        <p v-if="!isRunning && tasks.length > 0 && selected.size === 0"
          class="text-xs text-warning text-center">
          Select at least one trial to enable export.
        </p>
        <Button variant="primary" size="lg" :full-width="true" :disabled="!canRun" :loading="isRunning" @click="handleRun">
          <template v-if="!isRunning" #icon><Play :size="15" /></template>
          {{ isRunning ? 'Exporting…' : 'Export Dataset' }}
        </Button>

        <div v-if="displayError" class="flex items-start gap-2 bg-danger/10 border border-danger/30 rounded-xl p-3">
          <AlertCircle :size="14" class="text-danger flex-shrink-0 mt-0.5" />
          <p class="text-xs text-danger leading-relaxed whitespace-pre-wrap">{{ displayError }}</p>
        </div>

        <!-- Past exports -->
        <div v-if="pastExports.length > 0" class="bg-surface rounded-xl border border-border p-3 space-y-2">
          <p class="text-xs font-semibold text-muted ">Past exports</p>
          <div v-for="exp in pastExports" :key="exp.timestamp" class="flex items-center gap-2 text-xs">
            <span class="text-muted font-mono flex-1">{{ exp.timestamp }}</span>
            <a :href="datasetDownloadUrl(exp.timestamp, 'construction_vicon_train.npz')" class="text-accent hover:text-accent/80">train</a>
            <a :href="datasetDownloadUrl(exp.timestamp, 'construction_vicon_test.npz')" class="text-accent hover:text-accent/80">test</a>
          </div>
        </div>
      </div>

      <!-- ── Right panel ─────────────────────────────────────────────────────── -->
      <div class="col-span-3 h-full overflow-y-auto">
        <Transition name="pf" mode="out-in">

          <div v-if="showProgress" key="prog" class="rounded-xl border border-border p-5 space-y-4">
            <div class="flex items-center justify-between">
              <span class="text-xs font-semibold text-muted ">Exporting</span>
              <StatusBadge status="running" />
            </div>
            <ProgressSteps :job-status="jobStatus" />
          </div>

          <div v-else-if="showResult" key="res" class="space-y-4">
            <div class="bg-success/10 border border-success/30 rounded-xl p-4 flex items-center justify-between">
              <div>
                <div class="flex items-center gap-2 mb-0.5">
                  <CheckCircle :size="13" class="text-success" />
                  <p class="text-sm font-semibold text-success">Dataset exported</p>
                </div>
                <p class="text-xs text-muted font-mono">{{ result!.export_dir }}</p>
              </div>
              <div class="flex gap-2">
                <a :href="datasetDownloadUrl(result!.timestamp, 'construction_vicon_train.npz')">
                  <Button variant="ghost" size="sm"><template #icon><Download :size="12" /></template>train.npz</Button>
                </a>
                <a :href="datasetDownloadUrl(result!.timestamp, 'construction_vicon_test.npz')">
                  <Button variant="ghost" size="sm"><template #icon><Download :size="12" /></template>test.npz</Button>
                </a>
              </div>
            </div>

            <MetricsGrid :cols="4">
              <MetricsCard label="Train windows" :value="result!.n_train_windows" :delay="0" />
              <MetricsCard label="Test windows" :value="result!.n_test_windows" :delay="0.05" />
              <MetricsCard label="Train trials" :value="result!.n_train_trials" :delay="0.1" />
              <MetricsCard label="Test trials" :value="result!.n_test_trials" :delay="0.15" />
            </MetricsGrid>

            <div class="rounded-xl border border-border p-4 space-y-2">
              <p class="text-xs font-semibold text-muted ">Calibrated spine ratios (train-only, frozen)</p>
              <div class="flex gap-3 text-xs font-mono">
                <span class="px-2 py-1 rounded-lg bg-surface border border-border text-text">t_T10 = {{ result!.calibrated_t_t10 }}</span>
                <span class="px-2 py-1 rounded-lg bg-surface border border-border text-text">t_L45 = {{ result!.calibrated_t_l45 }}</span>
              </div>
              <p class="text-xs text-muted">Overrides the placeholder ratios used by 2D Pose Extraction, for both train and test splits.</p>
            </div>

            <div class="rounded-xl border border-border p-4 space-y-2">
              <p class="text-xs font-semibold text-muted ">Subject split</p>
              <p class="text-xs"><span class="text-muted">Train:</span> <span class="font-mono text-text">{{ result!.train_subjects.join(', ') }}</span></p>
              <p class="text-xs"><span class="text-muted">Test:</span> <span class="font-mono text-text">{{ result!.test_subjects.join(', ') }}</span></p>
            </div>

            <div v-if="result!.skipped_trials.length > 0" class="bg-warning/10 border border-warning/30 rounded-xl p-4 space-y-1.5">
              <p class="text-xs font-semibold text-warning">Skipped trials ({{ result!.skipped_trials.length }})</p>
              <p v-for="s in result!.skipped_trials" :key="s.task_name" class="text-xs text-muted">
                <span class="font-mono text-text">{{ s.task_name }}</span> — {{ s.reason }}
              </p>
            </div>

            <img v-if="result!.visualize_png" :src="datasetDownloadUrl(result!.timestamp, 'visualize_check.png')"
              class="w-full rounded-xl border border-border" alt="QA visualization" />
          </div>

          <div v-else key="empty" class="flex flex-col items-center justify-center h-full text-muted space-y-3">
            <Database :size="40" class="opacity-20" />
            <div class="text-center">
              <p class="text-sm font-medium">No dataset exported yet</p>
              <p class="text-xs mt-1">Configure the split and click Export Dataset</p>
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
