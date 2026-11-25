<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { Camera, Crosshair } from '@lucide/vue'
import { listProjects as listCalib } from '@renderer/calib/lib/api'
import { listProjects as listSync } from '@renderer/sync/lib/api'
import { qualityColor } from '@renderer/calib/lib/types'
import { syncMethodColor } from '@renderer/sync/lib/types'
import { STAGES } from '@renderer/sync/lib/stages'
import { useServerStatus as useCalibStatus } from '@renderer/calib/composables/useServerStatus'
import { useServerStatus as useSyncStatus } from '@renderer/sync/composables/useServerStatus'
import type { ProjectEntry as CalibEntry } from '@renderer/calib/lib/types'
import type { ProjectEntry as SyncEntry } from '@renderer/sync/lib/types'

const router = useRouter()
const calib  = ref<CalibEntry[]>([])
const sync   = ref<SyncEntry[]>([])
const { state: calibState } = useCalibStatus()
const { state: syncState }  = useSyncStatus()

onMounted(async () => {
  try { calib.value = await listCalib() } catch { /* backend offline */ }
  try { sync.value = await listSync() } catch { /* backend offline */ }
})

const services = computed(() => [
  { name: 'Calibration', state: calibState.value, n: calib.value.length, unit: 'results' },
  { name: 'Sync',        state: syncState.value,  n: sync.value.length,  unit: 'trials' },
])
const STATUS = { online: 'Online', offline: 'Offline', checking: 'Connecting' } as const
const dot = (s: string) => s === 'online' ? 'bg-success' : s === 'offline' ? 'bg-danger' : 'bg-warning animate-pulse'

const STEPS = [
  { to: '/intrinsic', label: 'Intrinsic', icon: Camera },
  { to: '/extrinsic', label: 'Extrinsic', icon: Crosshair },
  ...STAGES.map(s => ({ to: s.to, label: s.label, icon: s.icon })),
]
const recentCalib = computed(() => calib.value.slice(0, 3))
const recentSync  = computed(() => sync.value.slice(0, 3))
</script>

<template>
  <div class="grid grid-cols-[minmax(0,1fr)_340px] gap-6 items-start animate-fade-in">
    <div class="flex flex-col gap-6 min-w-0">
      <section class="border border-border rounded p-6 flex flex-col gap-5">
        <div>
          <h1 class="text-[40px] leading-[1.1] font-bold tracking-tight">ViconMoCap Sync Studio</h1>
          <p class="text-base text-muted mt-2 max-w-[620px] leading-normal">
            One workspace for the whole chain: calibrate the phone camera against Vicon, then synchronize, overlay, score and export each trial.
          </p>
        </div>
        <div class="flex items-center gap-2 flex-wrap">
          <template v-for="(s, i) in STEPS" :key="s.to">
            <button class="flex items-center gap-2 h-10 px-3 border rounded hover:bg-tint hover:border-accent transition-colors"
              :class="i === 0 ? 'border-accent' : 'border-border'" @click="router.push(s.to)">
              <component :is="s.icon" :size="15" :stroke-width="1.5" :class="i === 0 ? 'text-accent' : 'text-muted'" />
              <span class="text-[13px] font-medium">{{ s.label }}</span>
            </button>
            <span v-if="i < STEPS.length - 1" class="w-3 border-t border-dim" />
          </template>
        </div>
      </section>

      <section>
        <h2 class="text-base font-semibold mb-3">1 · Calibration</h2>
        <div class="grid grid-cols-2 gap-4">
          <button v-for="t in [
            { to: '/intrinsic', icon: Camera, title: 'Intrinsic', desc: 'Camera matrix and lens distortion from a ChArUco recording.', needs: 'Video or images of the board' },
            { to: '/extrinsic', icon: Crosshair, title: 'Extrinsic', desc: 'Camera position and orientation in the Vicon world.', needs: 'Phone video, Vicon data, intrinsics' },
          ]" :key="t.to" class="border border-border rounded p-5 text-left flex flex-col gap-3 hover:border-accent hover:bg-tint transition-colors" @click="router.push(t.to)">
            <component :is="t.icon" :size="22" :stroke-width="1.5" class="text-accent" />
            <div class="text-lg font-semibold">{{ t.title }}</div>
            <div class="text-sm text-muted leading-normal">{{ t.desc }}</div>
            <div class="text-xs text-muted">Needs: <span class="text-text">{{ t.needs }}</span></div>
          </button>
        </div>
      </section>

      <section>
        <h2 class="text-base font-semibold mb-3">2 · Trial pipeline</h2>
        <div class="grid grid-cols-3 gap-4">
          <button v-for="stage in STAGES" :key="stage.to"
            class="border border-border rounded p-5 text-left flex flex-col gap-3 hover:border-accent hover:bg-tint transition-colors"
            @click="router.push(stage.to)">
            <div class="flex justify-between items-center">
              <component :is="stage.icon" :size="22" :stroke-width="1.5" class="text-accent" />
              <span class="font-mono text-xs font-medium text-muted">{{ String(stage.num).padStart(2, '0') }}</span>
            </div>
            <div class="text-lg font-semibold">{{ stage.label }}</div>
            <div class="text-sm text-muted leading-normal">{{ stage.desc }}</div>
            <div class="text-xs text-muted">{{ stage.needs }}</div>
          </button>
        </div>
      </section>

      <section class="border border-border rounded p-6 flex flex-col gap-3">
        <h2 class="text-base font-semibold">How the two parts connect</h2>
        <div v-for="t in [
          'Run Intrinsic once per camera and Extrinsic once per recording session. If the camera moved, redo the extrinsic.',
          'Sync and Overlay can use the latest calibration results directly: pick them from the Calibration panel instead of uploading files again.',
          'The XCP file from Vicon Nexus is required for Sync: it stores the exact capture start time.',
          'Always check the Overlay video before trusting a trial. If the skeleton drifts, redo the Sync.',
        ]" :key="t" class="flex gap-2.5 items-baseline text-sm">
          <span class="w-1.5 h-1.5 bg-accent flex-none rounded-sm" />{{ t }}
        </div>
      </section>
    </div>

    <div class="flex flex-col gap-5">
      <section class="border border-border rounded p-6 flex flex-col gap-4">
        <h2 class="text-base font-semibold">Backends</h2>
        <div v-for="s in services" :key="s.name" class="flex items-center justify-between border border-border rounded px-3 py-2.5">
          <div>
            <div class="flex items-center gap-2 text-[13px] font-medium" role="status">
              <span :class="['w-2 h-2 rounded-full', dot(s.state)]" />{{ s.name }}
            </div>
            <div class="text-xs text-muted mt-0.5">{{ STATUS[s.state] }}</div>
          </div>
          <div class="text-right">
            <div class="font-mono text-[22px] font-medium leading-none">{{ s.n }}</div>
            <div class="text-xs text-muted mt-1">{{ s.unit }}</div>
          </div>
        </div>
      </section>

      <section class="border border-border rounded p-5 flex flex-col gap-1.5">
        <h2 class="text-base font-semibold mb-2">Recent calibrations</h2>
        <p v-if="recentCalib.length === 0" class="text-[13px] text-muted py-2">No results yet.</p>
        <div v-for="(r, i) in recentCalib" :key="r.id" :class="['flex items-center justify-between gap-2 py-2', i < recentCalib.length - 1 && 'border-b border-border']">
          <div class="min-w-0">
            <div class="font-mono text-[13px] font-medium truncate">{{ r.name }} <span class="text-muted">· {{ r.type }}</span></div>
            <div class="text-xs text-muted">{{ r.created_at ? new Date(r.created_at).toLocaleDateString() : '' }}</div>
          </div>
          <div class="flex items-center gap-2 flex-shrink-0">
            <span class="font-mono text-[13px]">{{ r.rms_px?.toFixed(2) ?? '—' }}</span>
            <span class="w-2 h-2 rounded-full" :style="{ background: qualityColor(r.quality) }" />
          </div>
        </div>
      </section>

      <section class="border border-border rounded p-5 flex flex-col gap-1.5">
        <div class="flex items-center justify-between mb-2">
          <h2 class="text-base font-semibold">Recent sessions</h2>
          <button class="text-xs text-accent hover:text-accent-hover font-medium" @click="router.push('/projects')">View all</button>
        </div>
        <p v-if="recentSync.length === 0" class="text-[13px] text-muted py-2">No sessions yet. Start with C3D Convert.</p>
        <div v-for="(p, i) in recentSync" :key="p.id" :class="['flex items-center justify-between gap-2 py-2', i < recentSync.length - 1 && 'border-b border-border']">
          <div class="min-w-0">
            <div class="font-mono text-[13px] font-medium truncate">{{ p.name }}</div>
            <div class="text-xs text-muted">{{ p.created_at?.slice(0, 10) ?? '' }} · {{ p.has_overlay ? 'overlay ✓' : 'sync only' }}</div>
          </div>
          <span class="w-2 h-2 rounded-full flex-shrink-0" :style="{ background: syncMethodColor(p.sync_method ?? '') }" :title="p.sync_method ?? 'no sync'" />
        </div>
      </section>
    </div>
  </div>
</template>
