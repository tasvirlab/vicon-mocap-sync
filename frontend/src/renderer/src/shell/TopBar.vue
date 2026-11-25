<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { Minus, X } from '@lucide/vue'
import { useServerStatus as useCalibStatus } from '@renderer/calib/composables/useServerStatus'
import { useServerStatus as useSyncStatus } from '@renderer/sync/composables/useServerStatus'
import { cn } from '@renderer/shared/cn'

defineProps<{ appName: string }>()

// Two backends (calibration :8765, sync :8766) behind one window; each has its own dot.
const { state: calib } = useCalibStatus()
const { state: sync }  = useSyncStatus()
const SERVICES = computed(() => [
  { name: 'Calibration', state: calib.value },
  { name: 'Sync',        state: sync.value },
])
const dot = (s: string) => s === 'online' ? 'bg-success' : s === 'offline' ? 'bg-danger' : 'bg-warning animate-pulse'
const LABEL = { online: 'online', offline: 'offline', checking: 'connecting' } as const

const isMaximized = ref(false)
let pollInterval: ReturnType<typeof setInterval> | null = null

onMounted(() => {
  const win = (window as any).electron
  if (!win?.windowIsMaximized) return
  const poll = () => win.windowIsMaximized().then((v: boolean) => { isMaximized.value = v }).catch(() => {})
  poll()
  pollInterval = setInterval(poll, 1000)
})
onUnmounted(() => { if (pollInterval) clearInterval(pollInterval) })

function minimize()       { (window as any).electron?.windowMinimize() }
function toggleMaximize() { (window as any).electron?.windowMaximize() }
function close() {
  const win = (window as any).electron
  if (win?.windowClose) win.windowClose()
  else window.close()
}
const NO_DRAG = '-webkit-app-region: no-drag'
</script>

<template>
  <div class="flex items-center justify-between h-10 bg-surface border-b border-border titlebar-drag pl-4 flex-shrink-0">
    <span class="text-[15px] font-semibold select-none pointer-events-none">{{ appName }}</span>

    <div class="flex items-center gap-5 h-full">
      <div class="flex items-center gap-4 text-[13px]" :style="NO_DRAG" role="status">
        <span v-for="s in SERVICES" :key="s.name" class="flex items-center gap-2" :title="`${s.name} backend ${LABEL[s.state]}`">
          <span :class="cn('w-2 h-2', dot(s.state))" />
          <span :class="s.state === 'online' ? '' : 'text-muted'">{{ s.name }}</span>
        </span>
      </div>
      <div class="flex h-full" :style="NO_DRAG">
        <button title="Minimize" class="w-[46px] grid place-items-center text-muted hover:bg-tint hover:text-text transition-colors" @click="minimize">
          <Minus :size="14" :stroke-width="1.5" />
        </button>
        <button :title="isMaximized ? 'Restore' : 'Maximize'" class="w-[46px] grid place-items-center text-muted hover:bg-tint hover:text-text transition-colors" @click="toggleMaximize">
          <svg v-if="isMaximized" width="12" height="12" viewBox="0 0 11 11" fill="none" stroke="currentColor" stroke-width="1.1">
            <rect x="2.5" y="0.55" width="8" height="8" rx="0.3" /><path d="M0.55 2.5 V10.45 H8.5" stroke-linecap="square" />
          </svg>
          <svg v-else width="12" height="12" viewBox="0 0 11 11" fill="none" stroke="currentColor" stroke-width="1.1">
            <rect x="0.55" y="0.55" width="9.9" height="9.9" rx="0.3" />
          </svg>
        </button>
        <button title="Close" class="w-[46px] grid place-items-center text-muted hover:bg-danger hover:text-white transition-colors" @click="close">
          <X :size="14" :stroke-width="1.5" />
        </button>
      </div>
    </div>
  </div>
</template>
