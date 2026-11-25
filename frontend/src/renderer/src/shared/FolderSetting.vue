<script setup lang="ts">
// One configurable folder: shows the current path and lets the user pick another one.
// In the desktop app the native folder dialog is used; in a plain browser the path is typed.
import { ref, computed } from 'vue'
import { FolderOpen } from '@lucide/vue'
import Button from '@renderer/calib/components/ui/Button.vue'
import AlertBox from '@renderer/calib/components/ui/AlertBox.vue'

const props = defineProps<{
  title: string
  description: string
  current: string
  apply: (path: string) => Promise<void>
}>()

const draft = ref('')
const busy = ref(false)
const error = ref('')
const saved = ref(false)

const hasDialog = computed(() => typeof (window as any).electron?.openFolderDialog === 'function')

async function commit(path: string) {
  error.value = ''
  saved.value = false
  if (!path || path === props.current) return
  busy.value = true
  try {
    await props.apply(path)
    saved.value = true
  } catch (e: any) {
    error.value = e?.response?.data?.detail ?? e?.message ?? String(e)
  } finally {
    busy.value = false
  }
}

async function browse() {
  const path: string | null = await (window as any).electron.openFolderDialog()
  if (path) await commit(path)
}
</script>

<template>
  <section class="border border-border rounded p-6 grid grid-cols-[220px_minmax(0,1fr)] gap-6">
    <div>
      <h2 class="text-base font-semibold">{{ title }}</h2>
      <p class="text-[13px] text-muted mt-1.5 leading-normal">{{ description }}</p>
    </div>
    <div class="flex flex-col gap-3 min-w-0">
      <span class="text-xs font-medium text-muted">Current folder</span>
      <code class="h-9 flex items-center bg-surface border border-border rounded px-2.5 font-mono text-sm truncate" :title="current">{{ current || '…' }}</code>

      <div v-if="hasDialog">
        <Button variant="secondary" :loading="busy" @click="browse">
          <template #icon><FolderOpen :size="14" :stroke-width="1.5" /></template>Choose folder…
        </Button>
      </div>
      <form v-else class="flex gap-2" @submit.prevent="commit(draft.trim())">
        <input v-model="draft" placeholder="Absolute folder path" aria-label="Folder path"
          class="flex-1 h-9 bg-surface border border-border rounded px-2.5 font-mono text-sm outline-none focus:border-accent" />
        <Button variant="secondary" :loading="busy" :disabled="!draft.trim()" @click="commit(draft.trim())">Apply</Button>
      </form>

      <AlertBox v-if="error" tone="error">{{ error }}</AlertBox>
      <AlertBox v-else-if="saved" tone="info">Saved. New results are written to this folder.</AlertBox>
    </div>
  </section>
</template>
