<script setup lang="ts">
// Calibration → Sync/Overlay handoff. Lists the results the calibration backend already holds
// (exports/) and turns the chosen intrinsic + extrinsic into File objects, so nothing is uploaded twice.
// Defaults to the newest result of each type.
import { ref, computed, onMounted } from 'vue'
import { listProjects, projectFileUrl } from '@renderer/calib/lib/api'
import type { ProjectEntry } from '@renderer/calib/lib/types'
import Button from '@renderer/sync/components/ui/Button.vue'

const props = defineProps<{ disabled?: boolean }>()
const emit = defineEmits<{ pick: [files: { intrinsics: File | null; extrinsics: File | null }] }>()

const projects = ref<ProjectEntry[]>([])
const offline  = ref(false)
const busy     = ref(false)
const error    = ref('')
const intrinsicId = ref('')
const extrinsicId = ref('')

const byNewest = (a: ProjectEntry, b: ProjectEntry) => b.created_at.localeCompare(a.created_at)
const intrinsics = computed(() => projects.value.filter(p => p.type === 'intrinsic').sort(byNewest))
const extrinsics = computed(() => projects.value.filter(p => p.type === 'extrinsic').sort(byNewest))

onMounted(async () => {
  try {
    projects.value = await listProjects()
    intrinsicId.value = intrinsics.value[0]?.id ?? ''
    extrinsicId.value = extrinsics.value[0]?.id ?? ''
  } catch { offline.value = true }
})

const label = (p: ProjectEntry) => `${p.name} · ${p.rms_px.toFixed(2)} px · ${new Date(p.created_at).toLocaleDateString()}`

async function fetchJson(id: string, names: string[], saveAs: string): Promise<File | null> {
  if (!id) return null
  for (const n of names) {
    const r = await fetch(projectFileUrl(id, n))
    if (r.ok) return new File([await r.blob()], saveAs, { type: 'application/json' })
  }
  throw new Error(`No ${names[0]} in ${id}`)
}

async function use() {
  busy.value = true; error.value = ''
  try {
    const [i, e] = await Promise.all([
      fetchJson(intrinsicId.value, ['intrinsic.json', 'intrinsic_params.json'], 'intrinsic.json'),
      fetchJson(extrinsicId.value, ['extrinsics.json', 'extrinsic_params.json'], 'extrinsics.json'),
    ])
    emit('pick', { intrinsics: i, extrinsics: e })
  } catch (err: any) {
    error.value = err?.message ?? String(err)
  } finally { busy.value = false }
}
</script>

<template>
  <div class="border border-border rounded p-3 flex flex-col gap-2.5">
    <div class="flex items-center justify-between">
      <span class="text-xs font-medium text-muted">From Calibration</span>
      <span v-if="offline" class="text-xs text-warning">Calibration backend offline</span>
    </div>
    <template v-if="!offline && (intrinsics.length || extrinsics.length)">
      <label class="flex flex-col gap-1">
        <span class="text-xs text-muted">Intrinsic</span>
        <select v-model="intrinsicId" :disabled="props.disabled || busy"
          class="h-9 bg-surface border border-border rounded px-2.5 font-mono text-xs outline-none focus:border-accent disabled:opacity-45">
          <option value="">none</option>
          <option v-for="p in intrinsics" :key="p.id" :value="p.id">{{ label(p) }}</option>
        </select>
      </label>
      <label class="flex flex-col gap-1">
        <span class="text-xs text-muted">Extrinsic</span>
        <select v-model="extrinsicId" :disabled="props.disabled || busy"
          class="h-9 bg-surface border border-border rounded px-2.5 font-mono text-xs outline-none focus:border-accent disabled:opacity-45">
          <option value="">none</option>
          <option v-for="p in extrinsics" :key="p.id" :value="p.id">{{ label(p) }}</option>
        </select>
      </label>
      <Button variant="secondary" size="sm" :loading="busy" :disabled="props.disabled || (!intrinsicId && !extrinsicId)" @click="use">Use selected calibration</Button>
      <p v-if="error" class="text-xs text-danger">{{ error }}</p>
    </template>
    <p v-else-if="!offline" class="text-xs text-muted">No calibration results yet. Run Intrinsic and Extrinsic first, or upload files below.</p>
  </div>
</template>
