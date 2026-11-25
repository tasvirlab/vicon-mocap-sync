<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { FolderOpen, RefreshCw, Camera, Crosshair } from '@lucide/vue'
import Button        from '@renderer/calib/components/ui/Button.vue'
import ConfirmDialog from '@renderer/calib/components/ui/ConfirmDialog.vue'
import QualityBadge  from '@renderer/calib/components/ui/QualityBadge.vue'
import SegmentedControl from '@renderer/calib/components/ui/SegmentedControl.vue'
import { listProjects, deleteProject, projectThumbnailUrl, projectFileUrl, openProjectFolder } from '@renderer/calib/lib/api'
import { qualityColor } from '@renderer/calib/lib/types'
import type { ProjectEntry } from '@renderer/calib/lib/types'

type Filter = 'all' | 'intrinsic' | 'extrinsic'

const projects = ref<ProjectEntry[]>([])
const loading  = ref(true)
const deleting = ref<string | null>(null)
const pending  = ref<ProjectEntry | null>(null)   // project awaiting delete confirmation
const filter   = ref<Filter>('all')
const FILTERS  = [{ value: 'all', label: 'All' }, { value: 'intrinsic', label: 'Intrinsic' }, { value: 'extrinsic', label: 'Extrinsic' }]

const filtered = computed(() =>
  filter.value === 'all' ? projects.value : projects.value.filter(p => p.type === filter.value)
)

async function load() {
  loading.value = true
  try { projects.value = await listProjects() } catch { /* ignore */ }
  finally { loading.value = false }
}
onMounted(load)

async function confirmDelete() {
  const p = pending.value
  if (!p) return
  deleting.value = p.id
  try {
    await deleteProject(p.id)
    projects.value = projects.value.filter(x => x.id !== p.id)
  } finally {
    deleting.value = null
    pending.value  = null
  }
}

function openJson(p: ProjectEntry) {
  const filename = p.type === 'intrinsic' ? 'intrinsic_params.json' : 'extrinsic_params.json'
  const url = projectFileUrl(p.id, filename)
  const el = (window as any).electron
  if (el?.openExternal) el.openExternal(url)
  else window.open(url, '_blank')
}

async function openFolder(p: ProjectEntry) {
  const el = (window as any).electron
  // Prefer showItemInFolder (selects the project folder in Explorer), then openPath,
  // finally the backend POST which calls os.startfile on the server side.
  if (el?.showItemInFolder) el.showItemInFolder(p.output_dir)
  else if (el?.openPath) el.openPath(p.output_dir)
  else await openProjectFolder(p.id)
}
</script>

<template>
  <div class="animate-fade-in">
    <div class="flex items-end justify-between gap-6 mb-6">
      <div>
        <h1 class="text-[32px] leading-[1.1] font-bold tracking-tight">Projects</h1>
        <p class="text-[15px] text-muted mt-1.5">
          Saved calibration results · {{ filtered.length }} of {{ projects.length }}
        </p>
      </div>
      <div class="flex items-center gap-2.5">
        <SegmentedControl v-model="filter" :options="FILTERS" class="min-w-[250px]" />
        <Button variant="secondary" :loading="loading" @click="load">
          <template #icon><RefreshCw :size="14" :stroke-width="1.5" /></template>Refresh
        </Button>
      </div>
    </div>

    <!-- Loading -->
    <div v-if="loading" class="grid grid-cols-3 gap-5">
      <div v-for="i in 6" :key="i" class="h-[290px] rounded shimmer" />
    </div>

    <!-- Empty -->
    <div v-else-if="projects.length === 0" class="border border-dashed border-dim rounded px-6 py-[72px] flex flex-col items-center gap-2.5 text-center">
      <FolderOpen :size="28" :stroke-width="1.5" class="text-muted" />
      <div class="text-lg font-semibold">No results yet</div>
      <div class="text-muted">Run a calibration and it will appear here.</div>
    </div>

    <!-- Empty filtered -->
    <div v-else-if="filtered.length === 0" class="border border-dashed border-dim rounded px-6 py-12 flex flex-col items-center gap-2 text-center">
      <div class="font-medium">No {{ filter }} results</div>
      <Button variant="ghost" size="sm" @click="filter = 'all'">Show all</Button>
    </div>

    <!-- Grid -->
    <div v-else class="grid grid-cols-3 gap-5">
      <article v-for="(p, i) in filtered" :key="p.id" class="border border-border rounded overflow-hidden flex flex-col rise"
        :style="{ animationDelay: `${i * 0.05}s` }">
        <div class="aspect-video hatch border-b border-border relative overflow-hidden">
          <img v-if="p.thumbnail" :src="projectThumbnailUrl(p.id)" :alt="p.name" class="absolute inset-0 w-full h-full object-cover"
            @error="($event.target as HTMLImageElement).style.display = 'none'" />
          <component :is="p.type === 'intrinsic' ? Camera : Crosshair" :size="24" :stroke-width="1.5" class="absolute inset-0 m-auto text-dim" />
          <span class="absolute top-2.5 left-2.5 font-mono text-xs font-medium px-2 py-[3px] border border-muted rounded-sm bg-bg capitalize">{{ p.type }}</span>
          <span class="absolute top-2.5 right-2.5"><QualityBadge :quality="p.quality" /></span>
        </div>
        <div class="p-4 flex flex-col gap-2.5">
          <div class="flex justify-between items-baseline gap-2">
            <span class="font-mono text-[15px] font-medium truncate" :title="p.name">{{ p.name }}</span>
            <span class="font-mono text-xl font-medium flex-shrink-0" :style="{ color: qualityColor(p.quality) }">{{ p.rms_px.toFixed(2) }}<span class="text-xs text-muted"> px</span></span>
          </div>
          <div class="text-xs text-muted">{{ new Date(p.created_at).toLocaleString() }}</div>
          <div class="flex gap-2 mt-1">
            <Button size="sm" class="flex-1" @click="openJson(p)">Open JSON</Button>
            <Button size="sm" class="flex-1" @click="openFolder(p)">Folder</Button>
            <Button size="sm" class="!text-danger !border-border hover:!border-danger" :loading="deleting === p.id" @click="pending = p">Delete</Button>
          </div>
        </div>
      </article>
    </div>

    <ConfirmDialog
      :open="!!pending" title="Delete result?" confirm-label="Delete" :busy="!!deleting"
      :body="pending ? `${pending.name} · ${pending.type} · ${pending.rms_px.toFixed(2)} px will be removed from disk. This cannot be undone.` : ''"
      @confirm="confirmDelete" @cancel="pending = null"
    />
  </div>
</template>
