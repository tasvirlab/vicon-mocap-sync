<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { FolderOpen, Trash2, RefreshCw, ExternalLink, Film, Activity, User, Layers } from '@lucide/vue'
import Button from '@renderer/sync/components/ui/Button.vue'
import PageHeader from '@renderer/sync/components/ui/PageHeader.vue'
import { listProjects, deleteProject, openProjectFolder } from '@renderer/sync/lib/api'
import { syncMethodLabel, syncMethodColor } from '@renderer/sync/lib/types'
import type { ProjectEntry } from '@renderer/sync/lib/types'

const router   = useRouter()
const projects = ref<ProjectEntry[]>([])
const loading  = ref(false)
const error    = ref<string | null>(null)
const deleteTarget = ref<string | null>(null)

const UNASSIGNED = '(unassigned)'

// Subject -> Task -> trials, grouping the flat list for the new
// outputs/SUB01/<task>/T01/ hierarchy. Entries with no parsed subject_id
// (legacy folders moved to outputs/_UNASSIGNED/) fall into one bucket.
const grouped = computed(() => {
  const bySubject = new Map<string, Map<string, ProjectEntry[]>>()
  for (const p of projects.value) {
    const subject = p.subject_id ?? UNASSIGNED
    const taskKey  = p.task ?? p.name
    if (!bySubject.has(subject)) bySubject.set(subject, new Map())
    const byTask = bySubject.get(subject)!
    if (!byTask.has(taskKey)) byTask.set(taskKey, [])
    byTask.get(taskKey)!.push(p)
  }
  // Sort: real subjects (SUB01, SUB02, ...) first, unassigned last.
  return [...bySubject.entries()].sort(([a], [b]) => {
    if (a === UNASSIGNED) return 1
    if (b === UNASSIGNED) return -1
    return a.localeCompare(b)
  })
})

async function load() {
  loading.value = true
  error.value   = null
  try {
    projects.value = await listProjects()
  } catch (e: unknown) {
    error.value = `Failed to load sessions: ${e instanceof Error ? e.message : String(e)}`
  } finally {
    loading.value = false
  }
}

function remove(id: string) {
  deleteTarget.value = id
}

async function confirmDelete() {
  if (!deleteTarget.value) return
  await deleteProject(deleteTarget.value)
  deleteTarget.value = null
  await load()
}

async function openFolder(id: string) {
  await openProjectFolder(id)
}

function runOverlay(id: string) {
  router.push('/overlay')
}

onMounted(load)
</script>

<template>
  <div class="p-6 space-y-5 animate-fade-in">

    <!-- Header -->
    <PageHeader title="Saved Sessions" :icon="FolderOpen" color="#8B5CF6"
      subtitle="Browse processed trials grouped by subject → task → trial (outputs/SUB<NN>/<task>/T<NN>/).">
      <template #actions>
        <Button variant="ghost" size="sm" :loading="loading" @click="load">
          <template #icon><RefreshCw :size="13" /></template>
          Refresh
        </Button>
      </template>
    </PageHeader>

    <div v-if="error" class="bg-danger/10 border border-danger/30 rounded-xl p-3 text-xs text-danger">
      {{ error }}
    </div>

    <!-- Loading skeletons -->
    <div v-if="loading && !projects.length" class="space-y-2">
      <div v-for="i in 4" :key="i" class="h-20 rounded-xl shimmer" />
    </div>

    <!-- Empty state -->
    <div v-else-if="!projects.length"
      class="flex flex-col items-center justify-center py-20 text-muted space-y-3">
      <FolderOpen :size="40" class="opacity-20" />
      <div class="text-center">
        <p class="text-sm font-medium">No sessions yet</p>
        <p class="text-xs mt-1">Run a synchronisation to create the first session.</p>
      </div>
      <Button size="sm" @click="router.push('/sync')">
        <template #icon><Activity :size="12" /></template>
        Go to Sync
      </Button>
    </div>

    <!-- Grouped: Subject > Task > Trial -->
    <div v-else class="space-y-3">
      <details v-for="[subject, byTask] in grouped" :key="subject" open
        class="rounded-xl border border-border overflow-hidden">
        <summary class="flex items-center gap-2 p-3 cursor-pointer select-none hover:bg-surface/50">
          <User :size="13" class="text-accent3 flex-shrink-0" />
          <span class="text-sm font-semibold text-text">{{ subject }}</span>
          <span class="text-xs text-muted">
            {{ [...byTask.values()].reduce((n, v) => n + v.length, 0) }} trial(s) ·
            {{ byTask.size }} task(s)
          </span>
        </summary>

        <div class="border-t border-border divide-y divide-border">
          <div v-for="[taskKey, trials] in byTask" :key="taskKey" class="p-3 space-y-2">
            <div class="flex items-center gap-2">
              <Layers :size="11" class="text-muted flex-shrink-0" />
              <span class="text-xs font-medium text-text">{{ taskKey }}</span>
              <span class="text-xs text-muted">{{ trials.length }} trial(s)</span>
            </div>

            <div class="space-y-1.5 pl-4">
              <div v-for="p in trials" :key="p.id"
                class="flex items-center gap-3 bg-surface rounded-lg p-2.5 hover:border-accent/30 border border-transparent transition-colors">
                <div class="w-2 h-2 rounded-full flex-shrink-0"
                  :style="{ background: syncMethodColor(p.sync_method ?? '') }" />

                <div class="flex-1 min-w-0">
                  <div class="flex items-center gap-2">
                    <span class="text-xs font-medium text-text">{{ p.trial ?? p.id }}</span>
                    <span v-if="p.has_overlay"
                      class="flex items-center gap-1 text-xs text-success font-medium bg-success/10 border border-success/30 px-1.5 py-0.5 rounded-full">
                      <Film :size="8" /> overlay
                    </span>
                    <span v-if="p.has_synced_csv"
                      class="text-xs text-accent bg-accent/10 border border-accent/30 px-1.5 py-0.5 rounded-full">
                      synced.csv
                    </span>
                  </div>
                  <div class="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs text-muted mt-0.5">
                    <span>{{ p.created_at.slice(0, 16).replace('T', ' ') }}</span>
                    <span>·</span>
                    <span class="font-mono">{{ (p.sync_offset_s ?? 0) > 0 ? '+' : '' }}{{ (p.sync_offset_s ?? 0).toFixed(4) }}s</span>
                    <span>·</span>
                    <span :style="{ color: syncMethodColor(p.sync_method ?? '') }">
                      {{ syncMethodLabel(p.sync_method ?? 'unknown') }}
                    </span>
                  </div>
                  <p class="text-xs text-muted font-mono truncate mt-1">{{ p.output_dir }}</p>
                </div>

                <div class="flex items-center gap-1 flex-shrink-0">
                  <Button size="sm" variant="ghost" title="Open output folder" @click="openFolder(p.id)">
                    <template #icon><ExternalLink :size="12" /></template>
                  </Button>
                  <Button v-if="!p.has_overlay" size="sm" variant="ghost" title="Generate overlay"
                    @click="runOverlay(p.id)">
                    <template #icon><Film :size="12" /></template>
                  </Button>
                  <Button size="sm" variant="danger" title="Delete session" @click="remove(p.id)">
                    <template #icon><Trash2 :size="12" /></template>
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </details>
    </div>

    <!-- Summary -->
    <div v-if="projects.length > 0"
      class="text-xs text-muted text-center pt-2">
      {{ projects.length }} trial{{ projects.length === 1 ? '' : 's' }} ·
      {{ grouped.length }} subject{{ grouped.length === 1 ? '' : 's' }} ·
      {{ projects.filter(p => p.has_overlay).length }} with overlay ·
      {{ projects.filter(p => p.has_synced_csv).length }} with synced CSV
    </div>

    <!-- Delete confirmation modal -->
    <Teleport to="body">
      <div v-if="deleteTarget"
        class="fixed inset-0 bg-black/60 flex items-center justify-center z-50"
        @click.self="deleteTarget = null">
        <div class="bg-card border border-border rounded-2xl p-5 w-80 space-y-4 shadow-2xl">
          <p class="text-sm font-semibold text-text">Delete session?</p>
          <p class="text-xs text-muted leading-relaxed">
            <span class="font-mono text-text">{{ deleteTarget }}</span> and all its output files will be permanently removed.
          </p>
          <div class="flex gap-2 justify-end">
            <Button size="sm" variant="ghost" @click="deleteTarget = null">Cancel</Button>
            <Button size="sm" variant="danger" @click="confirmDelete">
              <template #icon><Trash2 :size="12" /></template>
              Delete
            </Button>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>
