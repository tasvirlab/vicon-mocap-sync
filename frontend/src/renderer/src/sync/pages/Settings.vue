<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { Settings as SettingsIcon, FolderOpen, AlertCircle, CheckCircle } from '@lucide/vue'
import Button from '@renderer/sync/components/ui/Button.vue'
import PageHeader from '@renderer/sync/components/ui/PageHeader.vue'
import { getSettings, setDataDir, browseNativeFolder } from '@renderer/sync/lib/api'
import type { AppSettings } from '@renderer/sync/lib/api'

const current = ref<AppSettings | null>(null)
const loadError = ref<string | null>(null)

const browsing = ref(false)
const saveError = ref<string | null>(null)
const saved = ref(false)

async function loadCurrent() {
  loadError.value = null
  try {
    current.value = await getSettings()
  } catch (e: unknown) {
    loadError.value = e instanceof Error ? e.message : String(e)
  }
}
onMounted(loadCurrent)

async function chooseFolder() {
  browsing.value = true
  saveError.value = null
  saved.value = false
  try {
    const path = await browseNativeFolder()
    if (!path) return   // user cancelled the dialog
    current.value = await setDataDir(path)
    saved.value = true
  } catch (e: unknown) {
    saveError.value = e instanceof Error ? e.message : String(e)
  } finally {
    browsing.value = false
  }
}
</script>

<template>
  <div class="p-6 space-y-5 animate-fade-in max-w-2xl">

    <PageHeader title="Settings" :icon="SettingsIcon" color="#94A3B8"
      subtitle="Choose where this app stores its data — lets the app be shared and used on any machine." />

    <div v-if="loadError" class="flex items-start gap-2 bg-danger/10 border border-danger/30 rounded-xl p-3">
      <AlertCircle :size="14" class="text-danger flex-shrink-0 mt-0.5" />
      <p class="text-xs text-danger">{{ loadError }}</p>
    </div>

    <div class="rounded-xl border border-border p-4 space-y-3">
      <p class="text-xs font-semibold text-muted ">Data Folder</p>
      <p class="text-xs text-muted leading-relaxed">
        All processed trials (<code class="text-text">outputs/</code>) and upload staging
        (<code class="text-text">temp/</code>) are written here. Change it to point at another
        drive, a shared folder, or anywhere with enough space — new work goes to the new
        location immediately; anything already processed in the old location stays there
        untouched.
      </p>

      <div class="bg-surface rounded-lg border border-border p-3 font-mono text-xs text-text break-all">
        {{ current?.data_dir ?? '…' }}
      </div>

      <Button variant="secondary" size="sm" :loading="browsing" @click="chooseFolder">
        <template #icon><FolderOpen :size="13" /></template>
        {{ browsing ? 'Waiting for folder selection…' : 'Browse for folder…' }}
      </Button>
      <p v-if="browsing" class="text-xs text-muted">
        The Windows folder picker opened in a separate window — check your taskbar if you don't see it.
      </p>

      <p v-if="saveError" class="flex items-start gap-1.5 text-xs text-danger">
        <AlertCircle :size="12" class="flex-shrink-0 mt-0.5" /> {{ saveError }}
      </p>
      <p v-if="saved" class="flex items-center gap-1.5 text-xs text-success">
        <CheckCircle :size="12" /> Saved — new jobs will write here from now on.
      </p>
    </div>
  </div>
</template>
