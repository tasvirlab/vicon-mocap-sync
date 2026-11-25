<script setup lang="ts">
// Folders (where results are stored) + the calibration and sync defaults.
import { ref, onMounted } from 'vue'
import SegmentedControl from '@renderer/calib/components/ui/SegmentedControl.vue'
import FolderSetting from '@renderer/shared/FolderSetting.vue'
import CalibSettings from '@renderer/calib/pages/Settings.vue'
import SyncSettings from '@renderer/sync/pages/Settings.vue'
import { getCalibSettings, setExportsDir } from '@renderer/calib/lib/api'
import { getSettings as getSyncSettings, setDataDir } from '@renderer/sync/lib/api'

const TABS = [
  { value: 'folders', label: 'Folders' },
  { value: 'calib', label: 'Calibration' },
  { value: 'sync', label: 'Sync' },
]
const tab = ref<string>(localStorage.getItem('settings_tab') || 'folders')
function setTab(v: string) { tab.value = v; localStorage.setItem('settings_tab', v) }

const exportsDir = ref('')
const dataDir = ref('')
const offline = ref<string[]>([])

onMounted(async () => {
  try { exportsDir.value = (await getCalibSettings()).exports_dir } catch { offline.value.push('Calibration') }
  try { dataDir.value = (await getSyncSettings()).data_dir } catch { offline.value.push('Sync') }
})

async function applyExports(path: string) { exportsDir.value = (await setExportsDir(path)).exports_dir }
async function applyData(path: string) { dataDir.value = (await setDataDir(path)).data_dir }
</script>

<template>
  <div>
    <div class="px-8 pt-5 max-w-[460px]">
      <SegmentedControl :model-value="tab" :options="TABS" @update:model-value="setTab" />
    </div>

    <div v-if="tab === 'folders'" class="max-w-[1200px] mx-auto px-8 pt-5 pb-14 flex flex-col gap-5 animate-fade-in">
      <div>
        <h1 class="text-[32px] leading-[1.1] font-bold tracking-tight">Folders</h1>
        <p class="text-[15px] text-muted mt-1.5">Choose where results are stored. Existing files are not moved.</p>
      </div>
      <p v-if="offline.length" class="text-[13px] text-warning">Backend offline: {{ offline.join(', ') }}. Start the app again to change these folders.</p>
      <FolderSetting title="Calibration results" :current="exportsDir" :apply="applyExports"
        description="Intrinsic and extrinsic results (parameters, reports, check images) are saved here, one subfolder per subject." />
      <FolderSetting title="Sync data" :current="dataDir" :apply="applyData"
        description="Trial outputs (synced CSV, overlay and RGB video, REBA, 2D pose, dataset exports) are saved in an outputs folder inside this folder; temporary files in temp." />
    </div>
    <div v-else-if="tab === 'calib'" class="max-w-[1200px] mx-auto px-8 pt-5 pb-14"><CalibSettings /></div>
    <SyncSettings v-else />
  </div>
</template>
