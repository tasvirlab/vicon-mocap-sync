<script setup lang="ts">
// One "Projects" entry, two lists: calibration results (exports/) and synchronized trials (outputs/).
import { ref } from 'vue'
import SegmentedControl from '@renderer/calib/components/ui/SegmentedControl.vue'
import CalibProjects from '@renderer/calib/pages/Projects.vue'
import SyncProjects  from '@renderer/sync/pages/Projects.vue'

const TABS = [{ value: 'sync', label: 'Trial sessions' }, { value: 'calib', label: 'Calibration results' }]
const tab = ref<string>(localStorage.getItem('projects_tab') || 'sync')
function setTab(v: string) { tab.value = v; localStorage.setItem('projects_tab', v) }
</script>

<template>
  <div>
    <div class="px-8 pt-5 max-w-[360px]">
      <SegmentedControl :model-value="tab" :options="TABS" @update:model-value="setTab" />
    </div>
    <div v-if="tab === 'calib'" class="max-w-[1200px] mx-auto px-8 pt-5 pb-14"><CalibProjects /></div>
    <SyncProjects v-else />
  </div>
</template>
