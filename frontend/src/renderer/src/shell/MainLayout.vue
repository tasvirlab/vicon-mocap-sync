<script setup lang="ts">
import { RouterView, useRoute } from 'vue-router'
import Sidebar from './Sidebar.vue'
import TopBar from './TopBar.vue'

const route = useRoute()
</script>

<template>
  <div class="flex flex-col h-screen w-screen overflow-hidden bg-bg min-w-[1180px]">
    <TopBar app-name="ViconMoCap Sync Studio" />
    <div class="flex flex-1 min-h-0">
      <Sidebar />
      <main class="flex-1 min-w-0 overflow-y-auto overflow-x-hidden bg-bg">
        <!-- Calibration pages + Home use the centered 1200 px column; pipeline pages fill the width (they carry their own padding). -->
        <div v-if="route.meta.contained" class="max-w-[1200px] mx-auto px-8 pt-7 pb-14">
          <RouterView />
        </div>
        <RouterView v-else />
      </main>
    </div>
  </div>
</template>
