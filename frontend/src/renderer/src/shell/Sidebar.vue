<script setup lang="ts">
import { Home, Camera, Crosshair, FolderOpen, SlidersHorizontal, BarChart3, Focus } from '@lucide/vue'
import AppSidebar from './AppSidebar.vue'
import { STAGES } from '@renderer/sync/lib/stages'
import type { NavGroup } from '@renderer/shared/nav'

const groups: NavGroup[] = [
  { items: [{ to: '/', label: 'Home', icon: Home }] },
  { heading: 'Calibration', items: [
    { to: '/intrinsic', label: 'Intrinsic', icon: Camera },
    { to: '/extrinsic', label: 'Extrinsic', icon: Crosshair },
  ] },
  { heading: 'Pipeline', items: STAGES.map(s => ({ to: s.to, label: s.label, num: s.num })) },
  { heading: 'Browse', items: [
    { to: '/analyse', label: 'Analyse', icon: BarChart3 },
    { to: '/projects', label: 'Projects', icon: FolderOpen },
    { to: '/settings', label: 'Settings', icon: SlidersHorizontal },
  ] },
]
</script>

<template>
  <AppSidebar brand="ViconMoCap" sub="Sync Studio" :logo="Focus" :groups="groups" />
</template>
