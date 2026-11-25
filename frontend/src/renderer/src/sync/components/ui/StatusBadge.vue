<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ status: string }>()

const cfg = computed(() => {
  switch (props.status) {
    case 'running':   return { label: 'Running', cls: 'border-accent text-accent' }
    case 'completed': return { label: 'Done',    cls: 'border-success text-success' }
    case 'failed':    return { label: 'Failed',  cls: 'border-danger text-danger' }
    case 'queued':    return { label: 'Queued',  cls: 'border-warning text-warning' }
    default:          return { label: props.status, cls: 'border-border text-muted' }
  }
})
</script>

<template>
  <span :class="`inline-flex items-center gap-1.5 h-6 px-2 text-xs font-semibold rounded-sm border ${cfg.cls}`">
    <i :class="['w-1.5 h-1.5 bg-current', status === 'running' && 'animate-pulse']" />
    {{ cfg.label }}
  </span>
</template>
