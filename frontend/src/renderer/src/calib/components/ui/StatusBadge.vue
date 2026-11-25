<script setup lang="ts">
import { cn } from '@renderer/shared/cn'

const props = defineProps<{ status: string; class?: string }>()
const STYLES: Record<string, string> = {
  queued:    'border-dim text-muted',
  running:   'border-accent text-accent',
  completed: 'border-success text-success',
  failed:    'border-danger text-danger',
}
const DOT: Record<string, string> = { running: 'bg-accent animate-pulse', completed: 'bg-success', failed: 'bg-danger' }
</script>

<template>
  <span :class="cn('inline-flex items-center gap-1.5 h-6 px-2 border rounded-sm text-xs font-semibold capitalize', STYLES[status] ?? 'border-border text-muted', props.class)">
    <i :class="cn('w-1.5 h-1.5', DOT[status] ?? 'bg-muted')" />{{ status }}
  </span>
</template>
