<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ quality: string; confidence?: number }>()

// Text always accompanies the color (never color-only).
const tone = computed(() => {
  const q = (props.quality || '').toUpperCase()
  if (q.startsWith('EXCELLENT') || q.startsWith('HIGH')) return 'text-success border-success'
  if (q.startsWith('GOOD') || q.startsWith('MEDIUM'))    return 'text-accent border-accent'
  if (q.startsWith('ACCEPT'))                            return 'text-warning border-warning'
  return 'text-danger border-danger'
})
</script>

<template>
  <span :class="['inline-flex items-center gap-1.5 h-[26px] px-2.5 border rounded-sm text-xs font-semibold', tone]">
    <i class="w-1.5 h-1.5 bg-current" />
    {{ quality }}
    <span v-if="confidence !== undefined" class="font-mono font-normal opacity-80">({{ (confidence * 100).toFixed(0) }}%)</span>
  </span>
</template>
