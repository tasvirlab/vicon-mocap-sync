<script setup lang="ts">
import { computed } from 'vue'
import { cn } from '@renderer/shared/cn'

const props = withDefaults(defineProps<{ cols?: 2 | 3 | 4; weights?: string }>(), { cols: 3 })
const cls = computed(() => cn('grid gap-5',
  !props.weights && props.cols === 2 && 'grid-cols-2',
  !props.weights && props.cols === 3 && 'grid-cols-3',
  !props.weights && props.cols === 4 && 'grid-cols-4'))
</script>

<template>
  <div :class="cls" :style="weights ? { gridTemplateColumns: weights } : undefined"><slot /></div>
</template>
