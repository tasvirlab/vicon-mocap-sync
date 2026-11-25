<script setup lang="ts">
import { computed } from 'vue'
import { qualityLevel } from '@renderer/calib/lib/types'

const props = defineProps<{ quality: string }>()
const lvl = computed(() => qualityLevel(props.quality))
const CLS = {
  excellent:  'border-success text-success',
  good:       'border-accent text-accent',
  acceptable: 'border-warning text-warning',
  poor:       'border-danger text-danger',
}
const DOT = { excellent: 'bg-success', good: 'bg-accent', acceptable: 'bg-warning', poor: 'bg-danger' }
const LABEL = { excellent: 'Excellent', good: 'Good', acceptable: 'Acceptable', poor: 'Poor' }
</script>

<template>
  <span :class="['inline-flex items-center gap-1.5 h-[26px] px-2.5 border rounded-sm text-xs font-semibold', CLS[lvl]]">
    <i :class="['w-1.5 h-1.5', DOT[lvl]]" />{{ LABEL[lvl] }}
  </span>
</template>
