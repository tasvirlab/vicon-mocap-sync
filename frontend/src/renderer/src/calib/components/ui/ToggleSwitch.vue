<script setup lang="ts">
import { cn } from '@renderer/shared/cn'

defineProps<{ modelValue: boolean; label?: string; disabled?: boolean }>()
const emit = defineEmits<{ 'update:modelValue': [v: boolean] }>()
</script>

<template>
  <button
    type="button" role="switch" :aria-checked="modelValue" :disabled="disabled"
    class="inline-flex items-center gap-2.5 text-sm disabled:opacity-45 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent rounded"
    @click="emit('update:modelValue', !modelValue)"
  >
    <span :class="cn('relative w-9 h-5 rounded-full border transition-colors', modelValue ? 'bg-accent border-accent' : 'border-muted')">
      <i :class="cn('absolute top-0.5 w-3.5 h-3.5 rounded-full transition-all', modelValue ? 'left-[18px] bg-on-accent' : 'left-0.5 bg-muted')" />
    </span>
    <span v-if="label" class="text-muted">{{ label }}</span>
  </button>
</template>
