<script setup lang="ts">
import { cn } from '@renderer/shared/cn'

withDefaults(defineProps<{
  label?: string; hint?: string; error?: string; unit?: string
  modelValue?: string | number; type?: string; placeholder?: string
  disabled?: boolean; step?: number | string; min?: number | string; mono?: boolean
}>(), { type: 'text', mono: true })

const emit = defineEmits<{ 'update:modelValue': [v: string | number] }>()
function onInput(e: Event, type: string) {
  const v = (e.target as HTMLInputElement).value
  emit('update:modelValue', type === 'number' ? (v === '' ? '' : Number(v)) : v)
}
</script>

<template>
  <label class="flex flex-col gap-1.5">
    <span v-if="label" class="text-xs font-medium text-muted">{{ label }}</span>
    <span :class="cn(
      'flex items-center bg-surface border rounded transition-shadow',
      error ? 'border-danger' : 'border-border focus-within:border-accent focus-within:shadow-[0_0_0_2px_rgb(var(--color-accent)/0.13)]',
      disabled && 'opacity-45',
    )">
      <input
        :value="modelValue" :type="type" :placeholder="placeholder" :disabled="disabled" :step="step" :min="min"
        :class="cn('w-full min-w-0 h-9 bg-transparent px-2.5 text-sm outline-none placeholder:text-dim', mono && 'font-mono')"
        @input="onInput($event, type)"
      />
      <span v-if="unit" class="pr-2.5 text-xs text-muted font-mono">{{ unit }}</span>
    </span>
    <span v-if="error" class="text-xs text-danger">{{ error }}</span>
    <span v-else-if="hint" class="text-xs text-muted">{{ hint }}</span>
  </label>
</template>
