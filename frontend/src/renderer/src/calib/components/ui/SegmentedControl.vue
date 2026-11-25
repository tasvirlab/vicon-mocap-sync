<script setup lang="ts">
import { computed } from 'vue'
import { cn } from '@renderer/shared/cn'

const props = withDefaults(defineProps<{
  modelValue: string
  options: (string | { value: string; label: string })[]
  mono?: boolean
  disabled?: boolean
  size?: 'sm' | 'md'
}>(), { mono: false, disabled: false, size: 'md' })
const emit = defineEmits<{ 'update:modelValue': [v: string] }>()

const opts = computed(() => props.options.map(o => typeof o === 'string' ? { value: o, label: o } : o))
</script>

<template>
  <div class="flex border border-border rounded overflow-hidden" role="group">
    <button
      v-for="o in opts" :key="o.value" type="button" :disabled="disabled"
      :aria-pressed="modelValue === o.value"
      :class="cn(
        'flex-1 flex items-center justify-center transition-colors disabled:opacity-50',
        size === 'sm' ? 'h-7 px-3 text-xs' : 'h-[34px] px-3 text-[13px]',
        mono && 'font-mono',
        modelValue === o.value ? 'bg-accent text-on-accent font-semibold' : 'text-text font-medium hover:bg-tint',
      )"
      @click="emit('update:modelValue', o.value)"
    >{{ o.label }}</button>
  </div>
</template>
