<script setup lang="ts">
import { computed } from 'vue'
import { cn } from '@renderer/shared/cn'

const props = withDefaults(defineProps<{
  variant?:  'primary' | 'secondary' | 'ghost' | 'danger'
  size?:     'sm' | 'md' | 'lg'
  disabled?: boolean
  loading?:  boolean
  fullWidth?: boolean
}>(), {
  variant: 'secondary', size: 'md',
  disabled: false, loading: false, fullWidth: false,
})

const classes = computed(() => cn(
  'inline-flex items-center justify-center gap-2 border rounded transition-colors select-none',
  'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent',
  props.size === 'sm' ? 'h-7 px-3 text-xs' :
  props.size === 'lg' ? 'h-12 px-6 text-lg font-bold' : 'h-9 px-4 text-[13px]',
  props.variant === 'primary'   ? 'bg-accent text-on-accent border-accent hover:bg-accent-hover active:bg-accent-pressed font-semibold' :
  props.variant === 'danger'    ? 'bg-danger text-white border-danger hover:opacity-90 font-semibold' :
  props.variant === 'ghost'     ? 'bg-transparent text-muted border-transparent hover:text-text hover:bg-tint font-medium' :
                                  'bg-transparent text-text border-muted hover:border-accent hover:bg-tint font-medium',
  props.fullWidth ? 'w-full' : '',
  props.disabled || props.loading ? 'opacity-45 cursor-not-allowed pointer-events-none' : 'cursor-pointer',
))
</script>

<template>
  <button type="button" :class="classes" :disabled="disabled || loading">
    <svg v-if="loading" class="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
      <path d="M21 12a9 9 0 1 1-6.2-8.55" />
    </svg>
    <slot v-else name="icon" />
    <slot />
  </button>
</template>
