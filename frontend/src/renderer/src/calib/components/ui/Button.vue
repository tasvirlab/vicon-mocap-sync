<script setup lang="ts">
import { computed } from 'vue'
import { Loader2 } from '@lucide/vue'
import { cn } from '@renderer/shared/cn'

const props = withDefaults(defineProps<{
  variant?:   'primary' | 'secondary' | 'danger' | 'ghost'
  size?:      'sm' | 'md' | 'lg'
  loading?:   boolean
  fullWidth?: boolean
  disabled?:  boolean
  class?:     string
}>(), { variant: 'secondary', size: 'md', loading: false, fullWidth: false, disabled: false })

const emit = defineEmits<{ click: [e: MouseEvent] }>()

const VARIANTS: Record<string, string> = {
  primary:   'bg-accent text-on-accent border-accent hover:bg-accent-hover active:bg-accent-pressed font-semibold',
  secondary: 'bg-transparent text-text border-muted hover:border-accent hover:bg-tint font-medium',
  danger:    'bg-danger text-white border-danger hover:opacity-90 font-semibold',
  ghost:     'bg-transparent text-muted border-transparent hover:text-text hover:bg-tint font-medium',
}
const SIZES: Record<string, string> = {
  sm: 'h-7 px-3 text-xs',
  md: 'h-9 px-4 text-[13px]',
  lg: 'h-12 px-6 text-lg !font-bold',
}

const cls = computed(() => cn(
  'inline-flex items-center justify-center gap-2 border rounded transition-colors select-none',
  'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent',
  'disabled:opacity-45 disabled:cursor-not-allowed',
  VARIANTS[props.variant], SIZES[props.size],
  props.fullWidth && 'w-full', props.class,
))
</script>

<template>
  <button type="button" :disabled="disabled || loading" :class="cls" @click="emit('click', $event)">
    <Loader2 v-if="loading" :size="16" :stroke-width="2" class="animate-spin flex-shrink-0" />
    <span v-else-if="$slots.icon" class="flex-shrink-0 inline-flex"><slot name="icon" /></span>
    <slot />
  </button>
</template>
