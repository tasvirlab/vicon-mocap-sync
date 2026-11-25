<script setup lang="ts">
import { AlertCircle, AlertTriangle, Info } from '@lucide/vue'
import { cn } from '@renderer/shared/cn'

withDefaults(defineProps<{ tone?: 'info' | 'warning' | 'error' }>(), { tone: 'info' })
const TONE = { info: 'border-accent bg-accent/10', warning: 'border-warning bg-warning/10', error: 'border-danger bg-danger/10' }
const ICON_TONE = { info: 'text-accent', warning: 'text-warning', error: 'text-danger' }
const ICON = { info: Info, warning: AlertTriangle, error: AlertCircle }
</script>

<template>
  <div :class="cn('flex gap-2.5 px-3 py-2.5 border rounded text-[13px]', TONE[tone])" :role="tone === 'error' ? 'alert' : 'status'">
    <component :is="ICON[tone]" :size="16" :stroke-width="1.5" :class="cn('flex-shrink-0 mt-px', ICON_TONE[tone])" />
    <div class="min-w-0"><slot /></div>
  </div>
</template>
