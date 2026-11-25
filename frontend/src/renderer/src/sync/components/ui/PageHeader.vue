<script setup lang="ts">
import type { Component } from 'vue'

defineProps<{
  title:    string
  subtitle: string
  icon:     Component
  color:    string      // legacy stage identity color; the design system uses the accent only
  step?:    number      // pipeline step number (omit for non-pipeline pages)
}>()
</script>

<template>
  <div class="flex items-end justify-between gap-6 flex-shrink-0">
    <div class="flex items-center gap-4 min-w-0">
      <span v-if="step" class="w-10 h-10 flex-shrink-0 grid place-items-center border border-accent rounded text-accent font-mono text-base">{{ String(step).padStart(2, '0') }}</span>
      <span v-else class="w-10 h-10 flex-shrink-0 grid place-items-center border border-accent rounded text-accent"><component :is="icon" :size="18" :stroke-width="1.5" /></span>
      <div class="min-w-0">
        <h1 class="text-[32px] leading-[1.1] font-bold tracking-tight">{{ title }}</h1>
        <p class="text-[15px] text-muted mt-1.5">{{ subtitle }}</p>
      </div>
    </div>
    <div class="flex items-center gap-2">
      <slot name="actions" />
    </div>
  </div>
</template>
