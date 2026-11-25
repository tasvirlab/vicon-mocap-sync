<script setup lang="ts">
import { Info } from '@lucide/vue'

defineProps<{ text: string }>()
</script>

<template>
  <span class="relative inline-flex group align-middle">
    <Info :size="13" :stroke-width="1.5" class="text-muted hover:text-accent cursor-help" />
    <span
      role="tooltip"
      class="invisible group-hover:visible absolute left-1/2 -translate-x-1/2 bottom-full mb-1.5 z-50
             w-60 bg-card border border-muted rounded px-3 py-2 shadow-[0_12px_32px_rgba(0,0,0,.35)]
             text-xs text-text font-normal normal-case tracking-normal leading-normal text-left">
      {{ text }}
    </span>
  </span>
</template>
