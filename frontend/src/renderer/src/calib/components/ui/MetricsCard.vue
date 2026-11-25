<script setup lang="ts">
withDefaults(defineProps<{
  label: string; value: string | number; unit?: string; subtitle?: string
  color?: string; size?: number; delay?: number
}>(), { size: 32, delay: 0 })
</script>

<template>
  <div class="border border-border rounded px-[22px] py-5 rise" :style="{ animationDelay: `${delay}s` }">
    <div class="text-xs font-medium text-muted">{{ label }}</div>
    <div class="mt-3.5 flex items-baseline gap-2 flex-wrap">
      <span class="font-mono font-medium leading-none tracking-tight" :style="{ fontSize: `${size}px`, color: color || undefined }">{{ value }}</span>
      <span v-if="unit" class="text-[13px] text-muted font-mono">{{ unit }}</span>
    </div>
    <div v-if="subtitle" class="text-xs text-muted mt-2">{{ subtitle }}</div>
  </div>
</template>
