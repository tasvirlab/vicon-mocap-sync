<script setup lang="ts">
defineProps<{
  label:      string
  value:      string | number
  unit?:      string
  subtitle?:  string
  color?:     string
  highlight?: boolean
  delay?:     number
}>()
</script>

<template>
  <div
    class="border border-border rounded px-4 py-3.5 rise"
    :style="{ animationDelay: `${delay ?? 0}s`, borderColor: highlight && color ? color : undefined }"
  >
    <p class="text-xs font-medium text-muted mb-2">{{ label }}</p>
    <p class="text-xl font-medium font-mono leading-none tracking-tight" :style="color ? { color } : undefined">
      {{ value }}<span v-if="unit" class="text-[13px] font-normal text-muted ml-1.5">{{ unit }}</span>
    </p>
    <p v-if="subtitle" class="text-xs text-muted mt-2">{{ subtitle }}</p>
  </div>
</template>
