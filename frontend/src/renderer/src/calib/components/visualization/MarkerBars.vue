<script setup lang="ts">
// Horizontal per-marker error bars (extrinsic). Bar color encodes the same warn / bad limits as the numbers beside it.
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  labels: string[]
  data:   number[]
  title?: string
  warnAt?: number
  badAt?:  number
  unit?:   string
}>(), { title: 'Per-marker error', warnAt: 3, badAt: 8, unit: 'px' })

const max = computed(() => Math.max(props.badAt, ...props.data, 0.0001))
const color = (v: number) => v >= props.badAt ? 'rgb(var(--color-danger))' : v >= props.warnAt ? 'rgb(var(--color-warning))' : 'rgb(var(--color-accent))'
</script>

<template>
  <section class="border border-border rounded p-6 rise">
    <div class="flex justify-between items-center flex-wrap gap-2 mb-4">
      <h2 class="text-base font-semibold">{{ title }}</h2>
      <span class="text-xs text-muted">Warn ≥ {{ warnAt }} {{ unit }} · Bad ≥ {{ badAt }} {{ unit }}</span>
    </div>
    <div class="flex flex-col gap-2">
      <div v-for="(v, i) in data" :key="i" class="grid grid-cols-[64px_1fr_72px] gap-3 items-center font-mono text-xs">
        <span class="text-muted truncate" :title="labels[i]">{{ labels[i] ?? `#${i + 1}` }}</span>
        <div class="h-2.5 bg-surface rounded-sm">
          <div class="h-full" :style="{ width: `${Math.min(100, v / max * 100)}%`, background: `color-mix(in srgb, ${color(v)} 30%, transparent)`, borderRight: `2px solid ${color(v)}` }" />
        </div>
        <span class="text-right">{{ v.toFixed(2) }} {{ unit }}</span>
      </div>
    </div>
  </section>
</template>
