<script setup lang="ts">
// Per-frame error bars with dashed warn / bad threshold lines (DOM, theme-token colors, no chart library).
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  data:    number[]
  title?:  string
  warnAt?: number
  badAt?:  number
  unit?:   string
  firstLabel?: string
  lastLabel?:  string
}>(), { title: 'Per-frame RMS', warnAt: 0.5, badAt: 1.0, unit: 'px' })

const max = computed(() => Math.max(props.badAt * 1.15, ...props.data, 0.0001))
const pct = (v: number) => Math.min(100, (v / max.value) * 100)
const color = (v: number) => v >= props.badAt ? 'rgb(var(--color-danger))' : v >= props.warnAt ? 'rgb(var(--color-warning))' : 'rgb(var(--color-accent))'
const fmt = (v: number) => (v >= 10 ? v.toFixed(0) : v.toFixed(2))
</script>

<template>
  <section class="border border-border rounded p-[22px] rise">
    <div class="flex justify-between items-center flex-wrap gap-2 mb-4">
      <h2 class="text-base font-semibold">{{ title }}</h2>
      <div class="flex gap-4 text-xs text-muted">
        <span class="flex items-center gap-1.5"><i class="w-3.5 border-t-2 border-dashed border-warning" />Warn {{ warnAt }} {{ unit }}</span>
        <span class="flex items-center gap-1.5"><i class="w-3.5 border-t-2 border-dashed border-danger" />Bad {{ badAt }} {{ unit }}</span>
      </div>
    </div>

    <div class="flex gap-2.5">
      <div class="flex flex-col justify-between items-end font-mono text-[11px] text-muted h-[180px]">
        <span>{{ fmt(max) }}</span><span>{{ fmt(max / 2) }}</span><span>0</span>
      </div>
      <div class="relative flex-1 h-[180px] border-l border-b border-border flex items-end gap-[3px] px-1"
           role="img" :aria-label="`${title}: ${data.length} values, maximum ${fmt(Math.max(...data, 0))} ${unit}`">
        <div class="absolute inset-x-0 top-1/2 border-t border-border" />
        <div class="absolute inset-x-0 top-0 border-t border-border" />
        <div v-for="(v, i) in data" :key="i" class="flex-1 min-w-[2px] relative z-[1]"
             :style="{ height: `${pct(v)}%`, background: `color-mix(in srgb, ${color(v)} 28%, transparent)`, borderTop: `2px solid ${color(v)}` }"
             :title="`#${i + 1}: ${v.toFixed(4)} ${unit}`" />
        <div class="absolute inset-x-0 border-t-2 border-dashed border-warning opacity-90 z-[2]" :style="{ bottom: `${pct(warnAt)}%` }" />
        <div class="absolute inset-x-0 border-t-2 border-dashed border-danger opacity-90 z-[2]" :style="{ bottom: `${pct(badAt)}%` }" />
      </div>
    </div>
    <div class="flex justify-between mt-1.5 ml-9 font-mono text-[11px] text-muted">
      <span>{{ firstLabel ?? 'frame 1' }}</span><span>{{ lastLabel ?? `frame ${data.length}` }}</span>
    </div>
  </section>
</template>
