<script setup lang="ts">
// Result layout shared by Extrinsic Board mode and Body mode (design: quality banner, metrics, tables, 3D, per-marker bars).
import { computed } from 'vue'
import { Download } from '@lucide/vue'
import Button          from './Button.vue'
import Card            from './Card.vue'
import MetricsCard     from './MetricsCard.vue'
import MetricsGrid     from './MetricsGrid.vue'
import QualityBadge    from './QualityBadge.vue'
import MarkerBars      from '@renderer/calib/components/visualization/MarkerBars.vue'
import ThreeScene      from '@renderer/calib/components/visualization/ThreeScene.vue'
import { qualityColor, qualityLevel } from '@renderer/calib/lib/types'

const props = withDefaults(defineProps<{
  result:      any
  subject?:    string
  caption?:    string                       // line under the quality message (mode-specific details)
  downloads:   { label: string; file: string }[]
  overlayUrl?: string                       // validation frame image, when the backend provides one
  thirdMetric: { label: string; value: string | number; subtitle?: string }
  warnAt?:     number
  badAt?:      number
  markerTitle?: string
}>(), { warnAt: 3, badAt: 8, markerTitle: 'Per-marker error' })

const emit = defineEmits<{ open: [file: string] }>()

const MSG: Record<string, string> = {
  excellent: 'Excellent — a precise camera pose.',
  good: 'Good — suitable for most work.',
  acceptable: 'Acceptable — check the overlay before relying on it.',
  poor: 'Poor — recalibrate before using this pose.',
}
const level = computed(() => qualityLevel(props.result.quality))
const pos   = computed(() => props.result.camera_position_world_mm)
const eul   = computed(() => props.result.euler_angles_deg)
const markers = computed<{ marker: string; error_px: number }[]>(() => props.result.per_marker_errors ?? [])
</script>

<template>
  <div class="flex flex-col gap-5 min-w-0">
    <div class="rise border rounded px-6 py-5 flex items-center justify-between gap-4 flex-wrap"
      :style="{ borderColor: qualityColor(result.quality), background: `color-mix(in srgb, ${qualityColor(result.quality)} 10%, transparent)` }">
      <div class="flex items-center gap-3.5">
        <QualityBadge :quality="result.quality" />
        <div>
          <div class="font-medium">{{ MSG[level] }}</div>
          <div class="font-mono text-xs text-muted mt-0.5">{{ subject || 'no subject' }}<span v-if="caption"> · {{ caption }}</span></div>
        </div>
      </div>
      <div class="flex gap-2 flex-wrap">
        <Button v-for="d in downloads" :key="d.file" @click="emit('open', d.file)">
          <template #icon><Download :size="14" :stroke-width="1.5" /></template>{{ d.label }}
        </Button>
      </div>
    </div>

    <MetricsGrid weights="1.6fr 1fr 1.2fr">
      <MetricsCard label="RMS error" :value="result.rms_px.toFixed(4)" unit="px" :size="48" :color="qualityColor(result.quality)" :delay="0.08" />
      <MetricsCard label="Camera distance" :value="pos.distance_from_origin_m.toFixed(3)" unit="m" subtitle="from Vicon origin" :size="32" :delay="0.12" />
      <MetricsCard :label="thirdMetric.label" :value="thirdMetric.value" :subtitle="thirdMetric.subtitle" :size="26" :delay="0.16" />
    </MetricsGrid>

    <div class="grid gap-5 rise" :class="eul ? 'grid-cols-2' : 'grid-cols-1'" style="animation-delay:.16s">
      <Card class="!p-[22px]" title="Position (mm)">
        <div class="border border-border rounded">
          <div v-for="(a, i) in (['x', 'y', 'z'] as const)" :key="a" :class="['flex justify-between px-3.5 py-[9px] font-mono text-sm', i < 2 && 'border-b border-border']">
            <span class="text-muted">{{ a }}</span><span>{{ pos[a].toFixed(1) }}</span>
          </div>
        </div>
      </Card>
      <Card v-if="eul" class="!p-[22px]" title="Euler angles (deg)">
        <div class="border border-border rounded">
          <div v-for="(a, i) in (['rx', 'ry', 'rz'] as const)" :key="a" :class="['flex justify-between px-3.5 py-[9px] font-mono text-sm', i < 2 && 'border-b border-border']">
            <span class="text-muted">{{ a }}</span><span>{{ eul[a].toFixed(2) }}°</span>
          </div>
        </div>
      </Card>
    </div>

    <Card v-if="result.R_matrix" class="rise" style="animation-delay:.24s">
      <template #header>
        <h2 class="text-base font-semibold">Camera in Vicon world</h2>
        <span class="text-xs text-muted">Drag to rotate · scroll to zoom</span>
      </template>
      <ThreeScene :result="result" />
    </Card>

    <MarkerBars v-if="markers.length" :labels="markers.map(m => m.marker)" :data="markers.map(m => m.error_px)"
      :title="markerTitle" :warn-at="warnAt" :bad-at="badAt" />

    <Card v-if="overlayUrl" title="Overlay validation" class="!p-[22px] rise" style="animation-delay:.4s">
      <div class="hatch border border-border rounded overflow-hidden">
        <img :src="overlayUrl" alt="Overlay validation frame" class="w-full object-contain max-h-[360px]"
          @error="($event.target as HTMLImageElement).style.display = 'none'" />
      </div>
    </Card>

    <slot />
  </div>
</template>
