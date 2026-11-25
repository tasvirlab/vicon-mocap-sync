<script setup lang="ts">
import { computed } from 'vue'
import type { JobStatus } from '@renderer/sync/lib/types'

const props = defineProps<{ jobStatus: JobStatus | null }>()

const progress = computed(() => props.jobStatus?.progress ?? 0)
const step     = computed(() => props.jobStatus?.step ?? 'Waiting…')
</script>

<template>
  <div class="space-y-3">
    <div class="flex items-center justify-between text-[13px]">
      <span class="text-muted">{{ step }}</span>
      <span class="font-mono">{{ progress }}%</span>
    </div>

    <div class="h-1.5 w-full bg-surface border border-border rounded-sm overflow-hidden" role="progressbar"
         :aria-valuenow="progress" aria-valuemin="0" aria-valuemax="100">
      <div class="h-full bg-accent transition-[width] duration-[400ms] ease-out" :style="{ width: `${progress}%` }" />
    </div>

    <p v-if="jobStatus?.message" class="text-xs text-muted leading-normal">{{ jobStatus.message }}</p>
  </div>
</template>
