<script setup lang="ts">
import { computed } from 'vue'
import { Check, Loader2 } from '@lucide/vue'
import { cn } from '@renderer/shared/cn'
import type { JobStatus } from '@renderer/calib/lib/types'

const props = defineProps<{ jobStatus: JobStatus | null }>()

const STEPS_INTRINSIC = [
  { label: 'Building board',     threshold: 5   },
  { label: 'Extracting frames',  threshold: 10  },
  { label: 'Detecting corners',  threshold: 58  },
  { label: 'Calibrating',        threshold: 75  },
  { label: 'Generating outputs', threshold: 90  },
  { label: 'Complete',           threshold: 100 },
]

const STEPS_EXTRINSIC = [
  { label: 'Loading intrinsics',       threshold: 10  },
  { label: 'Extracting frame',         threshold: 20  },
  { label: 'Detecting ChArUco',        threshold: 35  },
  { label: 'Loading Vicon',            threshold: 45  },
  { label: 'Building correspondences', threshold: 55  },
  { label: 'Solving PnP',              threshold: 70  },
  { label: 'Exporting',                threshold: 88  },
  { label: 'Complete',                 threshold: 100 },
]

const progress = computed(() => props.jobStatus?.progress ?? 0)
const steps    = computed(() =>
  props.jobStatus?.step?.toLowerCase().includes('vicon') ? STEPS_EXTRINSIC : STEPS_INTRINSIC
)

function stepDone(t: number)   { return progress.value >= t }
function stepActive(t: number) {
  return !stepDone(t) && progress.value >= t - 20 && props.jobStatus?.status === 'running'
}
function stateLabel(t: number) { return stepDone(t) ? 'done' : stepActive(t) ? 'running' : 'pending' }
</script>

<template>
  <div v-if="jobStatus" class="flex flex-col gap-6">
    <div>
      <div class="flex justify-between text-[13px] mb-2">
        <span class="text-muted truncate max-w-[70%]">{{ jobStatus.step || 'Progress' }}</span>
        <span class="font-mono">{{ progress }}%</span>
      </div>
      <div class="h-1.5 bg-surface border border-border rounded-sm overflow-hidden" role="progressbar" :aria-valuenow="progress" aria-valuemin="0" aria-valuemax="100">
        <div class="h-full bg-accent transition-[width] duration-[400ms] ease-out" :style="{ width: `${progress}%` }" />
      </div>
    </div>

    <ol class="flex flex-col border border-border rounded overflow-hidden">
      <li v-for="(step, i) in steps" :key="step.label"
        :class="cn('flex items-center gap-3.5 px-4 py-3', i < steps.length - 1 && 'border-b border-border', stepActive(step.threshold) && 'bg-tint')">
        <span :class="cn(
          'w-6 h-6 flex-shrink-0 grid place-items-center border rounded-sm font-mono text-xs',
          stepDone(step.threshold) ? 'bg-accent border-accent text-on-accent' :
          stepActive(step.threshold) ? 'border-accent text-accent' : 'border-border text-muted'
        )">
          <Check v-if="stepDone(step.threshold)" :size="14" :stroke-width="2" />
          <Loader2 v-else-if="stepActive(step.threshold)" :size="14" class="animate-spin" />
          <template v-else>{{ i + 1 }}</template>
        </span>
        <span :class="cn('flex-1 text-sm font-medium', stepDone(step.threshold) || stepActive(step.threshold) ? 'text-text' : 'text-muted')">{{ step.label }}</span>
        <span class="text-xs text-muted font-mono">{{ stateLabel(step.threshold) }}</span>
      </li>
    </ol>
  </div>
</template>
