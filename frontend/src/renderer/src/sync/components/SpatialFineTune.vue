<script setup lang="ts">
/**
 * Spatial Fine-tune — nudge the whole projected skeleton (position + tilt)
 * to match the body, independent of the temporal Offset Fine-tune panel.
 *
 * Unlike the temporal offset (saved per-task into sync.json), this saves into
 * extrinsics.json itself — the camera pose is a session-wide constant, so
 * every other task that shares the same calibration file picks up the fix
 * automatically on its next preview or full render.
 */
import { ref, computed } from 'vue'
import { SlidersHorizontal } from '@lucide/vue'
import Button from '@renderer/sync/components/ui/Button.vue'
import {
  saveExtrinsicCorrection, ZERO_EXTRINSIC_DELTA,
  type ExtrinsicDelta,
} from '@renderer/sync/lib/api_extrinsic_correction'

const props = defineProps<{
  modelValue: ExtrinsicDelta
  taskName: string
}>()
const emit = defineEmits<{
  'update:modelValue': [ExtrinsicDelta]
  saved: []
}>()

const saving = ref(false)
const saved  = ref(false)
const error  = ref<string | null>(null)

const isZero = computed(() =>
  (Object.keys(ZERO_EXTRINSIC_DELTA) as (keyof ExtrinsicDelta)[])
    .every(k => props.modelValue[k] === 0))

// Position fields are stored/sent in metres (matches extrinsics.json's
// tvec_m and the backend's compose_delta) but shown in mm — a human nudging
// a skeleton by eye thinks in "a few mm", not "0.003".
const POS_FIELDS = [
  { key: 'dx', label: 'X (mm)' },
  { key: 'dy', label: 'Y (mm)' },
  { key: 'dz', label: 'Z / depth (mm)' },
] as const
const ROT_FIELDS = [
  { key: 'droll',  label: 'Roll (°)' },
  { key: 'dpitch', label: 'Pitch (°)' },
  { key: 'dyaw',   label: 'Yaw (°)' },
] as const

function displayValue(key: keyof ExtrinsicDelta): number {
  const v = props.modelValue[key]
  return key === 'dx' || key === 'dy' || key === 'dz' ? v * 1000 : v
}

function onInput(key: keyof ExtrinsicDelta, raw: string) {
  const n = Number(raw)
  if (Number.isNaN(n)) return
  const stored = key === 'dx' || key === 'dy' || key === 'dz' ? n / 1000 : n
  emit('update:modelValue', { ...props.modelValue, [key]: stored })
}

function reset() {
  emit('update:modelValue', { ...ZERO_EXTRINSIC_DELTA })
}

async function save() {
  if (!props.taskName || isZero.value) return
  saving.value = true
  error.value  = null
  try {
    await saveExtrinsicCorrection(props.taskName, props.modelValue)
    saved.value = true
    setTimeout(() => { saved.value = false }, 2500)
    reset()   // baked into extrinsics.json now — start the next nudge from zero
    emit('saved')
  } catch (e: any) {
    error.value = e?.response?.data?.detail ?? e?.message ?? 'Save failed'
  } finally {
    saving.value = false
  }
}
</script>

<template>
  <div class="rounded-xl border border-border p-4 space-y-3">
    <div class="flex items-center justify-between">
      <div class="flex items-center gap-1.5">
        <SlidersHorizontal :size="11" class="text-accent" />
        <p class="text-xs font-semibold text-muted ">Spatial Fine-tune</p>
      </div>
      <span class="text-xs text-muted">saves to extrinsics.json</span>
    </div>
    <p class="text-xs text-muted leading-relaxed">
      Nudges the whole skeleton to sit on the body, then locks the correction
      into this subject's calibration — every other task sharing it picks up
      the fix too, no need to redo this per task.
    </p>

    <div class="grid grid-cols-3 gap-2">
      <div v-for="f in POS_FIELDS" :key="f.key">
        <label class="text-xs text-muted block mb-0.5">{{ f.label }}</label>
        <input type="number" step="1"
          :value="displayValue(f.key)"
          @input="onInput(f.key, ($event.target as HTMLInputElement).value)"
          class="w-full bg-surface border border-border rounded-lg px-2 py-1 text-xs font-mono text-text focus:outline-none focus:border-accent" />
      </div>
    </div>
    <div class="grid grid-cols-3 gap-2">
      <div v-for="f in ROT_FIELDS" :key="f.key">
        <label class="text-xs text-muted block mb-0.5">{{ f.label }}</label>
        <input type="number" step="0.5"
          :value="displayValue(f.key)"
          @input="onInput(f.key, ($event.target as HTMLInputElement).value)"
          class="w-full bg-surface border border-border rounded-lg px-2 py-1 text-xs font-mono text-text focus:outline-none focus:border-accent" />
      </div>
    </div>

    <p v-if="error" class="text-xs text-danger">{{ error }}</p>

    <div class="flex gap-2">
      <Button variant="ghost" size="sm" :disabled="isZero || saving" @click="reset">
        Reset
      </Button>
      <Button variant="secondary" size="sm" class="flex-1"
        :loading="saving" :disabled="saving || isZero" @click="save">
        {{ saved ? '✓ Saved to extrinsics.json' : 'Save spatial correction' }}
      </Button>
    </div>
  </div>
</template>
