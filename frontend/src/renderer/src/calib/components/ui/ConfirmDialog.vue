<script setup lang="ts">
import { onMounted, onUnmounted } from 'vue'
import Button from './Button.vue'

withDefaults(defineProps<{
  open: boolean; title: string; body?: string; confirmLabel?: string; busy?: boolean; danger?: boolean
}>(), { confirmLabel: 'Confirm', danger: true })
const emit = defineEmits<{ confirm: []; cancel: [] }>()

function onKey(e: KeyboardEvent) { if (e.key === 'Escape') emit('cancel') }
onMounted(() => window.addEventListener('keydown', onKey))
onUnmounted(() => window.removeEventListener('keydown', onKey))
</script>

<template>
  <Teleport to="body">
    <div v-if="open" class="fixed inset-0 z-50 grid place-items-center bg-black/60" @click="emit('cancel')">
      <div role="alertdialog" aria-modal="true" :aria-label="title"
        class="w-[400px] border border-muted bg-card rounded p-6 flex flex-col gap-3.5 shadow-[0_12px_32px_rgba(0,0,0,.35)]" @click.stop>
        <h2 class="text-[22px] font-bold">{{ title }}</h2>
        <p v-if="body" class="text-sm text-muted leading-normal">{{ body }}</p>
        <div class="flex gap-2 justify-end">
          <Button variant="secondary" @click="emit('cancel')">Cancel</Button>
          <Button :variant="danger ? 'danger' : 'primary'" :loading="busy" @click="emit('confirm')">{{ confirmLabel }}</Button>
        </div>
      </div>
    </div>
  </Teleport>
</template>
