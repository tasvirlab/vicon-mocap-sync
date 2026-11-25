<script setup lang="ts">
import { ref, computed } from 'vue'
import { Upload, X } from '@lucide/vue'
import { cn } from '@renderer/shared/cn'

const props = defineProps<{
  label:        string
  description?: string
  accept?:      string
  file?:        File | null
  disabled?:    boolean
}>()

const emit = defineEmits<{ 'update:file': [File | null] }>()

const isDragging = ref(false)
const inputRef   = ref<HTMLInputElement | null>(null)

const classes = computed(() => cn(
  'relative flex flex-col items-center justify-center gap-2 rounded border transition-colors cursor-pointer px-4 py-5 min-h-[80px]',
  props.file
    ? 'border-solid border-accent'
    : isDragging.value
      ? 'drop-zone-active border-dashed'
      : 'border-dashed border-dim hover:border-muted',
  props.disabled ? 'opacity-45 cursor-not-allowed pointer-events-none' : '',
))

function onDrop(e: DragEvent) {
  isDragging.value = false
  if (props.disabled) return
  const file = e.dataTransfer?.files[0]
  if (file) emit('update:file', file)
}

function onFileChange(e: Event) {
  const file = (e.target as HTMLInputElement).files?.[0]
  if (file) emit('update:file', file)
}

function onClick() {
  if (!props.disabled) inputRef.value?.click()
}

function clear(e: MouseEvent) {
  e.stopPropagation()
  emit('update:file', null)
  if (inputRef.value) inputRef.value.value = ''
}
</script>

<template>
  <div
    :class="classes"
    @click="onClick"
    @dragover.prevent="isDragging = !disabled"
    @dragleave="isDragging = false"
    @drop.prevent="onDrop"
  >
    <input ref="inputRef" type="file" :accept="accept" class="sr-only" @change="onFileChange" />

    <template v-if="file">
      <div class="flex items-center gap-2 w-full">
        <div class="flex-1 min-w-0">
          <p class="text-xs font-mono font-medium truncate">{{ file.name }}</p>
          <p class="text-xs text-muted mt-0.5">{{ (file.size / 1024 / 1024).toFixed(1) }} MB</p>
        </div>
        <button class="p-1 rounded-sm text-muted hover:text-danger hover:bg-danger/10 transition-colors" aria-label="Remove file" @click="clear">
          <X :size="14" :stroke-width="1.5" />
        </button>
      </div>
    </template>

    <template v-else>
      <Upload :size="20" :stroke-width="1.5" :class="isDragging ? 'text-accent' : 'text-muted'" />
      <div class="text-center">
        <p :class="['text-sm font-medium', isDragging && 'text-accent']">{{ isDragging ? 'Release to load' : label }}</p>
        <p v-if="description" class="text-xs text-muted mt-0.5" v-html="description" />
      </div>
    </template>
  </div>
</template>
