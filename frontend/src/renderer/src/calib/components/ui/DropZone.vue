<script setup lang="ts">
import { ref } from 'vue'
import { Upload, FileVideo, FileText, File as FileIcon, X } from '@lucide/vue'
import { cn } from '@renderer/shared/cn'

const props = withDefaults(defineProps<{
  label:       string
  description: string
  accept:      string
  file:        File | null
  disabled?:   boolean
  compact?:    boolean
}>(), { disabled: false, compact: false })

const emit = defineEmits<{ 'update:file': [f: File | null] }>()

const isDragging = ref(false)
const inputRef   = ref<HTMLInputElement | null>(null)

function fileKind(file: File) {
  if (file.type.startsWith('video/') || /\.(mp4|mov|avi|mkv)$/i.test(file.name)) return 'video'
  if (/\.(csv|txt|json)$/i.test(file.name)) return 'text'
  return 'file'
}

function formatSize(bytes: number) {
  if (bytes < 1024)    return `${bytes} B`
  if (bytes < 1024**2) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024**2).toFixed(1)} MB`
}

function onDrop(e: DragEvent) {
  e.preventDefault()
  isDragging.value = false
  if (props.disabled) return
  const f = e.dataTransfer?.files[0]
  if (f) emit('update:file', f)
}
function onDragOver(e: DragEvent) { e.preventDefault(); if (!props.disabled) isDragging.value = true }
function browse() { if (!props.disabled) inputRef.value?.click() }
function onClick() { if (!props.file) browse() }
function onChange(e: Event) {
  const f = (e.target as HTMLInputElement).files?.[0]
  if (f) emit('update:file', f)
  ;(e.target as HTMLInputElement).value = ''
}
</script>

<template>
  <!-- Compact: a labelled file row (used for multi-file forms) -->
  <div
    v-if="compact"
    :class="cn(
      'flex items-center justify-between gap-2.5 px-3 py-2.5 border rounded transition-colors',
      file ? 'border-accent' : isDragging ? 'drop-zone-active border-dashed' : 'border-border border-dashed hover:border-muted',
      disabled && 'opacity-45 cursor-not-allowed',
    )"
    @drop="onDrop" @dragover="onDragOver" @dragleave="isDragging = false"
  >
    <input ref="inputRef" type="file" :accept="accept" class="sr-only" :disabled="disabled" @change="onChange" />
    <div class="min-w-0">
      <div class="text-xs font-medium text-muted">{{ label }}</div>
      <div :class="cn('font-mono text-[13px] mt-0.5 truncate', file ? 'text-text' : 'text-muted')">
        {{ file ? file.name : description }}
      </div>
    </div>
    <div class="flex items-center gap-2 flex-shrink-0">
      <span v-if="file" class="text-xs text-muted font-mono">{{ formatSize(file.size) }}</span>
      <button v-if="file" type="button" class="p-1 rounded-sm text-muted hover:text-danger hover:bg-danger/10 transition-colors" :disabled="disabled"
        :aria-label="`Remove ${label}`" @click.stop="emit('update:file', null)"><X :size="14" :stroke-width="1.5" /></button>
      <button type="button" class="text-xs text-accent hover:text-accent-hover font-medium" :disabled="disabled" @click.stop="browse">
        {{ file ? 'Change' : 'Browse' }}
      </button>
    </div>
  </div>

  <!-- Full: large drop target -->
  <div
    v-else
    :class="cn(
      'flex flex-col items-center justify-center gap-2 text-center px-4 py-5 border rounded transition-colors',
      file ? 'border-solid border-accent cursor-default'
        : isDragging ? 'drop-zone-active border-dashed'
        : 'border-dashed border-dim hover:border-muted cursor-pointer',
      disabled && 'opacity-45 cursor-not-allowed',
    )"
    @drop="onDrop" @dragover="onDragOver" @dragleave="isDragging = false" @click="onClick"
  >
    <input ref="inputRef" type="file" :accept="accept" class="sr-only" :disabled="disabled" @change="onChange" />
    <template v-if="file">
      <FileVideo v-if="fileKind(file) === 'video'" :size="24" :stroke-width="1.5" class="text-accent" />
      <FileText  v-else-if="fileKind(file) === 'text'" :size="24" :stroke-width="1.5" class="text-accent" />
      <FileIcon  v-else :size="24" :stroke-width="1.5" class="text-muted" />
      <div class="font-mono text-xs font-medium max-w-full truncate">{{ file.name }}</div>
      <div class="text-xs text-muted">{{ formatSize(file.size) }}</div>
      <button type="button" class="text-xs text-muted hover:text-danger inline-flex items-center gap-1" :disabled="disabled" @click.stop="emit('update:file', null)">
        <X :size="12" :stroke-width="1.5" />Remove
      </button>
    </template>
    <template v-else>
      <Upload :size="24" :stroke-width="1.5" :class="isDragging ? 'text-accent' : 'text-muted'" />
      <div :class="cn('text-sm font-medium', isDragging && 'text-accent')">{{ isDragging ? 'Release to load' : label }}</div>
      <div class="text-xs text-muted">{{ description }}</div>
    </template>
  </div>
</template>
