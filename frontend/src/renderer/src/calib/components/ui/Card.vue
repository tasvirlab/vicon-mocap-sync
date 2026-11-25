<script setup lang="ts">
import { cn } from '@renderer/shared/cn'
defineProps<{ title?: string; class?: string }>()
</script>

<template>
  <section :class="cn('border border-border rounded p-6', $props.class)">
    <header v-if="title || $slots.header" class="flex items-center justify-between gap-3 mb-4">
      <h2 v-if="title" class="text-base font-semibold">{{ title }}</h2>
      <slot name="header" />
    </header>
    <slot />
  </section>
</template>
