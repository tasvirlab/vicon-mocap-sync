<script setup lang="ts">
defineProps<{ cols?: number }>()
</script>

<template>
  <div class="grid gap-3" :style="{ gridTemplateColumns: `repeat(${cols ?? 3}, minmax(0, 1fr))` }">
    <slot />
  </div>
</template>
