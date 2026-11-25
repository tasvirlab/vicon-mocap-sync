<script setup lang="ts">
import { onMounted } from 'vue'
import MainLayout from './shell/MainLayout.vue'
import { useTheme } from './shared/useTheme'

const { init } = useTheme()
onMounted(init)
</script>

<template>
  <MainLayout />
</template>
