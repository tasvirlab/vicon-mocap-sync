<script setup lang="ts">
// Shared by the calibration and sync modules: nav is data, so both render the same shell.
import type { Component } from 'vue'
import type { NavGroup } from '@renderer/shared/nav'
import { useRoute, useRouter } from 'vue-router'
import { Moon, Sun } from '@lucide/vue'
import { useTheme } from '@renderer/shared/useTheme'
import { cn } from '@renderer/shared/cn'

defineProps<{ brand: string; sub?: string; logo: Component; groups: NavGroup[] }>()

const route  = useRoute()
const router = useRouter()
const { isDark, setDark } = useTheme()

function isActive(to: string) {
  return to === '/' ? route.path === '/' : route.path.startsWith(to)
}
</script>

<template>
  <aside class="flex flex-col w-[200px] bg-surface border-r border-border h-full select-none flex-shrink-0 px-3 py-4">
    <!-- Brand -->
    <div class="flex items-center gap-2.5 px-2 pb-5">
      <div class="w-8 h-8 rounded border border-accent text-accent grid place-items-center flex-shrink-0">
        <component :is="logo" :size="16" :stroke-width="1.5" />
      </div>
      <div class="leading-none min-w-0">
        <div class="text-[17px] font-bold truncate">{{ brand }}</div>
        <div v-if="sub" class="text-xs text-muted mt-[3px]">{{ sub }}</div>
      </div>
    </div>

    <!-- Nav -->
    <nav class="flex flex-col gap-0.5 flex-1 overflow-y-auto">
      <template v-for="(g, gi) in groups" :key="gi">
        <p v-if="g.heading" class="text-[11px] font-medium text-muted px-2 pt-4 pb-1.5">{{ g.heading }}</p>
        <button
          v-for="it in g.items" :key="it.to"
          :class="cn(
            'relative flex items-center gap-2.5 h-9 px-2 rounded text-left text-sm font-medium transition-colors',
            isActive(it.to) ? 'bg-tint text-accent' : 'text-text hover:bg-tint'
          )"
          @click="router.push(it.to)"
        >
          <span v-if="isActive(it.to)" class="absolute left-0 top-1.5 bottom-1.5 w-[2px] bg-accent rounded-full" />
          <span v-if="it.num" class="w-5 h-5 flex-shrink-0 grid place-items-center rounded-sm border border-accent text-accent font-mono text-[11px]">{{ it.num }}</span>
          <component :is="it.icon" v-else-if="it.icon" :size="18" :stroke-width="1.5"
            :class="isActive(it.to) ? 'text-accent' : 'text-muted'" class="flex-shrink-0" />
          <span class="truncate">{{ it.label }}</span>
        </button>
      </template>
    </nav>

    <!-- Theme -->
    <div class="flex border border-border rounded h-[34px] mt-3 overflow-hidden" role="group" aria-label="Theme">
      <button
        :class="cn('flex-1 flex items-center justify-center gap-1.5 text-xs transition-colors', isDark ? 'bg-tint text-accent font-medium' : 'text-muted hover:text-text')"
        :aria-pressed="isDark" @click="setDark(true)"
      ><Moon :size="14" :stroke-width="1.5" />Dark</button>
      <button
        :class="cn('flex-1 flex items-center justify-center gap-1.5 text-xs transition-colors', !isDark ? 'bg-tint text-accent font-medium' : 'text-muted hover:text-text')"
        :aria-pressed="!isDark" @click="setDark(false)"
      ><Sun :size="14" :stroke-width="1.5" />Light</button>
    </div>
  </aside>
</template>
