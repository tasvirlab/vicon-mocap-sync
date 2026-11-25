import { ref, onMounted, onUnmounted } from 'vue'
import { checkHealth } from '@renderer/sync/lib/api'

export type ServerState = 'checking' | 'online' | 'offline'

export function useServerStatus() {
  const state = ref<ServerState>('checking')
  let timer: ReturnType<typeof setTimeout> | null = null
  let cancelled = false

  async function schedule(delay: number) {
    await new Promise<void>(r => { timer = setTimeout(r, delay) })
    if (cancelled) return
    const ok = await checkHealth()
    if (cancelled) return
    state.value = ok ? 'online' : 'offline'
    schedule(ok ? 30_000 : 5_000)
  }

  onMounted(() => { cancelled = false; schedule(0) })
  onUnmounted(() => { cancelled = true; if (timer) clearTimeout(timer) })

  return { state }
}
