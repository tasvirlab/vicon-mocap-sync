import { ref } from 'vue'

// Palette lives in index.css (:root = dark, html.light = light); this only flips the class.
const isDark = ref(true)

function apply() {
  const root = document.documentElement
  for (const k of ['bg', 'surface', 'card', 'border', 'text', 'muted', 'dim']) root.style.removeProperty(`--color-${k}`)
  root.classList.toggle('light', !isDark.value)
}

export function useTheme() {
  function init() {
    try { isDark.value = localStorage.getItem('theme') !== 'light' } catch { isDark.value = true }
    apply()
  }
  function setDark(v: boolean) {
    isDark.value = v
    try { localStorage.setItem('theme', v ? 'dark' : 'light') } catch { /* storage blocked: theme still applies for this session */ }
    apply()
  }
  function toggle() { setDark(!isDark.value) }
  return { isDark, init, toggle, setDark }
}
