<script setup lang="ts">
/**
 * ClickCanvas — interactive frame viewer for body-marker clicking.
 *
 * Magnifier behaviour:
 *   - Outside loupe: cursor pans the magnified region (live tracking)
 *   - Inside loupe:  focus FREEZES — cursor only moves the crosshair,
 *                    giving sub-pixel precision without jitter
 *   - Clicking inside loupe uses the precise crosshair position
 *   - Clicking outside loupe uses the normal canvas position
 */
import { ref, watch, onMounted, nextTick } from 'vue'

interface Pt { x: number; y: number }

const props = defineProps<{
  frameImage: string
  joints:     string[]
  modelValue: Pt[]
}>()

const emit = defineEmits<{
  'update:modelValue': [v: Pt[]]
}>()

// ── Canvas refs ───────────────────────────────────────────────────────────────
const mainRef = ref<HTMLCanvasElement | null>(null)
const magRef  = ref<HTMLCanvasElement | null>(null)
const imgEl   = ref<HTMLImageElement | null>(null)

// ── Magnifier constants ───────────────────────────────────────────────────────
const ZOOM = 8            // 8× magnification
const MAG  = 200          // magnifier panel display size (px)
const HALF = MAG / (2 * ZOOM)   // image-pixel half-width of source region = 12.5

// ── Cursor / loupe state ──────────────────────────────────────────────────────
// focusX/focusY — frozen focus point in CANVAS DISPLAY coordinates.
// Frozen when cursor enters the loupe; updated freely when outside.
const focusX  = ref(0)
const focusY  = ref(0)

// Cursor position within the loupe panel itself (0..MAG)
const loupeX  = ref(MAG / 2)
const loupeY  = ref(MAG / 2)

const inLoupe = ref(false)
const showMag = ref(false)

// ── Theme colors (design tokens; canvas cannot use CSS variables directly) ───
function cssRgb(name: string): string {
  const raw = getComputedStyle(document.documentElement).getPropertyValue(`--color-${name}`).trim()
  return `rgb(${raw.split(/\s+/).join(',')})`
}

// ── Loupe panel position (canvas display coords) ──────────────────────────────
// Panel is at top-2 right-2 = offset 8px from top/right.
function loupeBounds(): { x: number; y: number } | null {
  const canvas = mainRef.value
  if (!canvas) return null
  const w = canvas.getBoundingClientRect().width
  return { x: w - MAG - 8, y: 8 }
}

function isInLoupe(cx: number, cy: number): boolean {
  const b = loupeBounds()
  return !!b && cx >= b.x && cx < b.x + MAG && cy >= b.y && cy < b.y + MAG
}

// ── Image loading ─────────────────────────────────────────────────────────────
function toSrc(b64: string): string {
  return b64.startsWith('data:') ? b64 : `data:image/jpeg;base64,${b64}`
}

let _lastSrc = ''
async function ensureImage(): Promise<HTMLImageElement> {
  const src = toSrc(props.frameImage)
  if (imgEl.value && src === _lastSrc) return imgEl.value
  return new Promise((res, rej) => {
    const im = new Image()
    im.onload  = () => { imgEl.value = im; _lastSrc = src; res(im) }
    im.onerror = rej
    im.src     = src
  })
}

// ── Main canvas draw ──────────────────────────────────────────────────────────
async function redraw() {
  const canvas = mainRef.value
  if (!canvas || !props.frameImage) return
  const im = await ensureImage()
  imgEl.value     = im
  canvas.width    = im.naturalWidth
  canvas.height   = im.naturalHeight
  const ctx       = canvas.getContext('2d')!

  ctx.drawImage(im, 0, 0)

  props.modelValue.forEach((pt, i) => {
    const color = cssRgb('accent')
    ctx.beginPath()
    ctx.arc(pt.x, pt.y, 9, 0, Math.PI * 2)
    ctx.fillStyle   = color
    ctx.fill()
    ctx.strokeStyle = 'white'
    ctx.lineWidth   = 2.5
    ctx.stroke()
    ctx.fillStyle   = 'white'
    ctx.font        = 'bold 15px monospace'
    ctx.shadowColor = '#000'
    ctx.shadowBlur  = 4
    ctx.fillText(`${i + 1}:${props.joints[i]}`, pt.x + 13, pt.y - 5)
    ctx.shadowBlur  = 0
  })

  const nextIdx = props.modelValue.length
  if (nextIdx < props.joints.length) {
    const color = cssRgb('accent')
    ctx.fillStyle = 'rgba(0,0,0,0.6)'
    ctx.fillRect(8, 8, 320, 36)
    ctx.fillStyle = color
    ctx.font      = 'bold 16px monospace'
    ctx.fillText(`→ ${nextIdx + 1}/${props.joints.length}: ${props.joints[nextIdx]}`, 16, 32)
  }
}

// ── Magnifier draw ─────────────────────────────────────────────────────────────
function drawMagnifier() {
  const mCanvas = magRef.value
  const im      = imgEl.value
  if (!mCanvas || !im || !showMag.value || !mainRef.value) return

  const rect   = mainRef.value.getBoundingClientRect()
  const scaleX = im.naturalWidth  / rect.width
  const scaleY = im.naturalHeight / rect.height

  // Source region centred on the FROZEN focus point
  const imgX   = focusX.value * scaleX
  const imgY   = focusY.value * scaleY

  mCanvas.width  = MAG
  mCanvas.height = MAG
  const ctx = mCanvas.getContext('2d')!

  ctx.imageSmoothingEnabled = false
  ctx.drawImage(im,
    imgX - HALF, imgY - HALF, HALF * 2, HALF * 2,
    0, 0, MAG, MAG,
  )

  // Existing click dots projected into loupe space
  props.modelValue.forEach((pt, i) => {
    const px = (pt.x * scaleX - (imgX - HALF)) * ZOOM
    const py = (pt.y * scaleY - (imgY - HALF)) * ZOOM
    ctx.beginPath()
    ctx.arc(px, py, 7, 0, Math.PI * 2)
    ctx.fillStyle = cssRgb('accent')
    ctx.fill()
  })

  // Crosshair — follows loupe-local cursor when inside, otherwise centred
  const cxh = inLoupe.value ? loupeX.value : MAG / 2
  const cyh = inLoupe.value ? loupeY.value : MAG / 2
  ctx.strokeStyle = cssRgb('success')
  ctx.lineWidth   = 1.5
  ctx.beginPath()
  ctx.moveTo(cxh, cyh - 16); ctx.lineTo(cxh, cyh + 16)
  ctx.moveTo(cxh - 16, cyh); ctx.lineTo(cxh + 16, cyh)
  ctx.stroke()

  ctx.strokeStyle = cssRgb('success')
  ctx.lineWidth   = 2
  ctx.strokeRect(0, 0, MAG, MAG)

  // Label: lock icon when frozen inside loupe
  ctx.fillStyle = inLoupe.value ? cssRgb('success') : cssRgb('muted')
  ctx.font      = '10px monospace'
  ctx.fillText(inLoupe.value ? `${ZOOM}× ⊕ locked` : `${ZOOM}× zoom`, 5, MAG - 5)
}

// ── Mouse handlers ────────────────────────────────────────────────────────────
function onMouseMove(e: MouseEvent) {
  const canvas = mainRef.value
  if (!canvas) return
  const rect = canvas.getBoundingClientRect()
  const cx   = e.clientX - rect.left
  const cy   = e.clientY - rect.top

  if (isInLoupe(cx, cy)) {
    // INSIDE loupe — freeze focus, only update crosshair position
    inLoupe.value = true
    const b       = loupeBounds()!
    loupeX.value  = cx - b.x
    loupeY.value  = cy - b.y
    // focusX/Y intentionally NOT updated — keeps the view locked
  } else {
    // OUTSIDE loupe — pan normally
    inLoupe.value = false
    focusX.value  = cx
    focusY.value  = cy
  }

  showMag.value = true
  drawMagnifier()
}

function onMouseLeave() {
  showMag.value = false
  inLoupe.value = false
}

function onClick(e: MouseEvent) {
  const canvas = mainRef.value
  const im     = imgEl.value
  if (!canvas || !im) return
  if (props.modelValue.length >= props.joints.length) return

  const rect   = canvas.getBoundingClientRect()
  const scaleX = im.naturalWidth  / rect.width
  const scaleY = im.naturalHeight / rect.height
  const cx     = e.clientX - rect.left
  const cy     = e.clientY - rect.top

  let x: number, y: number

  if (isInLoupe(cx, cy)) {
    // Sub-pixel click from loupe crosshair position
    const imgFocusX = focusX.value * scaleX
    const imgFocusY = focusY.value * scaleY
    x = (imgFocusX - HALF) + loupeX.value / ZOOM
    y = (imgFocusY - HALF) + loupeY.value / ZOOM
  } else {
    // Normal canvas click
    x = cx * scaleX
    y = cy * scaleY
  }

  emit('update:modelValue', [...props.modelValue, { x, y }])
}

function undo() {
  if (props.modelValue.length > 0)
    emit('update:modelValue', props.modelValue.slice(0, -1))
}

// ── Reactivity ────────────────────────────────────────────────────────────────
watch(() => [props.frameImage, JSON.stringify(props.modelValue)], () => {
  nextTick(redraw)
}, { immediate: false })

onMounted(redraw)
</script>

<template>
  <div class="relative select-none">
    <!-- Main canvas -->
    <div class="relative rounded overflow-hidden border border-border bg-surface">
      <canvas
        ref="mainRef"
        class="w-full cursor-crosshair block"
        @mousemove="onMouseMove"
        @mouseleave="onMouseLeave"
        @click="onClick"
      />
      <!-- Magnifier panel: top-right, pointer-events-none so clicks pass through -->
      <div v-show="showMag" class="absolute top-2 right-2 pointer-events-none">
        <canvas
          ref="magRef"
          :width="MAG"
          :height="MAG"
          class="rounded shadow-lg"
          :style="{
            imageRendering: 'pixelated',
            width:  `${MAG}px`,
            height: `${MAG}px`,
            outline: inLoupe ? '2px solid rgb(var(--color-success))' : '2px solid rgb(var(--color-dim))',
          }"
        />
      </div>
    </div>

    <!-- Controls row -->
    <div class="flex items-center gap-2 mt-3">
      <button class="h-[34px] px-3.5 border border-muted rounded text-[13px] font-medium hover:border-accent hover:bg-tint transition-colors disabled:opacity-45 disabled:cursor-not-allowed"
        :disabled="modelValue.length === 0" @click="undo">Undo</button>
      <button class="h-[34px] px-3.5 border border-muted rounded text-[13px] font-medium hover:border-danger hover:text-danger transition-colors disabled:opacity-45 disabled:cursor-not-allowed"
        :disabled="modelValue.length === 0" @click="emit('update:modelValue', [])">Clear</button>
      <span class="ml-auto text-[13px] font-mono text-muted">
        <span :class="modelValue.length === joints.length ? 'text-success' : 'text-text'">{{ modelValue.length }}</span>/{{ joints.length }} joints
      </span>
    </div>

    <!-- Joint legend -->
    <div class="flex flex-wrap gap-1.5 mt-3">
      <span v-for="(joint, i) in joints" :key="joint"
        :class="['px-2 py-0.5 rounded-sm text-xs font-mono border', i < modelValue.length ? 'border-accent text-accent' : 'border-border text-muted']">
        {{ i + 1 }}:{{ joint }}<span v-if="i < modelValue.length" class="ml-1">✓</span>
      </span>
    </div>
  </div>
</template>
