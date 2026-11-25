<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { ExtrinsicResult } from '@renderer/calib/lib/types'
import { useTheme } from '@renderer/shared/useTheme'

const props = defineProps<{ result: ExtrinsicResult | null }>()

const containerRef = ref<HTMLDivElement | null>(null)
const { isDark } = useTheme()

let renderer: THREE.WebGLRenderer | null = null
let scene: THREE.Scene | null = null
let camera: THREE.PerspectiveCamera | null = null
let controls: OrbitControls | null = null
let animId = 0
let resultGroup: THREE.Group | null = null
let staticGroup: THREE.Group | null = null

// Design-system tokens (CSS variables hold "r g b") -> THREE.Color, so the scene follows the theme.
function tok(name: string): THREE.Color {
  const raw = getComputedStyle(document.documentElement).getPropertyValue(`--color-${name}`).trim()
  const [r, g, b] = raw.split(/\s+/).map(Number)
  return new THREE.Color((r ?? 128) / 255, (g ?? 128) / 255, (b ?? 128) / 255)
}

function makeAxes(len: number) {
  const g = new THREE.Group()
  g.add(new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), len, tok('danger'), 30, 15))
  g.add(new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), len, tok('success'), 30, 15))
  g.add(new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), len, tok('accent'), 30, 15))
  return g
}

function makeFrustum(fov = 60, aspect = 16 / 9, near = 50, far = 600) {
  const accent = tok('accent')
  const g = new THREE.Group()
  g.add(new THREE.Mesh(
    new THREE.BoxGeometry(60, 40, 40),
    new THREE.MeshBasicMaterial({ color: accent, transparent: true, opacity: 0.3, wireframe: true }),
  ))
  const hN = Math.tan((fov * Math.PI / 180) / 2) * near, wN = hN * aspect
  const hF = Math.tan((fov * Math.PI / 180) / 2) * far,  wF = hF * aspect
  const pts = [[-wN,-hN,near],[wN,-hN,near],[wN,hN,near],[-wN,hN,near],
               [-wF,-hF,far], [wF,-hF,far], [wF,hF,far], [-wF,hF,far]]
    .map(([x, y, z]) => new THREE.Vector3(x, y, z))
  const idx = [0,1,1,2,2,3,3,0, 4,5,5,6,6,7,7,4, 0,4,1,5,2,6,3,7]
  const verts: number[] = []
  for (let i = 0; i < idx.length; i += 2) {
    const a = pts[idx[i]], b = pts[idx[i + 1]]
    verts.push(a.x, a.y, a.z, b.x, b.y, b.z)
  }
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3))
  g.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: accent, transparent: true, opacity: 0.75 })))
  g.add(new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), far * 0.4, tok('accent2'), 20, 10))
  return g
}

function makeBoard(cols = 9, rows = 6, sq = 25) {
  const g = new THREE.Group()
  const w = cols * sq, h = rows * sq
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(w, h),
    new THREE.MeshBasicMaterial({ color: tok('border'), transparent: true, opacity: 0.6, side: THREE.DoubleSide }),
  )
  plane.rotation.x = -Math.PI / 2
  g.add(plane)
  const edges = new THREE.EdgesGeometry(new THREE.BoxGeometry(w, 5, h))
  g.add(new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: tok('accent'), transparent: true, opacity: 0.6 })))
  return g
}

function buildStatic() {
  if (!scene) return
  if (staticGroup) { scene.remove(staticGroup); staticGroup = null }
  const g = new THREE.Group()
  g.add(new THREE.GridHelper(3000, 30, tok('dim'), tok('border')))
  g.add(makeAxes(400))
  g.add(makeBoard())
  staticGroup = g
  scene.add(g)
  renderer?.setClearColor(tok('surface'))
}

function updateResult(result: ExtrinsicResult | null) {
  if (!scene) return
  if (resultGroup) { scene.remove(resultGroup); resultGroup = null }
  if (!result) return

  const group = new THREE.Group()
  const p = result.camera_position_world_mm
  const camPos = new THREE.Vector3(p.x, p.y, p.z)

  const R = result.R_matrix
  const m4 = new THREE.Matrix4().set(
     R[0][0],  R[0][1],  R[0][2], 0,
    -R[1][0], -R[1][1], -R[1][2], 0,
    -R[2][0], -R[2][1], -R[2][2], 0,
     0,        0,        0,        1,
  )
  const camQuat = new THREE.Quaternion().setFromRotationMatrix(m4).invert()

  const frustum = makeFrustum()
  frustum.position.copy(camPos)
  frustum.quaternion.copy(camQuat)
  group.add(frustum)

  result.per_marker_errors.forEach((m, i) => {
    const color = m.error_px < 3 ? tok('success') : m.error_px < 8 ? tok('warning') : tok('danger')
    const sphere = new THREE.Mesh(
      new THREE.SphereGeometry(12, 12, 12),
      new THREE.MeshBasicMaterial({ color }),
    )
    sphere.position.set(i * 80 - result.per_marker_errors.length * 40, 0, 0)
    group.add(sphere)
  })

  resultGroup = group
  scene.add(group)
}

onMounted(() => {
  const el = containerRef.value
  if (!el) return

  renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setSize(el.clientWidth, el.clientHeight)
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  el.appendChild(renderer.domElement)

  scene  = new THREE.Scene()
  camera = new THREE.PerspectiveCamera(50, el.clientWidth / el.clientHeight, 1, 10000)
  camera.position.set(800, 600, 800)

  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.dampingFactor = 0.08
  controls.rotateSpeed   = 0.6
  controls.zoomSpeed     = 0.8

  scene.add(new THREE.AmbientLight(0xffffff, 0.6))
  const pt = new THREE.PointLight(0xffffff, 0.5)
  pt.position.set(500, 500, 500)
  scene.add(pt)
  buildStatic()
  updateResult(props.result)

  const animate = () => {
    animId = requestAnimationFrame(animate)
    controls!.update()
    renderer!.render(scene!, camera!)
  }
  animate()
})

watch(() => props.result, updateResult)
// Theme flipped: variables change on <html>, so rebuild every colored object.
watch(isDark, () => { buildStatic(); updateResult(props.result) }, { flush: 'post' })

onUnmounted(() => {
  cancelAnimationFrame(animId)
  controls?.dispose()
  if (renderer) { renderer.dispose(); renderer.domElement.remove() }
})
</script>

<template>
  <div ref="containerRef" class="three-canvas relative w-full h-[300px] bg-surface rounded border border-border overflow-hidden cursor-grab">
    <div class="absolute bottom-3 left-3 flex items-center gap-3 text-xs text-muted pointer-events-none z-10 font-mono">
      <span class="flex items-center gap-1.5"><i class="w-3 h-0.5 bg-danger inline-block" />X</span>
      <span class="flex items-center gap-1.5"><i class="w-3 h-0.5 bg-success inline-block" />Y</span>
      <span class="flex items-center gap-1.5"><i class="w-3 h-0.5 bg-accent inline-block" />Z</span>
    </div>
  </div>
</template>
