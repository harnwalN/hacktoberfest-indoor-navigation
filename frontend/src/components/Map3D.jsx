import { useEffect, useRef, useState } from 'react'
import { isMinorNode, placeKind } from './ArrowOverlay'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

// 3D view of the solved building map: every place at its estimated position in metres, floors stacked,
// stairs/elevators as vertical links. Useful for judging how well the walks line up.
const FLOOR_H = 6 // metres between floors (exaggerated a little for readability)

// Colours come from the CSS custom properties (styles.css) so the 3D view follows the theme, light or dark.
const cssVar = (name, fallback) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback
function readPalette() {
  return {
    place: cssVar('--room-fg', '#0b3a78'), key: cssVar('--accent', '#0b7a5e'), vertical: cssVar('--stairs', '#f2c230'),
    minor: cssVar('--control', '#8c847c'), noexit: cssVar('--danger', '#b42318'), selected: cssVar('--landmark-fg', '#7a2a08'),
    line: cssVar('--control', '#8c847c'), signBg: cssVar('--sign-bg', '#2a2420'), signFg: cssVar('--sign-fg', '#ffffff'),
    noexitBg: cssVar('--noexit-bg', '#ffdde6'), noexitFg: cssVar('--noexit-fg', '#800f33'),
  }
}
function useColorScheme() {
  const [, setTick] = useState(0)
  useEffect(() => {
    const mq = window.matchMedia?.('(prefers-color-scheme: dark)')
    const bump = () => setTick((t) => t + 1)
    mq?.addEventListener?.('change', bump)
    return () => mq?.removeEventListener?.('change', bump)
  }, [])
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

const isMinor = isMinorNode
const kind = (n) =>
  placeKind(n) === 'noexit' ? 'noexit' : isMinor(n) ? 'minor' : n.type === 'stairs' || n.type === 'elevator' ? 'vertical'
    : placeKind(n) === 'exit' || ['entrance', 'restroom'].includes(n.type) ? 'key' : 'place'

function labelSprite(text, color, bg) {
  const c = document.createElement('canvas')
  const ctx = c.getContext('2d')
  const font = '600 44px "Atkinson Hyperlegible", system-ui, sans-serif'
  ctx.font = font
  const w = Math.ceil(ctx.measureText(text).width) + 24
  c.width = w
  c.height = 64
  ctx.font = font
  ctx.fillStyle = bg
  ctx.beginPath()
  ctx.roundRect(0, 4, w, 56, 14)
  ctx.fill()
  ctx.fillStyle = color
  ctx.textBaseline = 'middle'
  ctx.fillText(text, 12, 33)
  const tex = new THREE.CanvasTexture(c)
  tex.colorSpace = THREE.SRGBColorSpace
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }))
  sprite.scale.set(w / 44, 64 / 44, 1)
  sprite.renderOrder = 10
  return sprite
}

export default function Map3D({ map, selected, onSelect }) {
  const host = useRef(null)
  const scheme = useColorScheme()

  useEffect(() => {
    const el = host.current
    const nodes = map.nodes.filter((n) => n.x != null && n.y != null)
    if (!el || !nodes.length) return undefined

    const COLORS = readPalette()
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    el.appendChild(renderer.domElement)
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, 1, 0.5, 2000)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true

    const cx = nodes.reduce((s, n) => s + n.x, 0) / nodes.length
    const cy = nodes.reduce((s, n) => s + n.y, 0) / nodes.length
    const floors = [...new Set(nodes.map((n) => n.floor))].sort((a, b) => a - b)
    const f0 = floors[0]
    const at = (n) => new THREE.Vector3(n.x - cx, (n.floor - f0) * FLOOR_H, -(n.y - cy))
    const pos = Object.fromEntries(nodes.map((n) => [n.id, at(n)]))

    // floor slabs sized to the places on each floor
    for (const f of floors) {
      const fn = nodes.filter((n) => n.floor === f)
      const xs = fn.map((n) => pos[n.id].x)
      const zs = fn.map((n) => pos[n.id].z)
      const [minX, maxX, minZ, maxZ] = [Math.min(...xs) - 5, Math.max(...xs) + 5, Math.min(...zs) - 5, Math.max(...zs) + 5]
      const slab = new THREE.Mesh(
        new THREE.PlaneGeometry(maxX - minX, maxZ - minZ),
        new THREE.MeshBasicMaterial({ color: COLORS.key, transparent: true, opacity: 0.08, side: THREE.DoubleSide, depthWrite: false }),
      )
      slab.rotation.x = -Math.PI / 2
      slab.position.set((minX + maxX) / 2, (f - f0) * FLOOR_H - 0.3, (minZ + maxZ) / 2)
      scene.add(slab)
      const edge = new THREE.LineSegments(new THREE.EdgesGeometry(slab.geometry), new THREE.LineBasicMaterial({ color: COLORS.key, transparent: true, opacity: 0.4 }))
      edge.rotation.copy(slab.rotation)
      edge.position.copy(slab.position)
      scene.add(edge)
      const tag = labelSprite(`Floor ${f}`, COLORS.vertical, COLORS.signBg)
      tag.position.set(minX + 4, (f - f0) * FLOOR_H + 1, maxZ)
      scene.add(tag)
    }

    // connections
    for (const e of map.edges) {
      const a = pos[e.from_node]
      const b = pos[e.to_node]
      if (!a || !b) continue
      const vertical = e.vertical !== 'none'
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([a, b]),
        new THREE.LineBasicMaterial({ color: vertical ? COLORS.vertical : COLORS.line, transparent: true, opacity: vertical ? 0.95 : 0.6 }),
      )
      scene.add(line)
    }

    // places
    const pickables = []
    for (const n of nodes) {
      const k = kind(n)
      const r = k === 'minor' ? 0.35 : 0.8
      const geo = k === 'noexit' ? new THREE.OctahedronGeometry(r * 1.3) : k === 'vertical' ? new THREE.BoxGeometry(r * 1.8, r * 1.8, r * 1.8) : new THREE.SphereGeometry(r, 20, 14)
      const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: n.id === selected ? COLORS.selected : COLORS[k] }))
      mesh.position.copy(pos[n.id])
      mesh.userData.id = n.id
      scene.add(mesh)
      pickables.push(mesh)
      if ((n.verified && k !== 'minor') || n.id === selected) {
        const s = labelSprite((n.label || '').length > 26 ? `${n.label.slice(0, 25)}…` : n.label || '', k === 'noexit' ? COLORS.noexitFg : COLORS.signFg, k === 'noexit' ? COLORS.noexitBg : COLORS.signBg)
        s.position.copy(pos[n.id]).add(new THREE.Vector3(0, 2.2, 0))
        scene.add(s)
      }
    }

    // camera: look at the middle of the building from above at an angle
    const span = Math.max(30, ...nodes.map((n) => Math.hypot(n.x - cx, n.y - cy))) * 1.6
    const mid = ((floors.length - 1) * FLOOR_H) / 2
    camera.position.set(span * 0.7, mid + span * 0.8, span * 0.9)
    controls.target.set(0, mid, 0)

    const ray = new THREE.Raycaster()
    const onClick = (ev) => {
      const r = renderer.domElement.getBoundingClientRect()
      ray.setFromCamera(new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1), camera)
      const hit = ray.intersectObjects(pickables)[0]
      if (hit) onSelect?.(hit.object.userData.id)
    }
    renderer.domElement.addEventListener('click', onClick)

    const resize = () => {
      const w = el.clientWidth
      const h = el.clientHeight
      renderer.setSize(w, h)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
    }
    const ro = new ResizeObserver(resize)
    ro.observe(el)
    resize()

    let frame
    const loop = () => {
      controls.update()
      renderer.render(scene, camera)
      frame = requestAnimationFrame(loop)
    }
    loop()

    return () => {
      cancelAnimationFrame(frame)
      ro.disconnect()
      renderer.domElement.removeEventListener('click', onClick)
      controls.dispose()
      scene.traverse((o) => {
        o.geometry?.dispose()
        o.material?.map?.dispose()
        o.material?.dispose()
      })
      renderer.dispose()
      el.removeChild(renderer.domElement)
    }
  }, [map, selected, scheme])

  const places = (map.nodes || []).filter((n) => n.x != null && n.y != null && n.label)
  // The canvas is pointer-only; this visually hidden list is the keyboard / screen-reader path to the same selection.
  return (
    <>
      <div ref={host} className="map3d" role="img" aria-label="3D view of the building map. Pointer only; use the place list or the 2D map to select places by keyboard." />
      {places.length > 0 && (
        <ul aria-label="Places in the 3D view" className="sr-only">
          {places.map((n) => (
            <li key={n.id}>
              <button type="button" onClick={() => onSelect?.(n.id)} aria-pressed={n.id === selected}>
                {n.label}
              </button>
            </li>
          ))}
        </ul>
      )}
    </>
  )
}
