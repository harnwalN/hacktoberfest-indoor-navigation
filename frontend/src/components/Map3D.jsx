import { useEffect, useRef, useState } from 'react'
import { isMinorNode, placeKind } from './ArrowOverlay'
import { reducedMotion } from './Motion'
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

// Labels are screen-space: they stay LABEL_PX tall (text ~13px) at any zoom, so they read on a phone.
const LABEL_PX = 19
const ease = (t) => 1 - Math.pow(1 - Math.min(1, Math.max(0, t)), 4)

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
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true, sizeAttenuation: false }))
  sprite.userData.label = { w: (w / 64) * LABEL_PX, h: LABEL_PX } // size in CSS pixels, applied on resize
  sprite.renderOrder = 10
  return sprite
}

export default function Map3D({ map, selected, onSelect }) {
  const host = useRef(null)
  const scheme = useColorScheme()
  // Camera survives rebuilds (a new map or theme re-creates the scene) so it never snaps back or replays the intro.
  const cam = useRef(null)
  // Selection never rebuilds the scene: the build effect hands back `select`, and the selection effect calls it.
  const selRef = useRef(selected)
  selRef.current = selected
  const onSelectRef = useRef(onSelect)
  onSelectRef.current = onSelect
  const api = useRef(null)

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
    const calm = reducedMotion()
    scene.add(new THREE.AmbientLight(0xffffff, 1.6))
    const sun = new THREE.DirectionalLight(0xffffff, 2.2)
    sun.position.set(0.6, 1, 0.8)
    scene.add(sun)
    const labels = []

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
      tag.userData.prio = 4
      labels.push(tag)
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
    const meshById = {}
    const labelText = (n) => ((n.label || '').length > 26 ? `${n.label.slice(0, 25)}…` : n.label || '')
    const makeLabel = (n, k) => {
      const s = labelSprite(labelText(n), k === 'noexit' ? COLORS.noexitFg : COLORS.signFg, k === 'noexit' ? COLORS.noexitBg : COLORS.signBg)
      s.position.copy(pos[n.id]).add(new THREE.Vector3(0, 2.2, 0))
      scene.add(s)
      labels.push(s)
      return s
    }
    for (const n of nodes) {
      const k = kind(n)
      const r = k === 'minor' ? 0.35 : 0.8
      const geo = k === 'noexit' ? new THREE.OctahedronGeometry(r * 1.3) : k === 'vertical' ? new THREE.BoxGeometry(r * 1.8, r * 1.8, r * 1.8) : new THREE.SphereGeometry(r, 20, 14)
      // The No Exit octahedron is lit (flat-shaded facets) so it reads as a solid; the rest stay flat-coloured.
      const mat = k === 'noexit' ? new THREE.MeshStandardMaterial({ color: COLORS[k], flatShading: true, roughness: 0.5, metalness: 0.05 }) : new THREE.MeshBasicMaterial({ color: COLORS[k] })
      const mesh = new THREE.Mesh(geo, mat)
      mesh.position.copy(pos[n.id])
      mesh.userData = { id: n.id, node: n, kind: k, prio: k === 'vertical' || k === 'noexit' ? 3 : 2, label: null, tempLabel: false }
      scene.add(mesh)
      pickables.push(mesh)
      meshById[n.id] = mesh
      if (n.verified && k !== 'minor') {
        mesh.userData.label = makeLabel(n, k)
        mesh.userData.label.userData.prio = mesh.userData.prio
      }
    }

    // camera: look at the middle of the building from above at an angle, far enough back that the whole
    // building (its real bounds, not a guess) fits the narrower of the two fields of view, with a 10% margin.
    const box = new THREE.Box3().setFromPoints(Object.values(pos))
    const centre = box.getCenter(new THREE.Vector3())
    const radius = Math.max(6, ...Object.values(pos).map((p) => p.distanceTo(centre)))
    const VFOV = (camera.fov * Math.PI) / 180
    const fitDist = (aspect) => {
      const hfov = 2 * Math.atan(Math.tan(VFOV / 2) * (aspect || 1))
      return (radius / Math.sin(Math.min(VFOV, hfov) / 2)) * 1.1
    }
    let k = fitDist(el.clientWidth / (el.clientHeight || 1))
    const target = centre.clone()
    const rest = new THREE.Vector3(0.7, 0.8, 0.9).normalize().multiplyScalar(k) // resting offset from the target
    const restored = !!cam.current
    if (restored) {
      camera.position.copy(cam.current.pos)
      target.copy(cam.current.target)
      k = cam.current.k
      rest.setLength(k)
    } else camera.position.copy(target).add(rest)
    controls.target.copy(target)
    // Intro: fly in from a little further out and round the side, then drift round until the first touch.
    const t0 = performance.now()
    const INTRO = 1300
    let intro = !restored && !calm
    controls.autoRotate = !calm && !(cam.current?.touched)
    controls.autoRotateSpeed = 1.1
    let touched = !!cam.current?.touched
    const grow = (m, i) => ease((performance.now() - t0 - Math.min(i * 16, 450)) / 450)
    const stopAuto = () => {
      if (intro) { pickables.forEach((m) => m.scale.setScalar(1)); labels.forEach((l) => { l.material.opacity = 1 }) }
      intro = false; touched = true; controls.autoRotate = false
    }
    controls.addEventListener('start', stopAuto)
    if (intro) {
      pickables.forEach((m) => m.scale.setScalar(0.001))
      labels.forEach((l) => { l.material.opacity = 0 })
    }

    const ray = new THREE.Raycaster()
    const onClick = (ev) => {
      const r = renderer.domElement.getBoundingClientRect()
      ray.setFromCamera(new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1), camera)
      const hit = ray.intersectObjects(pickables)[0]
      if (hit) onSelectRef.current?.(hit.object.userData.id)
    }
    renderer.domElement.addEventListener('click', onClick)

    let viewH = el.clientHeight || 1
    const sizeLabel = (l) => {
      const f = camera.projectionMatrix.elements[5]
      const { w: lw, h: lh } = l.userData.label
      l.scale.set((lw * 2) / (f * viewH), (lh * 2) / (f * viewH), 1)
    }
    const resize = () => {
      const w = el.clientWidth
      const h = el.clientHeight
      if (!w || !h) return
      viewH = h
      renderer.setSize(w, h)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      // keep the building fitted when the view changes shape (phone rotation, panel resize)
      const nk = fitDist(w / h)
      if (Math.abs(nk - k) / k > 0.01) {
        const off = camera.position.clone().sub(controls.target).multiplyScalar(nk / k)
        camera.position.copy(controls.target).add(off)
        rest.multiplyScalar(nk / k)
        k = nk
      }
      labels.forEach(sizeLabel)
    }
    const ro = new ResizeObserver(resize)
    ro.observe(el)
    resize()

    let frame
    // Screen-space labels keep their size, so crowded ones would pile up: show the most important first and
    // hide any label that would land on top of one already shown (the keyboard list below still has every place).
    const proj = new THREE.Vector3()
    let tick = 0
    const declutter = () => {
      const w = el.clientWidth
      const h = el.clientHeight
      const shown = []
      for (const l of [...labels].sort((a, b) => b.userData.prio - a.userData.prio)) {
        proj.copy(l.position).project(camera)
        const x = (proj.x * 0.5 + 0.5) * w
        const y = (-proj.y * 0.5 + 0.5) * h
        const hw = l.userData.label.w / 2 + 3
        const hh = l.userData.label.h / 2 + 2
        const hit = proj.z > 1 || shown.some((q) => x - hw < q[2] && x + hw > q[0] && y - hh < q[3] && y + hh > q[1])
        l.visible = !hit
        if (!hit) shown.push([x - hw, y - hh, x + hw, y + hh])
      }
    }

    // Selection: recolour, re-prioritise the label (or add one for an unlabelled place), reset the old pulse.
    // No geometry, renderer or controls are touched, so it costs well under a frame.
    let current = null
    const select = (id) => {
      const next = meshById[id] || null
      if (next === current) return
      if (current) {
        const u = current.userData
        u.selected = false
        current.material.color.set(COLORS[u.kind])
        if (!intro) current.scale.setScalar(1)
        if (u.tempLabel && u.label) {
          scene.remove(u.label)
          labels.splice(labels.indexOf(u.label), 1)
          u.label.material.map?.dispose()
          u.label.material.dispose()
          u.label = null
          u.tempLabel = false
        } else if (u.label) u.label.userData.prio = u.prio
      }
      current = next
      if (next) {
        const u = next.userData
        u.selected = true
        next.material.color.set(COLORS.selected)
        if (!u.label) {
          u.label = makeLabel(u.node, u.kind)
          u.tempLabel = true
          sizeLabel(u.label)
          if (intro) u.label.material.opacity = 0
        }
        u.label.userData.prio = 5
      }
      declutter()
    }
    select(selRef.current)
    api.current = { select }

    const Y = new THREE.Vector3(0, 1, 0)
    const loop = () => {
      if (tick++ % 6 === 0) declutter()
      const now = performance.now()
      const e = now - t0
      if (intro) {
        if (e >= INTRO) intro = false
        const q = 1 - ease(e / INTRO)
        camera.position.copy(controls.target).add(rest.clone().multiplyScalar(1 + 0.55 * q).applyAxisAngle(Y, -0.9 * q))
        pickables.forEach((m, i) => m.scale.setScalar(Math.max(0.001, grow(m, i))))
        labels.forEach((l) => { l.material.opacity = ease((e - 350) / 450) })
        if (!intro) { pickables.forEach((m) => m.scale.setScalar(1)); labels.forEach((l) => { l.material.opacity = 1 }) }
      } else if (!calm && current) {
        current.scale.setScalar(1 + 0.16 * Math.sin(now / 260))
      }
      controls.update()
      renderer.render(scene, camera)
      frame = requestAnimationFrame(loop)
    }
    loop()

    return () => {
      cancelAnimationFrame(frame)
      api.current = null
      cam.current = { pos: camera.position.clone(), target: controls.target.clone(), k, touched }
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
  }, [map, scheme])

  useEffect(() => {
    api.current?.select(selected)
  }, [selected])

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
