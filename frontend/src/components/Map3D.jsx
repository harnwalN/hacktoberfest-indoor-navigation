import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

// 3D view of the solved building map: every place at its estimated position in metres, floors stacked,
// stairs/elevators as vertical links. Useful for judging how well the walks line up.
const FLOOR_H = 6 // metres between floors (exaggerated a little for readability)
const COLORS = { place: 0xc9d2e3, key: 0x5b83ff, vertical: 0xf2c230, minor: 0x677189, selected: 0xff7a6e }

const isMinor = (n) =>
  !n.verified && !(n.sign_texts || []).length && ['hallway', 'intersection', 'landmark'].includes(n.type)
const kind = (n) =>
  isMinor(n) ? 'minor' : n.type === 'stairs' || n.type === 'elevator' ? 'vertical'
    : ['entrance', 'exit', 'restroom'].includes(n.type) ? 'key' : 'place'

function labelSprite(text, color = '#eef2f8') {
  const c = document.createElement('canvas')
  const ctx = c.getContext('2d')
  const font = '600 44px "Atkinson Hyperlegible", system-ui, sans-serif'
  ctx.font = font
  const w = Math.ceil(ctx.measureText(text).width) + 24
  c.width = w
  c.height = 64
  ctx.font = font
  ctx.fillStyle = 'rgba(10,18,34,0.72)'
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

  useEffect(() => {
    const el = host.current
    const nodes = map.nodes.filter((n) => n.x != null && n.y != null)
    if (!el || !nodes.length) return undefined

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
        new THREE.MeshBasicMaterial({ color: 0x2357f2, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false }),
      )
      slab.rotation.x = -Math.PI / 2
      slab.position.set((minX + maxX) / 2, (f - f0) * FLOOR_H - 0.3, (minZ + maxZ) / 2)
      scene.add(slab)
      const edge = new THREE.LineSegments(new THREE.EdgesGeometry(slab.geometry), new THREE.LineBasicMaterial({ color: 0x5b83ff, transparent: true, opacity: 0.35 }))
      edge.rotation.copy(slab.rotation)
      edge.position.copy(slab.position)
      scene.add(edge)
      const tag = labelSprite(`Floor ${f}`, '#f2c230')
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
        new THREE.LineBasicMaterial({ color: vertical ? COLORS.vertical : 0x93a0b8, transparent: true, opacity: vertical ? 0.95 : 0.6 }),
      )
      scene.add(line)
    }

    // places
    const pickables = []
    for (const n of nodes) {
      const k = kind(n)
      const r = k === 'minor' ? 0.35 : 0.8
      const geo = k === 'vertical' ? new THREE.BoxGeometry(r * 1.8, r * 1.8, r * 1.8) : new THREE.SphereGeometry(r, 20, 14)
      const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: n.id === selected ? COLORS.selected : COLORS[k] }))
      mesh.position.copy(pos[n.id])
      mesh.userData.id = n.id
      scene.add(mesh)
      pickables.push(mesh)
      if ((n.verified && k !== 'minor') || n.id === selected) {
        const s = labelSprite(n.label.length > 26 ? `${n.label.slice(0, 25)}…` : n.label)
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
  }, [map, selected])

  return <div ref={host} className="map3d" aria-label="3D view of the building map" />
}
