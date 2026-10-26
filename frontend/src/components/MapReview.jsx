import { useEffect, useMemo, useRef, useState } from 'react'
import { Check, GitMerge, ImagePlus, Link2, Plus, Save, Trash2, X } from 'lucide-react'
import { api, detail, frameUrl } from '../api'

const TYPES = ['room', 'intersection', 'hallway', 'entrance', 'stairs', 'elevator', 'exit', 'restroom', 'landmark']
const VERTICALS = ['none', 'stairs_up', 'stairs_down', 'elevator', 'ramp_up', 'ramp_down']
const DIRS = [0, 45, 90, 135, 180, 225, 270, 315]
const ARROWS = ['↑', '↗', '→', '↘', '↓', '↙', '←', '↖']

// Dead-reckoning schematic of ONE floor: walk that floor's graph breadth-first, placing each neighbour
// one unit along the travel heading. Not to scale (no distances), but loops and branches read clearly.
// Connections to other floors (stairs/elevators) are drawn as "↕ floor N" badges instead of lines.
const STEP = 1.8 // spacing between connected places
const MIN_GAP = 1.4 // places never closer than this

// Dead-reckoning seed (one step along each travel heading), then a relaxation pass that keeps
// corridors near their direction while pushing places apart. Not to scale: the graph has no distances.
function layout(nodes, edges, axis) {
  const pos = {}
  const adj = {}
  const dirs = []
  for (const e of edges) {
    ;(adj[e.from_node] ||= []).push({ to: e.to_node, h: e.heading_out })
    ;(adj[e.to_node] ||= []).push({ to: e.from_node, h: (e.heading_in + 180) % 360 })
    const r = ((e.heading_out - axis) * Math.PI) / 180
    dirs.push([e.from_node, e.to_node, Math.sin(r), -Math.cos(r)])
  }
  let offsetX = 0
  for (const n of nodes) {
    if (pos[n.id]) continue
    pos[n.id] = [offsetX, 0]
    const piece = [n.id]
    const queue = [n.id]
    while (queue.length) {
      const u = queue.shift()
      for (const { to, h } of adj[u] || []) {
        if (pos[to]) continue
        const r = ((h - axis) * Math.PI) / 180
        pos[to] = [pos[u][0] + STEP * Math.sin(r), pos[u][1] - STEP * Math.cos(r)]
        queue.push(to)
        piece.push(to)
      }
    }
    const inPiece = new Set(piece)
    for (let it = 0; it < 240; it++) {
      for (const [a, b, dx, dy] of dirs) {
        if (!inPiece.has(a) || !pos[b]) continue
        const tx = pos[a][0] + STEP * dx - pos[b][0]
        const ty = pos[a][1] + STEP * dy - pos[b][1]
        pos[b][0] += tx * 0.08; pos[b][1] += ty * 0.08
        pos[a][0] -= tx * 0.08; pos[a][1] -= ty * 0.08
      }
      for (let i = 0; i < piece.length; i++) {
        for (let j = i + 1; j < piece.length; j++) {
          const p = pos[piece[i]]
          const q = pos[piece[j]]
          let dx = q[0] - p[0]
          let dy = q[1] - p[1]
          let d = Math.hypot(dx, dy)
          if (d >= MIN_GAP) continue
          if (d < 1e-3) { dx = 0.7; dy = 0.7; d = 1 }
          const push = ((MIN_GAP - d) / d) * 0.5
          p[0] -= dx * push; p[1] -= dy * push
          q[0] += dx * push; q[1] += dy * push
        }
      }
    }
    const minX = Math.min(...piece.map((id) => pos[id][0]))
    for (const id of piece) pos[id][0] += offsetX - minX
    offsetX = Math.max(...piece.map((id) => pos[id][0])) + 4
  }
  return pos
}

// Minor waypoints (walk ends, unnamed hallway points) render as quiet dots without labels.
const isMinor = (n) =>
  !n.verified && !(n.sign_texts || []).length && ['hallway', 'intersection', 'landmark'].includes(n.type)
const glyphClass = (n) =>
  isMinor(n) ? 'minor'
    : n.type === 'stairs' || n.type === 'elevator' ? 'vertical'
      : ['entrance', 'exit', 'restroom'].includes(n.type) ? 'key' : 'place'

const LABEL_H = 0.42
const labelW = (text) => 0.2 * Math.min(text.length, 22) + 0.2

function Schematic({ map, floor, floors, onFloor, selected, onSelect }) {
  const floorOf = Object.fromEntries(map.nodes.map((n) => [n.id, n.floor]))
  const nodes = map.nodes.filter((n) => n.floor === floor)
  const sameFloor = map.edges.filter((e) => floorOf[e.from_node] === floor && floorOf[e.to_node] === floor)
  const links = {}
  for (const e of map.edges) {
    const [a, b] = [floorOf[e.from_node], floorOf[e.to_node]]
    if (a === b) continue
    if (a === floor) (links[e.from_node] ||= new Set()).add(b)
    if (b === floor) (links[e.to_node] ||= new Set()).add(a)
  }
  const pos = useMemo(() => layout(nodes, sameFloor, map.dominant_axis_deg || 0), [map, floor])
  const [view, setView] = useState(null)
  const drag = useRef(null)
  const svgRef = useRef(null)

  const pts = Object.values(pos)
  const bounds = pts.length
    ? [Math.min(...pts.map((p) => p[0])) - 3.2, Math.min(...pts.map((p) => p[1])) - 1.2,
       Math.max(...pts.map((p) => p[0])) + 4.5, Math.max(...pts.map((p) => p[1])) + 1.2]
    : [0, 0, 10, 6]
  const fit = { x: bounds[0], y: bounds[1], w: bounds[2] - bounds[0], h: bounds[3] - bounds[1] }
  useEffect(() => setView(null), [floor, map])
  const v = view || fit

  // Greedy label placement: important places first, skip a label rather than overlap another.
  const labels = useMemo(() => {
    const placed = []
    const out = {}
    const order = [...nodes].sort((a, b) =>
      (b.id === selected) - (a.id === selected) || isMinor(a) - isMinor(b) || b.verified - a.verified)
    for (const n of order) {
      if (isMinor(n) && n.id !== selected) continue
      const p = pos[n.id]
      if (!p) continue
      const text = n.label.length > 22 ? `${n.label.slice(0, 21)}…` : n.label
      const w = labelW(text)
      for (const side of ['right', 'left', 'below']) {
        const x = side === 'right' ? p[0] + 0.32 : side === 'left' ? p[0] - 0.32 - w : p[0] - w / 2
        const y = side === 'below' ? p[1] + 0.32 : p[1] - LABEL_H / 2
        const box = [x, y, x + w, y + LABEL_H]
        const hit = placed.some((q) => box[0] < q[2] && box[2] > q[0] && box[1] < q[3] && box[3] > q[1])
          || nodes.some((m) => m.id !== n.id && pos[m.id] && pos[m.id][0] > box[0] - 0.15 && pos[m.id][0] < box[2] + 0.15
            && pos[m.id][1] > box[1] - 0.15 && pos[m.id][1] < box[3] + 0.15)
        if (!hit) {
          placed.push(box)
          out[n.id] = { text, x: side === 'left' ? x + w : side === 'below' ? p[0] : x, y: y + LABEL_H * 0.72,
                        anchor: side === 'left' ? 'end' : side === 'below' ? 'middle' : 'start' }
          break
        }
      }
    }
    return out
  }, [pos, selected, floor])

  const toSvg = (e) => {
    const r = svgRef.current.getBoundingClientRect()
    return [v.x + ((e.clientX - r.left) / r.width) * v.w, v.y + ((e.clientY - r.top) / r.height) * v.h]
  }
  const zoom = (factor, center) => {
    const [cx, cy] = center || [v.x + v.w / 2, v.y + v.h / 2]
    const w = Math.min(Math.max(v.w * factor, 3), fit.w * 3)
    const h = (w / v.w) * v.h
    setView({ x: cx - ((cx - v.x) / v.w) * w, y: cy - ((cy - v.y) / v.h) * h, w, h })
  }
  const onWheel = (e) => { e.preventDefault(); zoom(e.deltaY > 0 ? 1.15 : 0.87, toSvg(e)) }
  useEffect(() => {
    const el = svgRef.current
    if (!el) return
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  })
  const onDown = (e) => { drag.current = { sx: e.clientX, sy: e.clientY, v, moved: false } }
  const onMove = (e) => {
    const d = drag.current
    if (!d) return
    const r = svgRef.current.getBoundingClientRect()
    const dx = ((e.clientX - d.sx) / r.width) * d.v.w
    const dy = ((e.clientY - d.sy) / r.height) * d.v.h
    if (Math.abs(e.clientX - d.sx) + Math.abs(e.clientY - d.sy) > 4) d.moved = true
    setView({ ...d.v, x: d.v.x - dx, y: d.v.y - dy })
  }
  const onUp = () => { setTimeout(() => (drag.current = null), 0) }
  const pick = (id) => { if (!drag.current?.moved) onSelect(id) }

  const selEdges = new Set(sameFloor.filter((e) => e.from_node === selected || e.to_node === selected).map((e) => e.id))

  return (
    <div className="map-stage">
      {floors.length > 1 && (
        <div className="floor-panel" role="group" aria-label="Floor">
          {[...floors].reverse().map((f) => (
            <button key={f} className={f === floor ? 'active' : ''} onClick={() => onFloor(f)} aria-pressed={f === floor}>
              {f}
            </button>
          ))}
        </div>
      )}
      <div className="map-tools">
        <button className="icon-btn" aria-label="Zoom in" onClick={() => zoom(0.75)}>+</button>
        <button className="icon-btn" aria-label="Zoom out" onClick={() => zoom(1.33)}>−</button>
        <button className="icon-btn fit" aria-label="Fit floor" onClick={() => setView(null)}>Fit</button>
      </div>
      {!nodes.length ? (
        <p className="map-empty">Nothing mapped on floor {floor} yet.</p>
      ) : (
        <svg ref={svgRef} className="schematic" viewBox={`${v.x} ${v.y} ${v.w} ${v.h}`} preserveAspectRatio="xMidYMid meet"
          onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerLeave={onUp}>
          {sameFloor.map((e) => {
            const a = pos[e.from_node]
            const b = pos[e.to_node]
            if (!a || !b) return null
            return <line key={e.id} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]}
              className={`edge ${e.vertical !== 'none' ? 'vertical' : ''} ${selEdges.has(e.id) ? 'active' : ''}`} />
          })}
          {nodes.map((n) => {
            const p = pos[n.id]
            if (!p) return null
            const other = links[n.id] ? [...links[n.id]].sort((x, y) => x - y) : []
            const cls = glyphClass(n)
            const r = cls === 'minor' ? 0.09 : 0.2
            return (
              <g key={n.id} className={`snode ${cls} ${selected === n.id ? 'sel' : ''}`} onClick={() => pick(n.id)}>
                <title>{`${n.label} (floor ${n.floor})`}</title>
                <circle cx={p[0]} cy={p[1]} r={0.5} className="hit" />
                {cls === 'vertical'
                  ? <rect x={p[0] - r} y={p[1] - r} width={r * 2} height={r * 2} rx={0.06} />
                  : <circle cx={p[0]} cy={p[1]} r={r} />}
                {other.length > 0 && (
                  <text x={p[0]} y={p[1] + 0.075} className="floor-mark" textAnchor="middle">{other.length === 1 ? other[0] : '↕'}</text>
                )}
              </g>
            )
          })}
          {Object.entries(labels).map(([id, l]) => (
            <text key={id} x={l.x} y={l.y} textAnchor={l.anchor} className={`slabel ${id === selected ? 'sel' : ''}`}
              onClick={() => pick(id)}>
              {l.text}
            </text>
          ))}
        </svg>
      )}
      <ul className="map-legend">
        <li><i className="place" /> Room</li>
        <li><i className="key" /> Entrance, exit or restroom</li>
        <li><i className="vertical" /> Stairs or elevator (number = floor it reaches)</li>
        <li><i className="minor" /> Waypoint</li>
      </ul>
    </div>
  )
}

function DirSelect({ value, onChange }) {
  const nearest = DIRS.reduce((best, d) => (Math.abs(((value - d + 540) % 360) - 180) < Math.abs(((value - best + 540) % 360) - 180) ? d : best), 0)
  return (
    <select value={nearest} onChange={(e) => onChange(Number(e.target.value))} title={`${value}°`}>
      {DIRS.map((d, k) => (
        <option key={d} value={d}>
          {ARROWS[k]} {d}°
        </option>
      ))}
    </select>
  )
}

export default function MapReview({ building, onChanged }) {
  const [map, setMap] = useState(null)
  const [dirty, setDirty] = useState(false)
  const [selected, setSelected] = useState(null)
  const [mergeWith, setMergeWith] = useState('')
  const [newEdge, setNewEdge] = useState({ from_node: '', to_node: '', heading_out: 0, vertical: 'none' })
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [floor, setFloor] = useState(null)
  const [note, setNote] = useState('')

  const load = () => api.getMap(building).then((m) => { setMap(m); setDirty(false) }).catch((e) => setError(detail(e)))
  useEffect(() => {
    load()
  }, [building])

  if (!map) return <div className="card">Loading…</div>
  const floors = [...new Set(map.nodes.map((n) => n.floor))].sort((a, b) => a - b)
  const shownFloor = floors.includes(floor) ? floor : floors[0] ?? 0
  const byId = Object.fromEntries(map.nodes.map((n) => [n.id, n]))

  const patchNode = (id, patch) => {
    setMap({ ...map, nodes: map.nodes.map((n) => (n.id === id ? { ...n, ...patch } : n)) })
    setDirty(true)
  }
  const patchEdge = (id, patch) => {
    setMap({ ...map, edges: map.edges.map((e) => (e.id === id ? { ...e, ...patch, source: 'manual' } : e)) })
    setDirty(true)
  }
  const deleteNode = (id) => {
    setMap({ ...map, nodes: map.nodes.filter((n) => n.id !== id), edges: map.edges.filter((e) => e.from_node !== id && e.to_node !== id) })
    setSelected(null)
    setDirty(true)
  }
  const deleteEdge = (id) => {
    setMap({ ...map, edges: map.edges.filter((e) => e.id !== id) })
    setDirty(true)
  }
  const addEdge = () => {
    const { from_node, to_node, heading_out, vertical } = newEdge
    if (!from_node || !to_node || from_node === to_node) return
    const id = `e${Date.now().toString(36)}`
    setMap({ ...map, edges: [...map.edges, { id, from_node, to_node, heading_out, heading_in: heading_out, vertical, frames_fwd: [], frames_rev: null, source: 'manual', verified: true }] })
    setDirty(true)
  }
  const save = async () => {
    setBusy(true)
    try {
      setMap(await api.saveMap(building, map))
      setDirty(false)
      onChanged?.()
    } catch (e) {
      setError(detail(e))
    } finally {
      setBusy(false)
    }
  }
  const act = async (fn) => {
    if (dirty && !confirm('Discard unsaved edits?')) return
    try {
      setMap(await fn())
      setDirty(false)
      onChanged?.()
    } catch (e) {
      setError(detail(e))
    }
  }
  const addPhoto = async (file) => {
    if (!file) return
    setBusy(true)
    try {
      const r = await api.addPhoto(building, file, '')
      await load()
      setSelected(r.node.id)
    } catch (e) {
      setError(detail(e))
    } finally {
      setBusy(false)
    }
  }

  const sel = selected ? byId[selected] : null
  const selEdges = sel ? map.edges.filter((e) => e.from_node === sel.id || e.to_node === sel.id) : []

  return (
    <>
      <section className="map-panel">
        <header className="map-head">
          <div>
            <h2>Building map</h2>
            <p className="muted">
              {map.nodes.length} places, {map.edges.length} connections, {map.nodes.filter((n) => n.verified).length} verified
            </p>
          </div>
          <div className="map-actions">
            <button className="ghost" disabled={busy} title="Merge places seen in several videos into one map"
              onClick={async () => {
                setBusy(true)
                try {
                  const { job_id } = await api.consolidate(building)
                  for (;;) {
                    const j = await api.job(job_id)
                    if (j.status === 'done' || j.status === 'error') {
                      setError(j.status === 'error' ? j.error : '')
                      setNote(j.log.filter((l) => /Same|Aligned|Consolidated|Floor link|placed/.test(l)).map((l) => l.slice(11)).join('\n'))
                      break
                    }
                    await new Promise((r) => setTimeout(r, 1000))
                  }
                  await load()
                  onChanged?.()
                } finally {
                  setBusy(false)
                }
              }}>
              <Link2 size={16} /> {busy ? 'Linking…' : 'Link videos'}
            </button>
            <label className="ghost file-btn">
              <ImagePlus size={16} /> Add photo
              <input type="file" accept="image/*" hidden onChange={(e) => addPhoto(e.target.files[0])} />
            </label>
            <button className="primary" disabled={!dirty || busy} onClick={save}>
              <Save size={16} /> {dirty ? 'Save changes' : 'Saved'}
            </button>
          </div>
        </header>
        {error && <p className="notice error" onClick={() => setError('')}>{error}</p>}
        {note && <pre className="log" onClick={() => setNote('')}>{note}</pre>}
        <Schematic map={map} floor={shownFloor} floors={floors} onFloor={setFloor} selected={selected}
          onSelect={(id) => { setSelected(id); setFloor(byId[id]?.floor) }} />
      </section>

      {map.suggested_merges.length > 0 && (
        <section className="card">
          <h3>
            <GitMerge size={18} /> Same place? <small className="muted">loop-closure candidates</small>
          </h3>
          {map.suggested_merges.map((s) => (
            <div key={s.id} className="merge">
              <div className="pair">
                <figure>
                  <img src={frameUrl(building, byId[s.keep]?.frame_id)} alt="" />
                  <figcaption>{byId[s.keep]?.label}</figcaption>
                </figure>
                <figure>
                  <img src={frameUrl(building, byId[s.remove]?.frame_id)} alt="" />
                  <figcaption>{byId[s.remove]?.label}</figcaption>
                </figure>
              </div>
              <p>
                <b>{Math.round(s.confidence * 100)}%</b> · {s.reason}
              </p>
              <div className="row">
                <button className="primary" onClick={() => act(() => api.resolveMerge(building, s.id, true))}>
                  <Check size={16} /> Merge
                </button>
                <button onClick={() => act(() => api.resolveMerge(building, s.id, false))}>
                  <X size={16} /> Different places
                </button>
              </div>
            </div>
          ))}
        </section>
      )}

      {sel && (
        <section className="card node-editor">
          <div className="row spread">
            <h3>{sel.label}</h3>
            <button className="ghost danger" onClick={() => deleteNode(sel.id)}>
              <Trash2 size={16} /> Delete
            </button>
          </div>
          {sel.frame_id && <img className="node-photo" src={frameUrl(building, sel.frame_id)} alt="" />}
          <label className="field">
            <span>Label</span>
            <input value={sel.label} onChange={(e) => patchNode(sel.id, { label: e.target.value })} />
          </label>
          <label className="field">
            <span>Landmark phrase (used in "Turn left at …")</span>
            <input value={sel.description} onChange={(e) => patchNode(sel.id, { description: e.target.value })} />
          </label>
          <div className="row">
            <label className="field">
              <span>Type</span>
              <select value={sel.type} onChange={(e) => patchNode(sel.id, { type: e.target.value })}>
                {TYPES.map((t) => <option key={t}>{t}</option>)}
              </select>
            </label>
            <label className="field">
              <span>Floor</span>
              <input type="number" value={sel.floor} onChange={(e) => patchNode(sel.id, { floor: Number(e.target.value) })} />
            </label>
            <label className="check">
              <input type="checkbox" checked={sel.verified} onChange={(e) => patchNode(sel.id, { verified: e.target.checked })} />
              Verified (shown in dropdowns)
            </label>
          </div>
          {sel.texts.length > 0 && <p className="muted">Gemma read: {sel.texts.join(' · ')}</p>}
          <div className="row">
            <select value={mergeWith} onChange={(e) => setMergeWith(e.target.value)}>
              <option value="">Merge another node into this one…</option>
              {map.nodes.filter((n) => n.id !== sel.id).map((n) => <option key={n.id} value={n.id}>{n.label}</option>)}
            </select>
            <button disabled={!mergeWith} onClick={() => act(() => api.mergeNodes(building, sel.id, mergeWith)).then(() => setMergeWith(''))}>
              <GitMerge size={16} /> Merge
            </button>
          </div>
          <h4>Connections</h4>
          <table className="edges">
            <thead>
              <tr><th>From</th><th>To</th><th>Leave</th><th>Arrive</th><th>Level</th><th /></tr>
            </thead>
            <tbody>
              {selEdges.map((e) => (
                <tr key={e.id}>
                  <td>{byId[e.from_node]?.label}</td>
                  <td>{byId[e.to_node]?.label}</td>
                  <td><DirSelect value={e.heading_out} onChange={(v) => patchEdge(e.id, { heading_out: v })} /></td>
                  <td><DirSelect value={e.heading_in} onChange={(v) => patchEdge(e.id, { heading_in: v })} /></td>
                  <td title={e.inferred_floor_change ? 'Floor change inferred from room numbers; no stairs seen' : ''}>
                    {e.inferred_floor_change && <span className="pill paused">check</span>}
                    <select value={e.vertical} onChange={(ev) => patchEdge(e.id, { vertical: ev.target.value })}>
                      {VERTICALS.map((v) => <option key={v}>{v}</option>)}
                    </select>
                  </td>
                  <td><button className="ghost danger" onClick={() => deleteEdge(e.id)}><Trash2 size={14} /></button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <section className="card">
        <h3>
          <Plus size={18} /> Add connection
        </h3>
        <p className="muted">Directions are in the building frame (↑ = the main hallway axis).</p>
        <div className="row wrap">
          <select value={newEdge.from_node} onChange={(e) => setNewEdge({ ...newEdge, from_node: e.target.value })}>
            <option value="">From…</option>
            {map.nodes.map((n) => <option key={n.id} value={n.id}>{n.label}</option>)}
          </select>
          <DirSelect value={newEdge.heading_out} onChange={(v) => setNewEdge({ ...newEdge, heading_out: v })} />
          <select value={newEdge.to_node} onChange={(e) => setNewEdge({ ...newEdge, to_node: e.target.value })}>
            <option value="">To…</option>
            {map.nodes.map((n) => <option key={n.id} value={n.id}>{n.label}</option>)}
          </select>
          <select value={newEdge.vertical} onChange={(e) => setNewEdge({ ...newEdge, vertical: e.target.value })}>
            {VERTICALS.map((v) => <option key={v}>{v}</option>)}
          </select>
          <button onClick={addEdge}>Add</button>
        </div>
      </section>

      <section className="card">
        <h3>All places</h3>
        <div className="node-grid">
          {map.nodes.map((n) => (
            <button key={n.id} className={`node-card ${selected === n.id ? 'sel' : ''}`} onClick={() => setSelected(n.id)}>
              {n.frame_id ? <img src={frameUrl(building, n.frame_id)} loading="lazy" alt="" /> : <div className="nophoto" />}
              <span>{n.label}</span>
              <small>
                {n.type}, floor {n.floor}
                {n.verified ? ', verified' : ''}
                {n.soft ? ', blurry photo' : ''}
              </small>
            </button>
          ))}
        </div>
      </section>
    </>
  )
}
