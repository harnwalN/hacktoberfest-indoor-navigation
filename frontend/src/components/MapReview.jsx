import { useEffect, useMemo, useState } from 'react'
import { Check, GitMerge, ImagePlus, Plus, Save, Trash2, X } from 'lucide-react'
import { api, detail, frameUrl } from '../api'

const TYPES = ['room', 'intersection', 'hallway', 'entrance', 'stairs', 'elevator', 'exit', 'restroom', 'landmark']
const VERTICALS = ['none', 'stairs_up', 'stairs_down', 'elevator', 'ramp_up', 'ramp_down']
const DIRS = [0, 45, 90, 135, 180, 225, 270, 315]
const ARROWS = ['↑', '↗', '→', '↘', '↓', '↙', '←', '↖']

// Dead-reckoning schematic: walk the graph breadth-first, placing each neighbour one unit along the
// travel heading. It is not to scale (no distances), but loops and branches read clearly.
function layout(map) {
  const pos = {}
  const adj = {}
  for (const e of map.edges) {
    ;(adj[e.from_node] ||= []).push({ to: e.to_node, h: e.heading_out, e })
    ;(adj[e.to_node] ||= []).push({ to: e.from_node, h: (e.heading_in + 180) % 360, e })
  }
  const axis = map.dominant_axis_deg || 0
  let offsetX = 0
  for (const n of map.nodes) {
    if (pos[n.id]) continue
    pos[n.id] = [offsetX, 0]
    const queue = [n.id]
    while (queue.length) {
      const u = queue.shift()
      for (const { to, h } of adj[u] || []) {
        if (pos[to]) continue
        const r = ((h - axis) * Math.PI) / 180
        pos[to] = [pos[u][0] + Math.sin(r), pos[u][1] - Math.cos(r)]
        queue.push(to)
      }
    }
    offsetX = Math.max(...Object.values(pos).map((p) => p[0])) + 2
  }
  return pos
}

function Schematic({ map, selected, onSelect }) {
  const pos = useMemo(() => layout(map), [map])
  const pts = Object.values(pos)
  if (!pts.length) return null
  const xs = pts.map((p) => p[0])
  const ys = pts.map((p) => p[1])
  const pad = 0.8
  const [minX, maxX, minY, maxY] = [Math.min(...xs) - pad, Math.max(...xs) + pad, Math.min(...ys) - pad, Math.max(...ys) + pad]
  return (
    <svg className="schematic" viewBox={`${minX} ${minY} ${maxX - minX} ${maxY - minY}`} preserveAspectRatio="xMidYMid meet">
      {map.edges.map((e) => {
        const a = pos[e.from_node]
        const b = pos[e.to_node]
        if (!a || !b) return null
        return <line key={e.id} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} className={`edge ${e.vertical !== 'none' ? 'vertical' : ''}`} />
      })}
      {map.nodes.map((n) => {
        const p = pos[n.id]
        if (!p) return null
        return (
          <g key={n.id} onClick={() => onSelect(n.id)} className={`snode ${n.type} ${selected === n.id ? 'sel' : ''} ${n.verified ? 'verified' : ''}`}>
            <circle cx={p[0]} cy={p[1]} r={0.16} />
            <text x={p[0] + 0.22} y={p[1] + 0.07}>
              {n.label.length > 22 ? `${n.label.slice(0, 21)}…` : n.label}
            </text>
          </g>
        )
      })}
    </svg>
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

  const load = () => api.getMap(building).then((m) => { setMap(m); setDirty(false) }).catch((e) => setError(detail(e)))
  useEffect(() => {
    load()
  }, [building])

  if (!map) return <div className="card">Loading…</div>
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
      <section className="card">
        <div className="row spread">
          <h2>Building map</h2>
          <div className="row">
            <label className="ghost file-btn">
              <ImagePlus size={16} /> Add photo
              <input type="file" accept="image/*" hidden onChange={(e) => addPhoto(e.target.files[0])} />
            </label>
            <button className="primary" disabled={!dirty || busy} onClick={save}>
              <Save size={16} /> Save
            </button>
          </div>
        </div>
        <p className="muted">
          {map.nodes.length} places · {map.edges.length} connections · {map.nodes.filter((n) => n.verified).length} verified
          {map.processed_with?.model ? ` · labeled by ${map.processed_with.model}` : ''}
        </p>
        {error && <div className="banner error" onClick={() => setError('')}>{error}</div>}
        <Schematic map={map} selected={selected} onSelect={setSelected} />
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
                  <td>
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
                {n.type}
                {n.verified ? ' · ✓' : ''}
                {n.soft ? ' · blurry' : ''}
              </small>
            </button>
          ))}
        </div>
      </section>
    </>
  )
}
