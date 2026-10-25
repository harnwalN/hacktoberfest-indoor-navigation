import { useEffect, useMemo, useRef, useState } from 'react'
import { ArrowLeft, ArrowRight, Footprints, MapPin, RotateCcw } from 'lucide-react'
import { api, detail } from '../api'
import ArrowOverlay from './ArrowOverlay'

const TYPE_ORDER = ['entrance', 'room', 'restroom', 'elevator', 'stairs', 'exit', 'landmark', 'intersection', 'hallway']

function NodeSelect({ label, value, onChange, nodes }) {
  const groups = useMemo(() => {
    const g = {}
    for (const n of nodes) (g[n.type] ||= []).push(n)
    return TYPE_ORDER.filter((t) => g[t]).map((t) => [t, g[t].sort((a, b) => a.label.localeCompare(b.label))])
  }, [nodes])
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="">Select…</option>
        {groups.map(([type, list]) => (
          <optgroup key={type} label={type}>
            {list.map((n) => (
              <option key={n.id} value={n.id}>
                {n.label}
                {n.floor ? ` · floor ${n.floor > 0 ? '+' : ''}${n.floor}` : ''}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
    </label>
  )
}

export default function Navigator({ building }) {
  const [map, setMap] = useState(null)
  const [start, setStart] = useState('')
  const [target, setTarget] = useState('')
  const [steps, setSteps] = useState(null)
  const [i, setI] = useState(0)
  const [error, setError] = useState('')
  const [showAll, setShowAll] = useState(false)
  const touch = useRef(null)

  useEffect(() => {
    setSteps(null)
    api.getMap(building).then(setMap).catch((e) => setError(detail(e)))
  }, [building])

  const verified = (map?.nodes || []).filter((n) => n.verified)
  const choices = showAll || verified.length < 2 ? map?.nodes || [] : verified

  const go = async () => {
    setError('')
    try {
      const r = await api.route(building, start, target)
      setSteps(r.steps)
      setI(0)
    } catch (e) {
      setError(detail(e))
    }
  }

  // Preload the next photos so stepping forward is instant on a phone connection.
  useEffect(() => {
    steps?.slice(i + 1, i + 3).forEach((s) => s.frame_url && (new Image().src = s.frame_url))
  }, [steps, i])

  useEffect(() => {
    const onKey = (e) => {
      if (!steps) return
      if (e.key === 'ArrowRight') setI((x) => Math.min(x + 1, steps.length - 1))
      if (e.key === 'ArrowLeft') setI((x) => Math.max(x - 1, 0))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [steps])

  if (!map) return <div className="card">Loading map…</div>
  if (map.nodes.length < 2)
    return <div className="card empty">This building has no map yet. Upload a walkthrough in Build.</div>

  if (!steps) {
    return (
      <section className="card nav-setup">
        <h2>Where to?</h2>
        <NodeSelect label="Select your current position" value={start} onChange={setStart} nodes={choices} />
        <button className="swap" title="Swap" onClick={() => { setStart(target); setTarget(start) }}>
          ⇅
        </button>
        <NodeSelect label="Select destination" value={target} onChange={setTarget} nodes={choices} />
        <label className="check">
          <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} />
          Include unverified places
        </label>
        {error && <div className="banner error">{error}</div>}
        <button className="primary big" disabled={!start || !target} onClick={go}>
          <Footprints size={20} /> Start directions
        </button>
      </section>
    )
  }

  const s = steps[i]
  const onTouchEnd = (e) => {
    if (touch.current == null) return
    const dx = e.changedTouches[0].clientX - touch.current
    if (dx < -50) setI((x) => Math.min(x + 1, steps.length - 1))
    if (dx > 50) setI((x) => Math.max(x - 1, 0))
    touch.current = null
  }

  return (
    <section className="nav-step">
      <div className="step-head">
        <span>
          Step {i + 1} of {steps.length}
        </span>
        <button className="ghost" onClick={() => setSteps(null)}>
          <RotateCcw size={16} /> New route
        </button>
      </div>
      <div className="progress">
        {steps.map((_, k) => (
          <i key={k} className={k < i ? 'done' : k === i ? 'now' : ''} onClick={() => setI(k)} />
        ))}
      </div>
      <div className="photo" onTouchStart={(e) => (touch.current = e.touches[0].clientX)} onTouchEnd={onTouchEnd}>
        {s.frame_url ? <img src={s.frame_url} alt={s.label} /> : <div className="nophoto">No photo</div>}
        <ArrowOverlay action={s.action} vertical={s.vertical} />
        {s.view_note && <div className="view-note">{s.view_note}</div>}
        {s.floor !== 0 && <div className="floor-badge">Floor {s.floor > 0 ? `+${s.floor}` : s.floor}</div>}
      </div>
      <p className="instruction">{s.instruction_text}</p>
      <p className="place">
        <MapPin size={14} /> {s.label}
      </p>
      <div className="step-buttons">
        <button disabled={i === 0} onClick={() => setI(i - 1)}>
          <ArrowLeft size={20} /> Previous
        </button>
        <button className="primary" disabled={i === steps.length - 1} onClick={() => setI(i + 1)}>
          Next <ArrowRight size={20} />
        </button>
      </div>
    </section>
  )
}
