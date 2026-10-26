import { useEffect, useMemo, useRef, useState } from 'react'
import { ArrowUpDown, ChevronLeft, ChevronRight, X } from 'lucide-react'
import { api, detail } from '../api'
import SignGlyph, { glyphKind } from './ArrowOverlay'

const TYPE_ORDER = ['entrance', 'room', 'restroom', 'elevator', 'stairs', 'exit', 'landmark', 'intersection', 'hallway']
const GROUP_NAMES = {
  entrance: 'Entrances', room: 'Rooms', restroom: 'Restrooms', elevator: 'Elevators', stairs: 'Stairs',
  exit: 'Exits', landmark: 'Landmarks', intersection: 'Junctions', hallway: 'Hallways',
}
// Short, glanceable headline for the sign; the full sentence sits underneath.
const HEADLINE = {
  start: 'Start here', go_straight: 'Go straight', slight_left: 'Bear left', slight_right: 'Bear right',
  turn_left: 'Turn left', turn_right: 'Turn right', u_turn: 'Turn around', stairs_up: 'Stairs up',
  stairs_down: 'Stairs down', elevator: 'Take the elevator', ramp_up: 'Ramp up', ramp_down: 'Ramp down',
  destination_reached: 'You have arrived',
}

function PlaceSelect({ id, label, value, onChange, nodes }) {
  const groups = useMemo(() => {
    const g = {}
    for (const n of nodes) (g[n.type] ||= []).push(n)
    return TYPE_ORDER.filter((t) => g[t]).map((t) => [t, g[t].sort((a, b) => a.label.localeCompare(b.label))])
  }, [nodes])
  return (
    <label className="trip-field" htmlFor={id}>
      <span>{label}</span>
      <select id={id} value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="">Choose a place</option>
        {groups.map(([type, list]) => (
          <optgroup key={type} label={GROUP_NAMES[type] || type}>
            {list.map((n) => (
              <option key={n.id} value={n.id}>
                {n.label}, floor {n.floor}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
    </label>
  )
}

function RouteStrip({ steps, index, onPick }) {
  return (
    <ol className="route-strip" aria-label="Route progress">
      {steps.map((s, k) => {
        const vertical = glyphKind(s.action, s.vertical) === 'vertical'
        const state = k < index ? 'done' : k === index ? 'now' : 'next'
        const floorChange = k > 0 && steps[k - 1].floor !== s.floor
        return (
          <li key={k} className={`${state} ${vertical ? 'vertical' : ''}`}>
            <button onClick={() => onPick(k)} aria-label={`Step ${k + 1}: ${s.label}`} aria-current={k === index ? 'step' : undefined}>
              <i />
            </button>
            {floorChange && <small>F{s.floor}</small>}
          </li>
        )
      })}
    </ol>
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

  // Shareable directions: #/navigate?b=<building>&from=<place>&to=<place> opens straight into the route.
  useEffect(() => {
    if (!map) return
    const q = new URLSearchParams(window.location.hash.split('?')[1] || '')
    const from = q.get('from')
    const to = q.get('to')
    if (from && to) {
      setStart(from)
      setTarget(to)
      api.route(building, from, to).then((r) => { setSteps(r.steps); setI(Number(q.get('step') || 0)) })
        .catch((e) => setError(detail(e)))
    }
  }, [map])

  const verified = (map?.nodes || []).filter((n) => n.verified)
  const choices = showAll || verified.length < 2 ? map?.nodes || [] : verified
  const floors = new Set((map?.nodes || []).map((n) => n.floor)).size

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
      if (e.key === 'Escape') setSteps(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [steps])

  if (!map) return <div className="quiet-state">Loading the building map…</div>
  if (map.nodes.length < 2)
    return (
      <div className="quiet-state">
        <h2>No map yet</h2>
        <p>Record or upload a walk through the building in Build, and directions will appear here.</p>
      </div>
    )

  if (!steps) {
    return (
      <section className="planner">
        <h1>Where are you going?</h1>
        <div className="trip">
          <div className="trip-line" aria-hidden="true">
            <i className="from" />
            <span />
            <i className="to" />
          </div>
          <div className="trip-fields">
            <PlaceSelect id="from" label="From" value={start} onChange={setStart} nodes={choices} />
            <PlaceSelect id="to" label="To" value={target} onChange={setTarget} nodes={choices} />
          </div>
          <button className="icon-btn swap" aria-label="Swap start and destination" onClick={() => { setStart(target); setTarget(start) }}>
            <ArrowUpDown size={18} />
          </button>
        </div>
        <label className="check">
          <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} />
          Show unverified places too
        </label>
        {error && <p className="notice error" role="alert">{error}</p>}
        <button className="go" disabled={!start || !target || start === target} onClick={go}>
          Start directions
        </button>
        <p className="planner-meta">
          {choices.length} places on {floors} {floors === 1 ? 'floor' : 'floors'}, mapped from {map.videos.length} walks
        </p>
      </section>
    )
  }

  const s = steps[i]
  const destination = steps[steps.length - 1]
  const kind = glyphKind(s.action, s.vertical)
  const detailText = s.instruction_text
  const onTouchEnd = (e) => {
    if (touch.current == null) return
    const dx = e.changedTouches[0].clientX - touch.current
    if (dx < -50) setI((x) => Math.min(x + 1, steps.length - 1))
    if (dx > 50) setI((x) => Math.max(x - 1, 0))
    touch.current = null
  }

  return (
    <section className="trip-view">
      <header className="trip-head">
        <button className="icon-btn" aria-label="End directions" onClick={() => setSteps(null)}>
          <X size={20} />
        </button>
        <p>
          To <b>{destination.label}</b>
        </p>
        <span className="step-count">
          {i + 1} of {steps.length}
        </span>
      </header>

      <figure className="scene" onTouchStart={(e) => (touch.current = e.touches[0].clientX)} onTouchEnd={onTouchEnd}>
        {s.frame_url ? <img key={s.frame_url} src={s.frame_url} alt={`View at ${s.label}`} /> : <div className="no-photo">No photo for this spot</div>}
        <span className="floor-chip">Floor {s.floor}</span>
        {s.view_note && <figcaption>{s.view_note}</figcaption>}
      </figure>

      <div className={`sign ${kind}`} aria-live="polite">
        <SignGlyph action={s.action} vertical={s.vertical} title={HEADLINE[s.action]} />
        <div className="sign-copy">
          <h2>{HEADLINE[s.action] || 'Continue'}</h2>
          <p>{detailText}</p>
        </div>
      </div>

      <RouteStrip steps={steps} index={i} onPick={setI} />

      <nav className="step-nav">
        <button disabled={i === 0} onClick={() => setI(i - 1)}>
          <ChevronLeft size={20} /> Back
        </button>
        {i < steps.length - 1 ? (
          <button className="next" onClick={() => setI(i + 1)}>
            Next <ChevronRight size={20} />
          </button>
        ) : (
          <button className="next" onClick={() => setSteps(null)}>
            Done
          </button>
        )}
      </nav>
    </section>
  )
}
