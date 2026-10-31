import { useEffect, useMemo, useRef, useState } from 'react'
import {
  ArrowRight, ArrowUp, ArrowUpDown, ArrowUpLeft, ArrowUpRight, ChevronLeft, ChevronRight, CornerUpLeft, CornerUpRight, Footprints,
  MapPin, RefreshCw, Search, TrendingDown, TrendingUp, Undo2, X,
} from 'lucide-react'
import { api, detail } from '../api'
import ErrorBoundary from './ErrorBoundary'
import ServerPhoto from './ServerPhoto'
import SignGlyph, { PlaceChip, glyphKind, isWaypoint, placeKind } from './ArrowOverlay'
import { Confetti, EmptyArt, PlannerArt, PlannerSkeleton } from './Motion'

const TYPE_RANK = { entrance: 0, room: 1, restroom: 2, elevator: 3, stairs: 4, exit: 5, landmark: 6 }
// Short, glanceable headline for the sign; the full sentence sits underneath.
const HEADLINE = {
  start: 'Start here', go_straight: 'Go straight', slight_left: 'Bear left', slight_right: 'Bear right',
  turn_left: 'Turn left', turn_right: 'Turn right', u_turn: 'Turn around', stairs_up: 'Stairs up',
  stairs_down: 'Stairs down', elevator: 'Take the elevator', ramp_up: 'Ramp up', ramp_down: 'Ramp down',
  destination_reached: 'You have arrived',
}
const STEP_ICON = {
  start: Footprints, go_straight: ArrowUp, slight_left: ArrowUpLeft, slight_right: ArrowUpRight, turn_left: CornerUpLeft,
  turn_right: CornerUpRight, u_turn: Undo2, stairs_up: TrendingUp, stairs_down: TrendingDown, elevator: ArrowUpDown,
  ramp_up: TrendingUp, ramp_down: TrendingDown, destination_reached: MapPin,
}

// Heading of each action in degrees (0 = straight on); the sign's arrow swings from the previous step's heading.
const ANGLE = { slight_right: 45, turn_right: 90, u_turn: 180, slight_left: -45, turn_left: -90 }
const headingOf = (s) => ANGLE[s?.action] || 0

const naturalSort = (a, b) => (a.label || '').localeCompare(b.label || '', undefined, { numeric: true, sensitivity: 'base' })

// ---------- distances ----------
// "about 5 m": coarse on purpose, the numbers are estimated from walking time.
export function fmtDistance(m) {
  if (m == null || !Number.isFinite(m) || m <= 0) return ''
  if (m < 2) return 'a few steps'
  if (m < 10) return `about ${Math.round(m)} m`
  return `about ${Math.round(m / 5) * 5} m`
}

function legDistance(map, a, b, byId) {
  const e = (map.edges || []).find((x) => (x.from_node === a && x.to_node === b) || (x.from_node === b && x.to_node === a))
  if (e?.length_m > 0 && (e.vertical || 'none') === 'none') return e.length_m
  const p = byId[a]
  const q = byId[b]
  if (p && q && p.floor === q.floor && p.x != null && p.y != null && q.x != null && q.y != null) return Math.hypot(p.x - q.x, p.y - q.y)
  return null
}

function floorChangeText(step, next) {
  const f = next.floor
  const up = f > step.floor
  switch (step.vertical) {
    case 'stairs_up': case 'stairs_down': return `Take the stairs ${step.vertical === 'stairs_up' ? 'up' : 'down'} to floor ${f}`
    case 'elevator': return `Take the elevator ${up ? 'up' : 'down'} to floor ${f}`
    case 'ramp_up': case 'ramp_down': return `Take the ramp ${step.vertical === 'ramp_up' ? 'up' : 'down'} to floor ${f}`
    default: return `Go ${up ? 'up' : 'down'} to floor ${f}`
  }
}

// Turn the API's steps into what the person sees: trivial hallway-point pass-throughs fold into the step before
// them (their walking distance adds up), each step gets a rounded distance to the next one, and floor changes are
// called out explicitly.
export function buildSteps(raw, map) {
  const byId = Object.fromEntries((map?.nodes || []).map((n) => [n.id, n]))
  const out = []
  raw.forEach((s, k) => {
    const next = raw[k + 1]
    const dist = next && s.floor === next.floor ? legDistance(map || {}, s.node_id, next.node_id, byId) : null
    const node = byId[s.node_id] || { label: s.label, type: 'hallway' }
    const prev = raw[k - 1]
    const trivial = k > 0 && next && s.action === 'go_straight' && (!s.vertical || s.vertical === 'none')
      && placeKind(node) === 'hallway' && s.floor === prev.floor && s.floor === next.floor && out.length > 0
    if (trivial) {
      const last = out[out.length - 1]
      last.meters = last.meters == null && dist == null ? null : (last.meters || 0) + (dist || 0)
      last.passed.push(s.label)
      return
    }
    out.push({ ...s, node, meters: dist, passed: [], instruction_text: s.instruction_text || '' })
  })
  out.forEach((s, j) => {
    const next = out[j + 1]
    s.distance = fmtDistance(s.meters)
    s.floorChange = next && next.floor !== s.floor ? floorChangeText(s, next) : ''
  })
  return out
}

// ---------- destination picker ----------
function PlacePicker({ id, label, value, onChange, nodes, allNodes }) {
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const box = useRef(null)
  const selected = allNodes.find((n) => n.id === value)

  const groups = useMemo(() => {
    const q = query.trim().toLowerCase()
    const hit = (n) =>
      !q || [n.label, n.description, placeKind(n) === 'noexit' ? 'no exit' : n.type, ...(n.texts || [])].some((t) => (t || '').toLowerCase().includes(q))
    const byFloor = {}
    for (const n of nodes) if (hit(n)) (byFloor[n.floor] ||= []).push(n)
    return Object.keys(byFloor).map(Number).sort((a, b) => a - b).map((f) => [
      f,
      byFloor[f].sort((a, b) => (TYPE_RANK[a.type] ?? 9) - (TYPE_RANK[b.type] ?? 9) || naturalSort(a, b)),
    ])
  }, [nodes, query])
  const count = groups.reduce((t, [, l]) => t + l.length, 0)

  const pick = (n) => { onChange(n.id); setQuery(''); setOpen(false) }
  const onKeyDown = (e) => {
    if (e.key === 'Escape' && open) { e.stopPropagation(); setOpen(false) }
    if (e.key === 'Enter' && open && query.trim() && count > 0) { e.preventDefault(); pick(groups[0][1][0]) }
    if (e.key === 'ArrowDown' && open) {
      e.preventDefault()
      box.current?.querySelector('.place-option')?.focus()
    }
  }
  const onOptionKey = (e) => {
    const items = [...box.current.querySelectorAll('.place-option')]
    const at = items.indexOf(e.currentTarget)
    if (e.key === 'ArrowDown') { e.preventDefault(); items[Math.min(at + 1, items.length - 1)]?.focus() }
    if (e.key === 'ArrowUp') { e.preventDefault(); (at === 0 ? box.current.querySelector('input') : items[at - 1])?.focus() }
    if (e.key === 'Escape') { e.preventDefault(); setOpen(false); box.current.querySelector('input')?.focus() }
  }

  return (
    <div className="trip-field place-picker" ref={box}
      onBlur={(e) => { if (!e.currentTarget.contains(e.relatedTarget)) setOpen(false) }}>
      <label htmlFor={id}><span>{label}</span></label>
      {selected && !open && (
        <p className="place-picked" key={value}>
          <b>{selected.label}</b> <small>floor {selected.floor}</small> <PlaceChip node={selected} />
        </p>
      )}
      <div className="place-search">
        <Search size={16} aria-hidden="true" />
        <input id={id} type="search" autoComplete="off" value={query} placeholder={selected ? 'Search to change' : 'Search places'}
          aria-expanded={open} aria-controls={`${id}-list`}
          onFocus={() => setOpen(true)} onClick={() => setOpen(true)}
          onChange={(e) => { setQuery(e.target.value); setOpen(true) }} onKeyDown={onKeyDown} />
      </div>
      {open && (
        <div id={`${id}-list`} className="place-picker-list" role="group" aria-label={`${label} choices`}
          onMouseDown={(e) => e.preventDefault()}>
          <p className="sr-only" role="status" aria-live="polite">{count} {count === 1 ? 'place' : 'places'} found</p>
          {count === 0 && <p className="muted place-empty">No place matches "{query}". Try a room number such as 112.</p>}
          {groups.map(([f, list]) => (
            <div key={f} role="group" aria-label={`Floor ${f}`}>
              <h3 className="place-floor">Floor {f}</h3>
              <ul className="place-options">
                {list.map((n) => (
                  <li key={n.id}>
                    <button type="button" className="place-option" aria-pressed={n.id === value} onClick={() => pick(n)} onKeyDown={onOptionKey}>
                      <span>{n.label || 'Unnamed place'}</span>
                      <PlaceChip node={n} />
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ---------- directions ----------
function RouteStrip({ steps, index, onPick }) {
  const strip = useRef(null)
  useEffect(() => {
    // Scroll only the strip (not the page) so the current dot stays visible when there are many steps.
    const el = strip.current
    const dot = el?.children[index]
    if (!el || !dot) return
    el.scrollLeft = dot.offsetLeft - (el.clientWidth - dot.offsetWidth) / 2
  }, [index, steps.length])
  return (
    <ol className="route-strip" aria-label="Route progress" ref={strip}>
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

function StepList({ steps, index, onPick }) {
  return (
    <ol className="step-list" aria-label="All steps">
      {steps.map((s, k) => {
        const Icon = STEP_ICON[s.action] || ArrowUp
        return (
          <li key={k} className={`step ${k === index ? 'step--current' : ''}`} style={{ '--i': Math.min(k, 8) }}>
            <button type="button" onClick={() => onPick(k)} aria-current={k === index ? 'step' : undefined}>
              <span className="step-icon"><Icon size={22} aria-hidden="true" /></span>
              <span className="step-text">
                <b>{HEADLINE[s.action] || 'Continue'}</b> {s.instruction_text}
                {s.floorChange && <strong>{s.floorChange}</strong>}
              </span>
              {s.distance && s.action !== 'destination_reached' && <span className="step-distance">{s.distance}</span>}
            </button>
          </li>
        )
      })}
    </ol>
  )
}

const normalizeMap = (m) => ({
  ...m,
  nodes: (m?.nodes || []).map((n) => ({
    ...n, label: n.label ?? '', type: n.type || 'hallway', description: n.description ?? '',
    floor: Number.isFinite(n.floor) ? n.floor : 1, texts: n.texts || [], sign_texts: n.sign_texts || [], verified: !!n.verified,
  })),
  edges: m?.edges || [],
  videos: m?.videos || [],
})

// ---------- avoid stairs ----------
const AVOID_KEY = 'waypoint.avoidStairs'
const NO_ROUTE = 'No route found between those places.'
// The server's 400 when the only way climbs stairs. Only this answer offers the "Show route with stairs" retry.
const STEP_FREE_RE = /step-free/i

// The choice is kept on this device. Guarded: private mode can throw, and then the toggle simply starts off each time.
function savedAvoid() {
  try {
    return window.localStorage.getItem(AVOID_KEY) === '1'
  } catch {
    return false
  }
}
function saveAvoid(on) {
  try {
    window.localStorage.setItem(AVOID_KEY, on ? '1' : '0')
  } catch {
    /* private mode: the choice lasts for this page only */
  }
}

const hashQuery = () => new URLSearchParams(window.location.hash.split('?')[1] || '')

// A trip is shareable: #/navigate?b=<building>&from=<place>&to=<place>[&avoid=1] is the address of the trip on screen.
// It is written when a trip starts or changes, and dropped when the trip ends. replaceState keeps history clean.
function writeTripLink(building, trip) {
  const p = new URLSearchParams({ b: building })
  if (trip) {
    p.set('from', trip.from)
    p.set('to', trip.to)
    if (trip.avoid) p.set('avoid', '1')
  }
  try {
    window.history.replaceState(window.history.state, '', `${window.location.pathname}${window.location.search}#/navigate?${p}`)
  } catch {
    /* the trip works without the address bar; it just is not in the link */
  }
}

// The toggle sits in the planner and on the trip, so one component serves both. The helper text is the description.
function AvoidStairs({ on, onChange }) {
  return (
    <div className="avoid-stairs">
      <label className="check">
        <input id="avoid-stairs" type="checkbox" checked={on} onChange={(e) => onChange(e.target.checked)} aria-describedby="avoid-stairs-help" />
        <span>Avoid stairs</span>
      </label>
      <p id="avoid-stairs-help" className="check-help">Use elevators and ramps only</p>
    </div>
  )
}

// Same look as every other error in this screen; the stairs offer is a single button under the server's message.
function RouteError({ message, onShowStairs }) {
  return (
    <div className="notice error route-error" role="alert">
      <span>{message}</span>
      {onShowStairs && (
        <button type="button" className="ghost" onClick={onShowStairs}>Show route with stairs</button>
      )}
    </div>
  )
}

// "No stairs" only when the server says the route is step-free. "Uses stairs" only when the toggle is off and the route
// climbs stairs (with the toggle on, a stairs route is not offered, so nothing is shown for it).
function StairsBadge({ stepFree, avoid }) {
  if (stepFree === true) return <span className="route-badge is-step-free">No stairs on this route</span>
  if (stepFree === false && !avoid) return <span className="route-badge is-stairs">This route uses stairs</span>
  return null
}

const TRK0 = { i: 0, dir: 1, prev: null }

export default function Navigator({ building }) {
  return (
    <ErrorBoundary key={building} title="Directions hit a problem">
      <NavigatorInner building={building} />
    </ErrorBoundary>
  )
}

function NavigatorInner({ building }) {
  const [map, setMap] = useState(null)
  const [loadError, setLoadError] = useState('')
  const [start, setStart] = useState('')
  const [target, setTarget] = useState('')
  const [raw, setRaw] = useState(null)
  const [i, setI] = useState(0)
  const [error, setError] = useState('')
  const [routing, setRouting] = useState(false)
  const [showAll, setShowAll] = useState(false)
  const [reload, setReload] = useState(0)
  const touch = useRef(null)
  // Which way the last step change went (for the slide direction) and the step we came from. Derived while
  // rendering with the React "adjust state on prop change" pattern, so no ref is mutated during render.
  const [trk, setTrk] = useState(TRK0)
  const [swapTurn, setSwapTurn] = useState(0)
  // The trip on screen: its ends, the stairs choice it was routed with, and whether its photo URLs were already
  // refetched once. Null when no trip is shown.
  const trip = useRef(null)
  // Avoid stairs. A shared link carries its own choice (absent means off); otherwise the saved choice applies.
  const [avoid, setAvoid] = useState(() => {
    const q = hashQuery()
    return q.get('from') && q.get('to') ? q.get('avoid') === '1' : savedAvoid()
  })
  // step_free from the answer on screen (null when the server did not say).
  const [stepFree, setStepFree] = useState(null)
  // Set with the error when the server refused a stairs-free route: the request to repeat with stairs.
  const [stairsRetry, setStairsRetry] = useState(null)
  // Counts route requests. Only the newest may change the screen, so an old slow answer cannot overwrite a newer one.
  const seq = useRef(0)

  useEffect(() => {
    let live = true
    trip.current = null
    seq.current += 1
    setRaw(null); setMap(null); setStart(''); setTarget(''); setError(''); setLoadError(''); setI(0)
    setStepFree(null); setStairsRetry(null)
    api.getMap(building)
      .then((m) => live && setMap(normalizeMap(m)))
      .catch((e) => live && setLoadError(detail(e)))
    return () => { live = false }
  }, [building, reload])

  const steps = useMemo(() => (raw && map ? buildSteps(raw, map) : null), [raw, map])

  // One route request. A failed request leaves the trip on screen as it was.
  const routeTrip = (from, to, avoidOn, step = 0) => {
    const id = ++seq.current
    setRouting(true)
    setError('')
    setStairsRetry(null)
    return api.route(building, from, to, { avoidStairs: avoidOn })
      .then((r) => {
        if (id !== seq.current) return
        if (!r?.steps?.length) throw new Error(NO_ROUTE)
        trip.current = { from, to, avoid: avoidOn, refetched: false }
        setRaw(r.steps)
        setStepFree(typeof r.step_free === 'boolean' ? r.step_free : null)
        setI(step)
        writeTripLink(building, { from, to, avoid: avoidOn })
      })
      .catch((e) => {
        if (id !== seq.current) return
        const msg = detail(e)
        setError(msg)
        if (avoidOn && e?.response?.status === 400 && STEP_FREE_RE.test(msg)) setStairsRetry({ from, to, step })
      })
      .finally(() => {
        if (id === seq.current) setRouting(false)
      })
  }

  // Shareable directions: #/navigate?b=<building>&from=<place>&to=<place>[&avoid=1][&step=n] opens straight into the route.
  useEffect(() => {
    if (!map) return undefined
    const q = hashQuery()
    const from = q.get('from')
    const to = q.get('to')
    if (!from || !to) return undefined
    const n = Number.parseInt(q.get('step') || '0', 10)
    setStart(from)
    setTarget(to)
    setAvoid(q.get('avoid') === '1')
    routeTrip(from, to, q.get('avoid') === '1', Number.isFinite(n) ? Math.max(0, n) : 0)
    return undefined
  }, [map])

  // Places you can pick: hallway points/ends/doorways are waypoints, not destinations. Unverified places are
  // hidden unless asked for (or until there are enough verified ones).
  const allNodes = map?.nodes || []
  const candidates = allNodes.filter((n) => !isWaypoint(n))
  const verified = candidates.filter((n) => n.verified)
  const choices = showAll || verified.length < 2 ? candidates : verified
  const floors = new Set(allNodes.map((n) => n.floor)).size

  const go = () => routeTrip(start, target, avoid, 0)

  // Ends the trip on screen. A late answer for an ended trip is ignored (the seq bump), and so is a late photo refetch.
  const endTrip = () => {
    seq.current += 1
    trip.current = null
    setRaw(null)
    setStepFree(null)
    setRouting(false)
    setError('')
    setStairsRetry(null)
    writeTripLink(building, null)
  }

  // A photo URL in this trip failed (revoked or expired). The route is fetched again once for fresh URLs, keeping the
  // same step. A second failure in the same trip just keeps the placeholder.
  const onPhotoStale = () => {
    const t = trip.current
    if (!t || t.refetched) return
    t.refetched = true
    api.route(building, t.from, t.to, { avoidStairs: t.avoid })
      .then((r) => {
        if (trip.current === t && r?.steps?.length) setRaw(r.steps)
      })
      .catch(() => {})
  }

  const last = steps ? steps.length - 1 : 0
  const goTo = (k) => setI(Math.max(0, Math.min(k, last)))

  // The choice is kept for next time. On a trip it re-routes from the place the trip is at now (from the start when
  // that place is the destination), and the trip goes back to its first step.
  const toggleAvoid = (on) => {
    setAvoid(on)
    saveAvoid(on)
    const t = trip.current
    if (!raw || !steps || !t) {
      setError('')
      setStairsRetry(null)
      return
    }
    const here = steps[Math.min(i, last)].node_id
    routeTrip(here === t.to ? t.from : here, t.to, on, 0)
  }

  // "Show route with stairs": repeat the refused request without the stairs rule, and keep the toggle in step with it.
  const showWithStairs = () => {
    const r = stairsRetry
    if (!r) return
    setAvoid(false)
    saveAvoid(false)
    routeTrip(r.from, r.to, false, r.step)
  }

  // Preload the next photos so stepping forward is instant on a phone connection.
  useEffect(() => {
    steps?.slice(i + 1, i + 3).forEach((s) => s.frame_url && (new Image().src = s.frame_url))
  }, [steps, i])

  useEffect(() => {
    if (!raw) return undefined
    const onKey = (e) => {
      if (e.target.closest?.('input, select, textarea') || e.altKey || e.ctrlKey || e.metaKey) return
      if (e.key === 'ArrowRight') setI((x) => Math.min(x + 1, last))
      if (e.key === 'ArrowLeft') setI((x) => Math.max(x - 1, 0))
      if (e.key === 'Escape') endTrip()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [raw, last])

  if (loadError)
    return (
      <div className="quiet-state" role="alert">
        <h2>Could not load the map</h2>
        <p>{loadError}</p>
        <button onClick={() => setReload((n) => n + 1)}><RefreshCw size={16} /> Try again</button>
      </div>
    )
  if (!map)
    return (
      <div className="loading-state" role="status" aria-live="polite">
        <PlannerSkeleton />
        <p className="muted">Loading the building map…</p>
      </div>
    )
  if (candidates.length < 2)
    return (
      <div className="quiet-state">
        <EmptyArt />
        <h2>No map yet</h2>
        <p>Record or upload a walk through the building in Build, and directions will appear here.</p>
      </div>
    )

  if (!steps) {
    if (trk !== TRK0) setTrk(TRK0)
    return (
      <section className="planner">
        <PlannerArt />
        <h1>Where are you going?</h1>
        <div className="trip">
          <div className="trip-line" aria-hidden="true">
            <i className="from" />
            <span />
            <i className="to" />
          </div>
          <div className="trip-fields">
            <PlacePicker id="from" label="From" value={start} onChange={setStart} nodes={choices} allNodes={allNodes} />
            <PlacePicker id="to" label="To" value={target} onChange={setTarget} nodes={choices} allNodes={allNodes} />
          </div>
          <button className="icon-btn swap" aria-label="Swap start and destination" onClick={() => { setStart(target); setTarget(start); setSwapTurn((n) => n + 1) }}>
            <ArrowUpDown size={18} style={{ transform: `rotate(${swapTurn * 180}deg)` }} className="swap-icon" />
          </button>
        </div>
        <label className="check">
          <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} />
          Show unverified places too
        </label>
        <AvoidStairs on={avoid} onChange={toggleAvoid} />
        {error && <RouteError message={error} onShowStairs={stairsRetry ? showWithStairs : undefined} />}
        <button className={`go ${start && target && start !== target ? 'ready' : ''} ${routing ? 'busy' : ''}`} disabled={!start || !target || start === target || routing} onClick={go}>
          {routing ? 'Finding your route…' : <>Start directions <ArrowRight size={22} aria-hidden="true" /></>}
        </button>
        {start && start === target && <p className="planner-meta" role="status">Pick two different places.</p>}
        <p className="planner-meta">
          {choices.length} places on {floors} {floors === 1 ? 'floor' : 'floors'}, mapped from {map.videos.length} {map.videos.length === 1 ? 'walk' : 'walks'}
        </p>
      </section>
    )
  }

  const idx = Math.min(i, last)
  const s = steps[idx]
  const t = trk.i === idx ? trk : { i: idx, dir: idx > trk.i ? 1 : -1, prev: steps[trk.i] || null }
  if (t !== trk) setTrk(t)
  const { dir, prev } = t
  const swing = Math.max(-45, Math.min(45, Math.round((headingOf(prev) - headingOf(s)) * 0.5)))
  const floorMoved = prev && prev.floor !== s.floor ? (s.floor > prev.floor ? 'up' : 'down') : ''
  const destination = steps[last]
  const kind = glyphKind(s.action, s.vertical)
  const onTouchEnd = (e) => {
    if (touch.current == null) return
    const dx = e.changedTouches[0].clientX - touch.current
    if (dx < -50) goTo(idx + 1)
    if (dx > 50) goTo(idx - 1)
    touch.current = null
  }
  const showDistance = s.distance && s.action !== 'destination_reached'

  return (
    <section className="trip-view" style={{ '--dir': dir }}>
      <header className="trip-head">
        <button className="icon-btn" aria-label="End directions" onClick={endTrip}>
          <X size={20} />
        </button>
        <p>
          To <b>{destination.label}</b>
        </p>
        <span className="step-count">
          {idx + 1} of {steps.length}
        </span>
      </header>
      <div className="trip-summary">
        <StairsBadge stepFree={stepFree} avoid={avoid} />
        <AvoidStairs on={avoid} onChange={toggleAvoid} />
      </div>
      {error && <RouteError message={error} onShowStairs={stairsRetry ? showWithStairs : undefined} />}

      <figure className="scene" onTouchStart={(e) => (touch.current = e.touches[0].clientX)} onTouchEnd={onTouchEnd}>
        <ServerPhoto
          key={s.frame_url || 'none'}
          src={s.frame_url}
          alt={`View at ${s.label}`}
          onStale={onPhotoStale}
          placeholder={<div className="no-photo">No photo for this spot</div>}
        />
        <span className={`floor-chip ${floorMoved ? `moved moved--${floorMoved}` : ''}`} key={`fc${idx}`}>Floor {s.floor}</span>
        {s.view_note && <figcaption>{s.view_note}</figcaption>}
      </figure>

      <div className={`sign ${kind}`} role="status" aria-live="polite" aria-atomic="true">
        {kind === 'arrive' && <Confetti />}
        <SignGlyph key={`g${idx}`} action={s.action} vertical={s.vertical} title={HEADLINE[s.action]} style={{ '--swing': `${swing}deg` }} />
        <div className="sign-copy" key={`c${idx}`}>
          <span className="sr-only">Step {idx + 1} of {steps.length}. </span>
          <h2>{HEADLINE[s.action] || 'Continue'}</h2>
          <p>{s.instruction_text}</p>
          {showDistance && <p className="step-distance">Walk {s.distance}</p>}
          {s.floorChange && <p className="floor-callout"><strong>{s.floorChange}</strong></p>}
        </div>
      </div>

      <StepList steps={steps} index={idx} onPick={goTo} />
      {/* strip sits right above the controls and docks with them, so its dots are never under the sticky nav */}
      <RouteStrip steps={steps} index={idx} onPick={goTo} />

      <nav className="step-nav" aria-label="Step controls">
        <button disabled={idx === 0} onClick={() => goTo(idx - 1)}>
          <ChevronLeft size={20} /> Back
        </button>
        {idx < last ? (
          <button className="next" onClick={() => goTo(idx + 1)}>
            Next <ChevronRight size={20} />
          </button>
        ) : (
          <button className="next" onClick={endTrip}>
            Done
          </button>
        )}
      </nav>
    </section>
  )
}
