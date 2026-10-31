import { ArrowUpDown, Ban, DoorClosed, DoorOpen, Footprints, LogIn, LogOut, MapPin, Toilet, TrendingUp } from 'lucide-react'

// ---- Place types: one classifier shared by the review and navigation screens ----
// Kinds map to the CSS modifier `place-chip--{kind}`: room | exit | noexit | stairs | hallway | landmark.
const NO_EXIT = /\bno[\s_-]*exit\b/i
const MINOR_LABEL = /^(hallway|corridor)\b|^doorway\b|^junction\b|^intersection\b/i

// A "No Exit" sign is never an exit, whatever type the backend filed it under.
export function placeKind(node) {
  const label = node?.label || ''
  const type = node?.type || ''
  if (NO_EXIT.test(label)) return 'noexit'
  if (type === 'exit') return 'exit'
  if (type === 'room') return 'room'
  if (type === 'stairs' || type === 'elevator' || node?.vertical) return 'stairs'
  if (type === 'hallway' || type === 'intersection' || MINOR_LABEL.test(label)) return 'hallway'
  return 'landmark' // entrance, restroom, landmark, anything new
}

// Hallway point / Hallway end / Doorway: minor waypoints rather than destinations.
export const isWaypoint = (node) => placeKind(node) === 'hallway' && (MINOR_LABEL.test(node?.label || '') || !node?.verified)

// Quiet dot on the maps: waypoints and unverified, unnamed hallway/landmark nodes. A No Exit sign is never quiet.
export function isMinorNode(n) {
  if (placeKind(n) === 'noexit') return false
  if (MINOR_LABEL.test(n?.label || '')) return true
  return !n?.verified && !(n?.sign_texts || []).length && ['hallway', 'intersection', 'landmark'].includes(n?.type)
}

const KIND_TEXT = { room: 'Room', exit: 'Exit', noexit: 'No exit', stairs: 'Stairs', hallway: 'Hallway', landmark: 'Landmark' }

export function placeLabel(node) {
  const kind = placeKind(node)
  const type = node?.type
  if (kind === 'stairs') return type === 'elevator' || node?.vertical === 'elevator' ? 'Elevator' : 'Stairs'
  if (kind === 'hallway' && /^doorway/i.test(node?.label || '')) return 'Doorway'
  if (kind === 'hallway' && /\bend\b/i.test(node?.label || '')) return 'Hallway end'
  if (kind === 'hallway') return 'Hallway point'
  if (kind === 'landmark') return type === 'entrance' ? 'Entrance' : type === 'restroom' ? 'Restroom' : 'Landmark'
  return KIND_TEXT[kind]
}

export function PlaceIcon({ node, size = 14 }) {
  const kind = placeKind(node)
  const type = node?.type
  const props = { size, 'aria-hidden': true, focusable: 'false' }
  if (kind === 'noexit') return <Ban {...props} />
  if (kind === 'exit') return <LogOut {...props} />
  if (kind === 'room') return <DoorClosed {...props} />
  if (kind === 'stairs') return type === 'elevator' || node?.vertical === 'elevator' ? <ArrowUpDown {...props} /> : <TrendingUp {...props} />
  if (kind === 'hallway') return /^doorway/i.test(node?.label || '') ? <DoorOpen {...props} /> : <Footprints {...props} />
  if (type === 'entrance') return <LogIn {...props} />
  if (type === 'restroom') return <Toilet {...props} />
  return <MapPin {...props} />
}

// Icon + words (never colour alone), so No exit cannot be mistaken for Exit.
export function PlaceChip({ node, className = '' }) {
  return (
    <span className={`place-chip place-chip--${placeKind(node)} ${className}`.trim()}>
      <PlaceIcon node={node} />
      <span>{placeLabel(node)}</span>
    </span>
  )
}

// Wayfinding pictograms. Turns are bent arrows (straight approach, then the turn) like building and
// road signage, so the shape itself says "go this far, then turn". Left turns mirror the right ones.
const STROKE = { fill: 'none', strokeWidth: 9, strokeLinecap: 'round', strokeLinejoin: 'round' }

function Head({ x, y, angle }) {
  return <path d="M0 -13 L13 4 L-13 4 Z" transform={`translate(${x} ${y}) rotate(${angle})`} className="glyph-fill" />
}

function Straight() {
  return (
    <g>
      <path d="M50 88 V26" {...STROKE} className="glyph-stroke" />
      <Head x={50} y={20} angle={0} />
    </g>
  )
}

function Turn() {
  return (
    <g>
      <path d="M36 88 V50 Q36 38 48 38 H70" {...STROKE} className="glyph-stroke" />
      <Head x={76} y={38} angle={90} />
    </g>
  )
}

function Slight() {
  return (
    <g>
      <path d="M40 88 V58 L66 32" {...STROKE} className="glyph-stroke" />
      <Head x={70} y={28} angle={45} />
    </g>
  )
}

function UTurn() {
  return (
    <g>
      <path d="M64 88 V44 Q64 24 48 24 Q32 24 32 44 V64" {...STROKE} className="glyph-stroke" />
      <Head x={32} y={70} angle={180} />
    </g>
  )
}

function Stairs({ up }) {
  return (
    <g>
      <path d="M14 84 H32 V68 H50 V52 H68 V36 H86" {...STROKE} className="glyph-stroke" />
      <path d={up ? 'M26 48 L50 24' : 'M50 24 L26 48'} {...STROKE} strokeWidth={7} className="glyph-stroke" />
      <Head x={up ? 53 : 23} y={up ? 21 : 51} angle={up ? 45 : 225} />
    </g>
  )
}

function Elevator() {
  return (
    <g>
      <rect x="24" y="16" width="52" height="70" rx="8" {...STROKE} strokeWidth={7} className="glyph-stroke" />
      <path d="M50 30 L62 44 H38 Z M50 72 L62 58 H38 Z" className="glyph-fill" />
    </g>
  )
}

function Arrive() {
  return (
    <g>
      <path d="M50 90 C50 90 22 58 22 40 A28 28 0 0 1 78 40 C78 58 50 90 50 90 Z" className="glyph-fill" />
      <circle cx="50" cy="40" r="10" className="glyph-hole" />
    </g>
  )
}

function Start() {
  return (
    <g>
      <circle cx="50" cy="74" r="12" className="glyph-fill" />
      <path d="M50 58 V28" {...STROKE} className="glyph-stroke" />
      <Head x={50} y={22} angle={0} />
    </g>
  )
}

const GLYPHS = {
  go_straight: [Straight, false], start: [Start, false], destination_reached: [Arrive, false],
  turn_right: [Turn, false], turn_left: [Turn, true], slight_right: [Slight, false], slight_left: [Slight, true],
  u_turn: [UTurn, false], elevator: [Elevator, false],
}

export function glyphKind(action, vertical) {
  if (action === 'destination_reached') return 'arrive'
  if (action?.startsWith('stairs') || action === 'elevator' || (vertical && vertical !== 'none')) return 'vertical'
  return 'turn'
}

export default function SignGlyph({ action, vertical, title }) {
  let body
  if (action === 'stairs_up' || action === 'stairs_down') body = <Stairs up={action === 'stairs_up'} />
  else if (action === 'ramp_up' || action === 'ramp_down') body = <Slight />
  else {
    const [Glyph, mirror] = GLYPHS[action] || GLYPHS.go_straight
    body = <g transform={mirror ? 'translate(100 0) scale(-1 1)' : undefined}><Glyph /></g>
  }
  return (
    <svg className={`glyph ${glyphKind(action, vertical)}`} viewBox="0 0 100 100" role="img" aria-label={title || action}>
      {body}
    </svg>
  )
}
