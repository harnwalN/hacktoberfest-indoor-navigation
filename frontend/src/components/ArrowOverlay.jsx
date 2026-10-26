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
