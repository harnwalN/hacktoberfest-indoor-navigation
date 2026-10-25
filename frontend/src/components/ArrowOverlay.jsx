// Direction glyph drawn over the step photo. Turns rotate a single up-arrow; stairs and arrival get
// their own glyphs so they read at a glance while walking.
const ROTATION = {
  start: 0, go_straight: 0, slight_left: -45, slight_right: 45, turn_left: -90, turn_right: 90, u_turn: 180,
}

function Arrow({ angle }) {
  return (
    <g transform={`rotate(${angle} 50 50)`}>
      <path d="M50 10 L80 44 L60 44 L60 88 L40 88 L40 44 L20 44 Z" />
    </g>
  )
}

function Stairs({ up }) {
  return (
    <g>
      <path d="M14 82 H34 V64 H52 V46 H70 V28 H88" fill="none" strokeWidth="9" strokeLinejoin="round" />
      <path d={up ? 'M30 40 L50 18 L50 32' : 'M70 70 L50 92 L50 78'} fill="none" strokeWidth="7" strokeLinecap="round" />
    </g>
  )
}

function Pin() {
  return (
    <g>
      <path d="M50 92 C50 92 20 58 20 40 A30 30 0 0 1 80 40 C80 58 50 92 50 92 Z" />
      <circle cx="50" cy="40" r="11" className="hole" />
    </g>
  )
}

export default function ArrowOverlay({ action, vertical }) {
  let glyph
  if (action === 'destination_reached') glyph = <Pin />
  else if (action === 'stairs_up' || action === 'stairs_down') glyph = <Stairs up={action === 'stairs_up'} />
  else glyph = <Arrow angle={ROTATION[action] ?? 0} />
  const kind = action === 'destination_reached' ? 'arrive' : vertical && vertical !== 'none' ? 'vertical' : 'turn'
  return (
    <svg className={`arrow-overlay ${kind}`} viewBox="0 0 100 100" aria-hidden="true">
      <circle cx="50" cy="50" r="48" className="disc" />
      {glyph}
    </svg>
  )
}
