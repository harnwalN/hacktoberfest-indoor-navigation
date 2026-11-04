import { useId, useRef, useState } from 'react'
import { Printer } from 'lucide-react'
import Poster from './Poster'

// The address a poster's code opens: this site, on the building's directions. Nothing is stored in it.
export function buildingLink(building) {
  return `${window.location.origin}${window.location.pathname}#/navigate?b=${encodeURIComponent(building)}`
}

// Entrance poster for the building. Keyed by building in App, so switching buildings starts with a closed poster.
export default function PosterPanel({ building, name }) {
  const uid = useId()
  const [open, setOpen] = useState(false)
  const btn = useRef(null)

  return (
    <section className="card poster-panel" aria-labelledby={`${uid}-title`}>
      <h3 id={`${uid}-title`}>
        <Printer size={18} aria-hidden="true" /> Entrance poster
      </h3>
      <p className="muted">
        Print a code for the entrance. Anyone who scans it opens directions for {name || 'this building'} on their phone. No sign-in is needed.
      </p>
      <button type="button" className="primary" ref={btn} onClick={() => setOpen(true)}>
        <Printer size={16} aria-hidden="true" /> Make a poster
      </button>
      {open && <Poster name={name} link={buildingLink(building)} onClose={() => setOpen(false)} returnFocusTo={btn} />}
    </section>
  )
}
