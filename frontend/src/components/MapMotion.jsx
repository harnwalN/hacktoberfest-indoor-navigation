import { motion } from 'motion/react'
import { stillMode } from '../motion/map'

// Sliding highlight for segmented controls (2D/3D, floors, report filters). The pill is one element moved between
// buttons by layoutId, so it glides from the old choice to the new one. Still mode: it jumps (duration 0).
export function SlidePill({ id, tone = '' }) {
  const still = stillMode()
  return (
    <motion.span
      layoutId={id}
      className={`slide-pill ${tone}`.trim()}
      aria-hidden="true"
      transition={still ? { duration: 0 } : { type: 'spring', stiffness: 520, damping: 42, mass: 0.7 }}
    />
  )
}

// Segmented button: the active one carries the sliding pill behind its label. Same role, name and aria-pressed as before.
export function SegButton({ id, active, className = '', onClick, children, ...rest }) {
  return (
    <button type="button" className={`${className} has-pill ${active ? 'active' : ''}`.trim()} aria-pressed={active} onClick={onClick} {...rest}>
      {active && <SlidePill id={id} />}
      {children}
    </button>
  )
}

