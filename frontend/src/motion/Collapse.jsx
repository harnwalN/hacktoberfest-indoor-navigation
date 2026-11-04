import { motion, useReducedMotion } from 'motion/react'
import { variants } from './tokens.js'

// Height-auto disclosure. Children stay mounted when closed (state is kept). Closed content is zero height,
// inert (not focusable) and hidden from assistive tech.
//   <Collapse open={showMore}> ...details... </Collapse>
// The trigger stays outside and should carry aria-expanded and aria-controls (pass the same id here).
const INSTANT = { duration: 0 }

export default function Collapse({ open, children, className = '', id, ...rest }) {
  const reduce = useReducedMotion()
  const motionVariants = reduce
    ? {
        closed: { height: 0, opacity: 0, transition: INSTANT },
        open: { height: 'auto', opacity: 1, transition: INSTANT },
      }
    : variants.collapse

  return (
    <motion.div
      id={id}
      className={`collapse ${className}`.trim()}
      initial={false}
      animate={open ? 'open' : 'closed'}
      variants={motionVariants}
      style={{ overflow: 'hidden' }}
      inert={open ? undefined : ''}
      aria-hidden={open ? undefined : 'true'}
      {...rest}
    >
      {children}
    </motion.div>
  )
}
