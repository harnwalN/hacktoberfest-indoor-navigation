// Shared motion tokens. Import from here, never inline magic numbers in a component.
// Rules live in ./README.md. Short version: animate transform and opacity (plus height for Collapse) only.
import { reducedMotion } from '../components/Motion.jsx'

// Seconds. "fast" for hover/press/colour, "base" for things that appear, "slow" for big surfaces and sheets.
export const duration = {
  instant: 0,
  fast: 0.14,
  base: 0.22,
  slow: 0.36,
  // Tab switch: old page leaves, then new page arrives. Total 0.28 s (budget 300 ms). Do not lengthen.
  tabExit: 0.08,
  tabEnter: 0.2,
}

// Cubic-bezier arrays (motion accepts these directly). Same curves as the CSS custom properties in styles.css.
export const ease = {
  out: [0.16, 1, 0.3, 1], // --ease-out: things that arrive
  standard: [0.2, 0.7, 0.2, 1], // --ease: hover, colour, small moves
  in: [0.4, 0, 1, 1], // things that leave
  pop: [0.34, 1.4, 0.64, 1], // one mild overshoot, celebrations only (--ease-pop)
}

// Spring presets. Use for taps and anything interruptible (a dragged or re-targeted pill).
export const spring = {
  snappy: { type: 'spring', stiffness: 520, damping: 38, mass: 0.6 }, // press feedback, nav pill
  gentle: { type: 'spring', stiffness: 260, damping: 30 }, // sheets, toasts settling
  bouncy: { type: 'spring', stiffness: 380, damping: 18 }, // rare celebratory bits only
}

// Delay (seconds) for item i of a list. Capped so a long list never waits for ever.
export function stagger(i, step = 0.05, max = 0.3) {
  return Math.min(Math.max(i, 0) * step, max)
}

// Transition for a preset, collapsed to instant when the user prefers reduced motion.
export function transitionFor(preset = { duration: duration.base, ease: ease.out }) {
  return reducedMotion() ? { duration: 0 } : preset
}

// Common variants. Use as <motion.div variants={fadeUp} initial="hidden" animate="show" />.
export const variants = {
  fadeUp: {
    hidden: { opacity: 0, y: 12 },
    show: { opacity: 1, y: 0, transition: { duration: duration.base, ease: ease.out } },
  },
  scaleIn: {
    hidden: { opacity: 0, scale: 0.96 },
    show: { opacity: 1, scale: 1, transition: { duration: duration.base, ease: ease.out } },
    exit: { opacity: 0, scale: 0.96, transition: { duration: duration.fast, ease: ease.in } },
  },
  // For items in a list; pair with stagger(i) in the parent's transition if wanted.
  listItem: {
    hidden: { opacity: 0, y: 8 },
    show: { opacity: 1, y: 0, transition: { duration: duration.base, ease: ease.out } },
    exit: { opacity: 0, transition: { duration: duration.fast, ease: ease.in } },
  },
  // Height is the one layout property allowed. Use the <Collapse> component rather than this directly.
  collapse: {
    closed: { height: 0, opacity: 0, transition: { duration: duration.base, ease: ease.in } },
    open: { height: 'auto', opacity: 1, transition: { duration: duration.slow, ease: ease.out } },
  },
  // Tab content: slides in from the side the new tab is on. dir is 1 (moving right) or -1 (moving left).
  tabPage: {
    enter: (dir) => ({ x: dir * 24, opacity: 0 }),
    center: { x: 0, opacity: 1, transition: { duration: duration.tabEnter, ease: ease.out } },
    exit: (dir) => ({ x: dir * -24, opacity: 0, transition: { duration: duration.tabExit, ease: ease.in } }),
  },
  // Same as tabPage without movement, for users who prefer reduced motion (App picks it via useReducedMotion).
  tabFade: {
    enter: { opacity: 0 },
    center: { opacity: 1, transition: { duration: duration.tabEnter, ease: ease.out } },
    exit: { opacity: 0, transition: { duration: duration.tabExit, ease: ease.in } },
  },
}

// True on phones, older laptops and anyone who asked for reduced motion. Use it to skip heavy effects
// (confetti bursts, long loops, big blurs). Light fades and taps are fine on every device.
export function isLowEnd() {
  if (typeof window === 'undefined' || typeof navigator === 'undefined') return false
  if (reducedMotion()) return true
  const cores = navigator.hardwareConcurrency
  const mem = navigator.deviceMemory
  return (typeof cores === 'number' && cores > 0 && cores <= 4) || (typeof mem === 'number' && mem > 0 && mem <= 4)
}
