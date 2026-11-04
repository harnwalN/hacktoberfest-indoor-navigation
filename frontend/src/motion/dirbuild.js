// Motion constants for the Directions and Build tabs. Kept apart from the shared tokens so they can be pointed
// at src/motion/tokens.js later. Durations are seconds (motion v14 units). Reduced motion: use pick().

export const EASE_OUT = [0.16, 1, 0.3, 1] // same curve as --ease-out in styles.css

export const INSTANT = { duration: 0 }
// Returns the instant transition when reduced motion is on, otherwise the given one.
export const pick = (reduce, t) => (reduce ? INSTANT : t)

// Swap: the two place values cross over with a light spring.
export const SWAP_SPRING = { type: 'spring', stiffness: 380, damping: 30, mass: 0.9 }
// Sliding highlight in the place list.
export const HIGHLIGHT = { type: 'spring', stiffness: 520, damping: 42 }
// Staggered row/item entrance: index-based delay, capped so long lists do not wait.
export const rowIn = (i, step = 0.045) => ({ duration: 0.35, ease: EASE_OUT, delay: Math.min(i, 6) * step })
export const COLLAPSE = { duration: 0.28, ease: EASE_OUT }
export const FADE = { duration: 0.22, ease: 'easeOut' }
export const CARD_IN = { duration: 0.32, ease: EASE_OUT }
