// Motion constants for the Map tab (2D schematic, 3D view, reports, edit cards). Shared timings live here so the
// JS tweens and the CSS in styles/motion-map.css agree. Nothing in this file touches the DOM at import time.
import { reducedMotion } from '../components/Motion'
import { isLowEnd } from './tokens.js'

export const CAMERA_TWEEN_MS = 900 // 3D: top-down to orbit angle
export const ZOOM_TWEEN_MS = 250 // 2D: + / − / Fit
export const GLOW_MS = 1200 // edit card highlight after a place is picked
export const FLOOR_SLIDE_MS = 320
export const PLANE_STAGGER_MS = 160 // 3D: one floor slab after another
export const NODE_STAGGER_MS = 22

export const ease = (t) => 1 - Math.pow(1 - Math.min(1, Math.max(0, t)), 3)
export const easeInOut = (t) => {
  const x = Math.min(1, Math.max(0, t))
  return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2
}

// True when the Map tab should show no movement at all: reduced motion, or a low-end device.
export function stillMode() {
  return reducedMotion() || isLowEnd()
}
