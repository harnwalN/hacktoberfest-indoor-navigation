import { forwardRef } from 'react'
import { motion, useReducedMotion } from 'motion/react'
import { spring } from './tokens.js'

// Press-down feel for BUTTONS only. Two ways to use it:
//   <PressButton className="primary" onClick={save}>Save</PressButton>
//   <motion.button {...usePressMotion(0.9)} className="icon-btn">...</motion.button>   (call the hook in your component)
// Press scale is 0.96 by default (pass pressScale, e.g. 0.9 for small icons).
// Disabled buttons do not react. Under reduced motion there is no press scale at all (usePressMotion returns {}).
//
// RULE: never put whileTap on a non-focusable element (a span, div or svg wrapper). motion's press gesture then
// gives that element tabindex="0", which adds a tab stop that Soo does not have. Use a <button> or CSS :active instead.
export function usePressMotion(scale = 0.96) {
  const reduce = useReducedMotion()
  if (reduce) return {}
  return { whileTap: { scale }, transition: spring.snappy }
}

const PressButton = forwardRef(function PressButton({ disabled, pressScale = 0.96, type = 'button', children, ...rest }, ref) {
  const press = usePressMotion(pressScale)
  return (
    <motion.button ref={ref} type={type} disabled={disabled} {...(disabled ? {} : press)} {...rest}>
      {children}
    </motion.button>
  )
})

export default PressButton
