import NumberFlow from '@number-flow/react'

// An animated number. NumberFlow draws its digits in a shadow root, so the page text would lose them: the plain
// value goes alongside in a visually hidden span, and the animated copy is hidden from assistive tech.
export default function FlowNumber({ value, suffix = '' }) {
  return (
    <>
      <span aria-hidden="true"><NumberFlow value={value} suffix={suffix} /></span>
      <span className="sr-only">{`${value}${suffix}`}</span>
    </>
  )
}
