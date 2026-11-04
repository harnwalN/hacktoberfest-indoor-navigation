import { useEffect, useState } from 'react'

// True only after `on` has stayed true for `ms`. Use it for loading placeholders: a fast load never shows one
// (no 20 ms flash), a slow load does. Turning `on` off hides it at once.
//   const showSkel = useDelayed(!loaded, 180)
export default function useDelayed(on, ms = 180) {
  const [show, setShow] = useState(false)
  useEffect(() => {
    if (!on) {
      setShow(false)
      return undefined
    }
    const t = setTimeout(() => setShow(true), ms)
    return () => clearTimeout(t)
  }, [on, ms])
  return on && show
}
