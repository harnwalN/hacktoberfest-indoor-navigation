import { useState } from 'react'

// A photo the server provides. Its URL is signed and expires after a day, and a revoke makes it 404, so a load error
// must never leave a broken image. On the first error this shows the placeholder and calls onStale(src) so the view
// can fetch fresh URLs. A URL that fails again stays on the placeholder: the caller limits refetches to one per view.
export default function ServerPhoto({ src, alt, className, loading, placeholder, onStale }) {
  const [failed, setFailed] = useState('')
  if (!src || failed === src) return placeholder
  return (
    <img
      src={src}
      alt={alt}
      className={className}
      loading={loading}
      onError={() => {
        setFailed(src)
        onStale?.(src)
      }}
    />
  )
}
