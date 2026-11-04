// Offline use: the service worker (public/sw.js) and the online state. The saved copy is the building map and route photos.
import { useEffect, useState } from 'react'

// Production builds only: the dev server's modules change on every save and are not worth caching.
export function registerOffline() {
  if (!import.meta.env.PROD || !('serviceWorker' in navigator)) return
  const register = () => navigator.serviceWorker.register('/sw.js').catch(() => {})
  if (document.readyState === 'complete') register()
  else window.addEventListener('load', register, { once: true })
}

// navigator.onLine is the browser's view. A reachable browser with a dead server still counts as online here.
export function useOnline() {
  const [online, setOnline] = useState(() => navigator.onLine !== false)
  useEffect(() => {
    const up = () => setOnline(true)
    const down = () => setOnline(false)
    window.addEventListener('online', up)
    window.addEventListener('offline', down)
    return () => {
      window.removeEventListener('online', up)
      window.removeEventListener('offline', down)
    }
  }, [])
  return online
}

// Asks the service worker to keep the photos of the route on screen, so they show with no signal.
export function keepFramesOffline(urls) {
  const list = (urls || []).filter((u) => typeof u === 'string' && u.startsWith('/bfiles/'))
  if (!list.length || !('serviceWorker' in navigator)) return
  navigator.serviceWorker.ready
    .then((reg) => reg.active?.postMessage({ type: 'cache-frames', urls: list }))
    .catch(() => {})
}
