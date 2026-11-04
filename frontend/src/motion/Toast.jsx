import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { CircleAlert, CircleCheck, Info, X } from 'lucide-react'
import { spring, transitionFor } from './tokens.js'

// Toasts: small status messages that clear themselves. Usage:
//   const toast = useToast()
//   toast.success('Saved'); toast.error('Upload failed'); toast.info('Map is updating')
// The region is always in the DOM with aria-live="polite", so screen readers announce every message.
// Desktop: slides in from the top. Phone: rises from the bottom, above the tab bar.

const ToastContext = createContext(null)
const MAX_VISIBLE = 3
const AUTO_MS = { success: 4000, info: 4000, error: 6000 }
const ICON = { success: CircleCheck, error: CircleAlert, info: Info }

export function ToastProvider({ children }) {
  const [items, setItems] = useState([])
  const nextId = useRef(0)
  const timers = useRef(new Map())

  const dismiss = useCallback((id) => {
    clearTimeout(timers.current.get(id))
    timers.current.delete(id)
    setItems((list) => list.filter((t) => t.id !== id))
  }, [])

  const push = useCallback(
    (kind, message) => {
      const id = ++nextId.current
      setItems((list) => [...list, { id, kind, message }].slice(-MAX_VISIBLE))
      timers.current.set(id, setTimeout(() => dismiss(id), AUTO_MS[kind]))
      return id
    },
    [dismiss],
  )

  // Clear pending timers when the provider unmounts.
  useEffect(() => {
    const t = timers.current
    return () => t.forEach((h) => clearTimeout(h))
  }, [])

  const api = useMemo(
    () => ({
      success: (m) => push('success', m),
      error: (m) => push('error', m),
      info: (m) => push('info', m),
      dismiss,
    }),
    [push, dismiss],
  )

  return (
    <ToastContext.Provider value={api}>
      {children}
      <Toaster items={items} onDismiss={dismiss} />
    </ToastContext.Provider>
  )
}

// Returns { success, error, info, dismiss }. Outside a provider it throws, so a missing provider is caught early.
export function useToast() {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error('useToast needs <ToastProvider> above it')
  return ctx
}

function isDesktop() {
  return typeof window !== 'undefined' && !!window.matchMedia?.('(min-width: 1160px)').matches
}

function Toaster({ items, onDismiss }) {
  // Direction follows the layout the CSS chose: top on desktop, bottom on phone.
  const [desktop, setDesktop] = useState(isDesktop)
  useEffect(() => {
    const mq = window.matchMedia?.('(min-width: 1160px)')
    if (!mq) return undefined
    const on = () => setDesktop(mq.matches)
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [])
  const from = desktop ? -18 : 18

  return (
    <div className="toast-region" aria-live="polite" aria-atomic="false">
      <AnimatePresence initial={false}>
        {items.map((t) => {
          const Icon = ICON[t.kind] || Info
          return (
            <motion.div
              key={t.id}
              layout="position"
              className={`toast toast-${t.kind}`}
              initial={{ opacity: 0, y: from, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1, transition: transitionFor(spring.gentle) }}
              exit={{ opacity: 0, y: from / 2, scale: 0.98, transition: transitionFor({ duration: 0.14 }) }}
            >
              <Icon size={18} aria-hidden="true" className="toast-icon" />
              <span className="toast-text">{t.message}</span>
              <button type="button" className="ghost icon-btn toast-close" onClick={() => onDismiss(t.id)} aria-label="Dismiss message">
                <X size={16} aria-hidden="true" />
              </button>
            </motion.div>
          )
        })}
      </AnimatePresence>
    </div>
  )
}
