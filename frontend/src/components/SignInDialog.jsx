import { useEffect, useRef, useState } from 'react'
import { X } from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { api, detail } from '../api'
import { duration, ease, spring, transitionFor } from '../motion/tokens.js'
import useModal from '../motion/useModal.js'

const EMAIL_OK = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

// The sign-in sheet. Email only: the server emails a one-time link. It opens in this browser and signs the user in.
// Shown by the sign-in button, by a write that needs an identity, or when a link or the session did not work.
// Opens and closes with a scrim fade and a short scale+fade of the sheet (AnimatePresence keeps it mounted while it exits).
export default function SignInDialog({ open, reason, onClose }) {
  // Keep the last reason while the sheet fades out, so its text does not blank during the exit.
  const lastReason = useRef(reason)
  if (open) lastReason.current = reason
  return (
    <AnimatePresence>
      {open && <SignInSheet key="signin" reason={lastReason.current} onClose={onClose} />}
    </AnimatePresence>
  )
}

function SignInSheet({ reason, onClose }) {
  const [email, setEmail] = useState('')
  const [status, setStatus] = useState('idle') // idle | sending | sent | error
  const [err, setErr] = useState('')
  const inFlight = useRef(false)
  const input = useRef(null)
  const sheet = useRef(null)

  // Shared modal layer (motion/useModal.js): the app behind is inert while the sheet is mounted, Tab wraps inside the sheet, and on close
  // (after the exit animation, because AnimatePresence keeps the sheet mounted until it has faded out) inert is
  // removed first and focus goes back to the element that opened the sheet. Declared before the focus effect below
  // so the opener is captured before focus moves into the sheet.
  useModal(sheet)

  // Focus the email field on open.
  useEffect(() => {
    input.current?.focus()
  }, [])

  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const send = async (e) => {
    e.preventDefault()
    if (inFlight.current) return
    const addr = email.trim()
    if (!EMAIL_OK.test(addr)) {
      setStatus('error')
      setErr('Enter a valid email address.')
      input.current?.focus()
      return
    }
    inFlight.current = true
    setStatus('sending')
    setErr('')
    try {
      await api.requestLink(addr)
      setStatus('sent')
    } catch (x) {
      setStatus('error')
      setErr(detail(x))
    } finally {
      inFlight.current = false
    }
  }

  return (
    <motion.div
      className="signin-scrim"
      onClick={onClose}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1, transition: transitionFor({ duration: duration.base, ease: ease.out }) }}
      exit={{ opacity: 0, transition: transitionFor({ duration: duration.fast, ease: ease.in }) }}
    >
      <motion.section
        ref={sheet}
        className="signin-sheet card"
        role="dialog"
        aria-modal="true"
        aria-labelledby="signin-title"
        onClick={(e) => e.stopPropagation()}
        initial={{ opacity: 0, scale: 0.94, y: 16 }}
        animate={{ opacity: 1, scale: 1, y: 0, transition: transitionFor(spring.gentle) }}
        exit={{ opacity: 0, scale: 0.96, y: 8, transition: transitionFor({ duration: duration.fast, ease: ease.in }) }}
      >
        <div className="row spread">
          <h2 id="signin-title">Sign in</h2>
          <button type="button" className="ghost icon-btn" onClick={onClose} aria-label="Close sign in">
            <X size={18} aria-hidden="true" />
          </button>
        </div>

        {status === 'sent' ? (
          <div role="status" aria-live="polite">
            <p>Check your email for a sign-in link.</p>
            <p className="muted">Open the link in this browser to finish signing in.</p>
            <button type="button" className="primary big" onClick={onClose}>Done</button>
          </div>
        ) : (
          <form onSubmit={send} noValidate>
            {reason && <p className="muted">{reason}</p>}
            <p className="muted">We will email you a sign-in link. No password.</p>
            <label className="field-label" htmlFor="signin-email">Email</label>
            <div className="row">
              <input
                id="signin-email"
                ref={input}
                type="email"
                className="field"
                autoComplete="email"
                inputMode="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                aria-invalid={status === 'error' ? 'true' : undefined}
                aria-describedby={err ? 'signin-err' : undefined}
              />
              <button type="submit" className="primary" disabled={status === 'sending'}>
                {status === 'sending' ? 'Sending…' : 'Send link'}
              </button>
            </div>
            {err && (
              <p id="signin-err" className="notice error" role="alert">{err}</p>
            )}
          </form>
        )}
        {/* The full sign-in page (#/sign-in) is the same flow. Close the sheet first so it does not stay over the page. */}
        <p className="sheet-alt">
          <a href="#/sign-in" onClick={onClose}>Use the full sign-in page</a>
        </p>
      </motion.section>
    </motion.div>
  )
}
