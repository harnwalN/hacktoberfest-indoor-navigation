import { useEffect, useRef } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { duration, ease, spring, transitionFor } from '../motion/tokens.js'
import useModal from '../motion/useModal.js'

// In-page confirm sheet (never window.confirm). Same scrim and sheet motion as SignInDialog. Escape and the scrim
// cancel; the confirm button runs onConfirm and stays busy until it resolves.
export default function ConfirmDialog({ open, title, children, confirmLabel = 'Confirm', danger = false, busy = false, onConfirm, onClose }) {
  return (
    <AnimatePresence>
      {open && (
        <ConfirmSheet key="confirm" title={title} confirmLabel={confirmLabel} danger={danger} busy={busy} onConfirm={onConfirm} onClose={onClose}>
          {children}
        </ConfirmSheet>
      )}
    </AnimatePresence>
  )
}

function ConfirmSheet({ title, children, confirmLabel, danger, busy, onConfirm, onClose }) {
  const sheet = useRef(null)
  const cancel = useRef(null)
  // Captures the opener first (see motion/README.md), then the cancel button gets the initial focus.
  useModal(sheet)
  useEffect(() => {
    cancel.current?.focus()
  }, [])
  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && !busy && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [busy, onClose])

  return (
    <motion.div
      className="signin-scrim"
      onClick={() => !busy && onClose()}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1, transition: transitionFor({ duration: duration.base, ease: ease.out }) }}
      exit={{ opacity: 0, transition: transitionFor({ duration: duration.fast, ease: ease.in }) }}
    >
      <motion.section
        ref={sheet}
        className="signin-sheet card"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
        onClick={(e) => e.stopPropagation()}
        initial={{ opacity: 0, scale: 0.94, y: 16 }}
        animate={{ opacity: 1, scale: 1, y: 0, transition: transitionFor(spring.gentle) }}
        exit={{ opacity: 0, scale: 0.96, y: 8, transition: transitionFor({ duration: duration.fast, ease: ease.in }) }}
      >
        <h2 id="confirm-title">{title}</h2>
        <div className="muted">{children}</div>
        <div className="row wrap">
          <button type="button" ref={cancel} className="ghost" onClick={onClose} disabled={busy}>Cancel</button>
          <button type="button" className={danger ? 'danger-solid' : 'primary'} onClick={onConfirm} disabled={busy}>
            {busy ? 'Working…' : confirmLabel}
          </button>
        </div>
      </motion.section>
    </motion.div>
  )
}
