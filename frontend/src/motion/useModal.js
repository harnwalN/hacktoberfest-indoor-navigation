import { useEffect } from 'react'

// Shared modal layer for every dialog (SignInDialog, Poster). While the dialog is mounted, every element that is not
// the dialog or one of its ancestors is inert: out of the tab order, out of the accessibility tree and no pointer input.
// Tab and Shift+Tab wrap inside the dialog, so focus never escapes to <body> from its last or first control.
// Focus that somehow lands outside the dialog is pulled back in.
//
// Cleanup order matters: the inert state is removed FIRST, then focus goes back to the opener. An inert element
// cannot take focus, so restoring focus before removing inert would silently do nothing.
//
// Usage (call it BEFORE any effect that moves focus into the dialog, so the opener is captured first):
//   const dialog = useRef(null)
//   useModal(dialog)                              // focus returns to whatever was focused on open
//   useModal(dialog, { returnFocusTo: btnRef })   // focus returns to btnRef.current on close
//
// The dialog must be in the DOM (inline or portaled) when the effect runs.
const FOCUSABLE = 'button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), a[href]'

export default function useModal(dialogRef, { returnFocusTo } = {}) {
  useEffect(() => {
    const dialog = dialogRef.current
    const opener = document.activeElement
    const made = []

    // Walk from the dialog up to <body>, making every sibling of each step inert.
    let node = dialog
    while (node && node.parentElement) {
      for (const sib of node.parentElement.children) {
        if (sib !== node && !sib.inert) {
          sib.inert = true
          made.push(sib)
        }
      }
      node = node.parentElement
    }

    // Focus trap: Tab and Shift+Tab cycle inside the dialog.
    const onTab = (e) => {
      const root = dialogRef.current
      if (e.key !== 'Tab' || !root) return
      const items = [...root.querySelectorAll(FOCUSABLE)]
      if (!items.length) return
      const first = items[0]
      const last = items[items.length - 1]
      const at = document.activeElement
      const inside = root.contains(at)
      if (e.shiftKey && (at === first || !inside)) {
        e.preventDefault()
        last.focus()
      } else if (!e.shiftKey && (at === last || !inside)) {
        e.preventDefault()
        first.focus()
      }
    }
    window.addEventListener('keydown', onTab)

    return () => {
      window.removeEventListener('keydown', onTab)
      for (const el of made) el.inert = false
      const target = returnFocusTo?.current || opener
      if (target && target.isConnected && typeof target.focus === 'function') target.focus({ preventScroll: true })
    }
    // The ref objects are stable; the effect runs once per mount, which is one per open dialog.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
}
