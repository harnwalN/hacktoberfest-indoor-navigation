import { useEffect, useId, useRef, useState } from 'react'
import { api } from '../api'
import { REPORT_KINDS, REPORT_NOTE_MAX, THANKS, reportError } from '../reports'

// A quiet "Something's wrong?" control under one direction step. It opens an inline panel (not a modal). The step's
// place and the trip's ends go with the report, so the editors see exactly where the walker was. The parent gives this
// component a key per step and trip, so a panel never carries over to a different step.
export default function ReportIssue({ building, nodeId, stepIndex, fromNode, toNode }) {
  const uid = useId()
  const [open, setOpen] = useState(false)
  const [kind, setKind] = useState('')
  const [note, setNote] = useState('')
  const [sending, setSending] = useState(false)
  const [err, setErr] = useState('')
  const [sent, setSent] = useState('')
  // A ref, not state, guards the send: a second click before React re-renders still cannot start a second request.
  const inFlight = useRef(false)
  const toggle = useRef(null)
  const panel = useRef(null)

  // Focus moves into the panel when it opens: the chosen kind, or the first kind.
  useEffect(() => {
    if (!open) return
    panel.current?.querySelector('input[type=radio]:checked, input[type=radio]')?.focus()
  }, [open])

  const openPanel = () => {
    setSent('')
    setOpen(true)
  }

  // Closing always returns focus to the button that opened the panel.
  const close = () => {
    setOpen(false)
    setErr('')
    toggle.current?.focus()
  }

  const send = async (e) => {
    e.preventDefault()
    if (inFlight.current) return
    if (!kind) {
      setErr('Choose what is wrong first.')
      panel.current?.querySelector('input[type=radio]')?.focus()
      return
    }
    const body = { kind }
    const text = note.trim()
    if (text) body.note = text
    if (nodeId) body.node_id = nodeId
    if (fromNode) body.from_node = fromNode
    if (toNode) body.to_node = toNode
    // 0 is a real step (the first one), so only a missing or out-of-range index is left out.
    if (Number.isInteger(stepIndex) && stepIndex >= 0 && stepIndex <= 500) body.step_index = stepIndex
    inFlight.current = true
    setSending(true)
    setErr('')
    try {
      await api.report(building, body)
      setKind('')
      setNote('')
      setSent(THANKS)
      setOpen(false)
      toggle.current?.focus()
    } catch (x) {
      setErr(reportError(x))
    } finally {
      inFlight.current = false
      setSending(false)
    }
  }

  // Keys stay inside the panel: Escape closes it, and arrow keys do not step the route underneath.
  const onKey = (e) => {
    e.stopPropagation()
    if (e.key === 'Escape') {
      e.preventDefault()
      close()
    }
  }

  return (
    <div className="report-issue">
      <button
        type="button"
        ref={toggle}
        className="report-toggle"
        aria-expanded={open}
        aria-controls={`${uid}-panel`}
        onClick={() => (open ? close() : openPanel())}
      >
        Something's wrong?
      </button>
      {open && (
        <form id={`${uid}-panel`} ref={panel} className="report-panel" onSubmit={send} onKeyDown={onKey} noValidate
          aria-label="Report a problem with this step">
          <fieldset className="report-kinds">
            <legend>What's wrong?</legend>
            {REPORT_KINDS.map((k) => (
              <label key={k.id} className="check report-choice">
                <input type="radio" name={`${uid}-kind`} value={k.id} checked={kind === k.id} onChange={() => setKind(k.id)} />
                <span>{k.label}</span>
              </label>
            ))}
          </fieldset>
          <label className="field report-note" htmlFor={`${uid}-note`}>
            <span>Add a note (optional)</span>
            <textarea id={`${uid}-note`} rows={3} value={note} maxLength={REPORT_NOTE_MAX} aria-describedby={`${uid}-count`}
              onChange={(e) => setNote(e.target.value)} />
            <small id={`${uid}-count`} className="muted">{note.length} of {REPORT_NOTE_MAX} characters</small>
          </label>
          {err && <p role="alert" className="notice error">{err}</p>}
          <div className="row wrap report-actions">
            <button type="submit" className="primary" aria-disabled={sending || undefined}>
              {sending ? 'Sending…' : 'Send'}
            </button>
            <button type="button" className="ghost" onClick={close}>Cancel</button>
          </div>
        </form>
      )}
      <p role="status" className="report-thanks">{sent && !open ? sent : ''}</p>
    </div>
  )
}
