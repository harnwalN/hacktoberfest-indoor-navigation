import { useCallback, useEffect, useRef, useState } from 'react'
import { Flag, MapPin, RefreshCw, RotateCcw, Check } from 'lucide-react'
import { api, detail } from '../api'
import { KIND_LABEL, RESOLUTION_NOTE_MAX, reporterName } from '../reports'

// Editors and owners only (MapReview is only offered to them). Newest first, with a filter, and each item keeps its
// history. Resolving or reopening calls onCountChanged so the open count (here and on the Map tab) is fetched again.
const FILTERS = [
  { id: 'open', label: 'Open' },
  { id: 'resolved', label: 'Resolved' },
  { id: 'all', label: 'All' },
]
const LIMIT = 100
const EMPTY = {
  open: 'No open reports. Nothing is waiting for a fix.',
  resolved: 'No resolved reports yet.',
  all: 'No reports yet. Visitors can flag a wrong step from Directions.',
}
const HISTORY_WORD = { open: 'Reported', resolved: 'Marked fixed' }

const when = (iso) => {
  const d = new Date(iso)
  return Number.isNaN(d.getTime())
    ? ''
    : d.toLocaleString(undefined, { day: 'numeric', month: 'short', year: 'numeric', hour: 'numeric', minute: '2-digit' })
}

export default function ReportsSection({ building, nodes, openCount, onCountChanged, onShowNode }) {
  const [filter, setFilter] = useState('open')
  const [items, setItems] = useState(null) // null while the list loads
  const [error, setError] = useState('')
  const [status, setStatus] = useState('')
  const [busy, setBusy] = useState('') // id of the report being changed
  const [fixing, setFixing] = useState(null) // { id, note } while the "Mark fixed" field is open
  const busyRef = useRef(false)
  const seq = useRef(0) // only the newest list request may change the screen
  const fixInput = useRef(null)
  const restoreFocus = useRef(null) // report id whose "Mark fixed" button gets focus back after the field closes

  const nodeById = new Map(nodes.map((n) => [n.id, n]))

  const load = useCallback(() => {
    const mine = ++seq.current
    setError('')
    return api
      .reports(building, { status: filter, limit: LIMIT })
      .then((list) => {
        if (mine === seq.current) setItems(Array.isArray(list) ? list : [])
      })
      .catch((e) => {
        if (mine === seq.current) setError(detail(e))
      })
  }, [building, filter])

  useEffect(() => {
    setItems(null)
    setFixing(null)
    setStatus('')
    load()
  }, [load])

  // The field takes focus when it opens; when it closes, focus goes back to that report's "Mark fixed" button.
  useEffect(() => {
    if (fixing) {
      fixInput.current?.focus()
    } else if (restoreFocus.current) {
      document.getElementById(`report-fix-${restoreFocus.current}`)?.focus()
      restoreFocus.current = null
    }
  }, [fixing?.id])

  // One change to one report. Only one can be in flight at a time.
  const change = (r, body, done) => {
    if (busyRef.current) return
    busyRef.current = true
    setBusy(r.id)
    setError('')
    setStatus('')
    return api
      .updateReport(building, r.id, body)
      .then(() => {
        setFixing(null)
        setStatus(done)
        return load().then(() => onCountChanged?.())
      })
      .catch((e) => setError(detail(e)))
      .finally(() => {
        busyRef.current = false
        setBusy('')
      })
  }

  const confirmFixed = (r) => {
    const text = (fixing?.note || '').trim()
    // The resolution note is sent only when there is one; the server keeps it on the report.
    change(r, text ? { status: 'resolved', resolution_note: text } : { status: 'resolved' }, 'Marked as fixed.')
  }

  const cancelFixed = (r) => {
    restoreFocus.current = r.id
    setFixing(null)
  }

  const placeText = (r) => {
    if (!r.node_id) return 'No place attached'
    const node = nodeById.get(r.node_id)
    const label = r.node_label || node?.label
    if (label) return label
    return node ? 'Unnamed place' : 'Place no longer on the map'
  }

  return (
    <section className="card reports" aria-labelledby="reports-title">
      <div className="row spread reports-head">
        <h3 id="reports-title">
          <Flag size={18} aria-hidden="true" /> Reports
          {typeof openCount === 'number' && (
            <span className={`count-badge ${openCount > 0 ? 'has-open' : ''}`}>{openCount} open</span>
          )}
        </h3>
        <div className="view-switch" role="group" aria-label="Show reports">
          {FILTERS.map((f) => (
            <button key={f.id} type="button" className={filter === f.id ? 'active' : ''} aria-pressed={filter === f.id}
              onClick={() => setFilter(f.id)}>
              {f.label}
            </button>
          ))}
        </div>
      </div>
      <p className="muted reports-help">Wrong steps and places that visitors flagged from Directions.</p>

      {error && (
        <div className="notice error" role="alert">
          <span>Could not update the reports: {error}</span>{' '}
          {items === null ? (
            <button type="button" className="ghost" onClick={load}><RefreshCw size={16} aria-hidden="true" /> Try again</button>
          ) : null}
        </div>
      )}
      <p role="status" aria-live="polite" className="muted">{status}</p>

      {items === null && !error && <p role="status" className="muted">Loading reports…</p>}
      {items?.length === 0 && <p className="muted reports-empty">{EMPTY[filter]}</p>}

      {items?.length > 0 && (
        <ul className="report-list" aria-label={`${FILTERS.find((f) => f.id === filter)?.label} reports`}>
          {items.map((r) => {
            const who = reporterName(r.reporter)
            const exists = !!r.node_id && nodeById.has(r.node_id)
            const isFixing = fixing?.id === r.id
            const isBusy = busy === r.id
            return (
              <li key={r.id} className={`report-item is-${r.status}`}>
                <div className="report-main">
                  <p className="report-kind">
                    <b>{KIND_LABEL[r.kind] || r.kind}</b>
                    {r.status === 'resolved' && <span className="pill done">Fixed</span>}
                  </p>
                  <p className="report-place">
                    <MapPin size={14} aria-hidden="true" /> {placeText(r)}
                    {r.step_index != null && <span className="muted"> · step {r.step_index + 1}</span>}
                  </p>
                  {r.note && <blockquote className="report-note-text">{r.note}</blockquote>}
                  <p className="muted report-meta">
                    <time dateTime={r.created_at}>{when(r.created_at)}</time>
                    {who && <span className="report-who"> · {who}</span>}
                  </p>
                  {r.status === 'resolved' && r.resolution_note && (
                    <p className="report-resolution">Fix: {r.resolution_note}</p>
                  )}
                </div>

                <div className="report-actions row wrap">
                  {exists && (
                    <button type="button" className="ghost" onClick={() => onShowNode?.(r.node_id)} disabled={isBusy}>
                      Show on map
                    </button>
                  )}
                  {r.status === 'open' && !isFixing && (
                    <button type="button" id={`report-fix-${r.id}`} className="primary" disabled={isBusy}
                      onClick={() => setFixing({ id: r.id, note: '' })}>
                      <Check size={16} aria-hidden="true" /> Mark fixed
                    </button>
                  )}
                  {r.status === 'resolved' && (
                    <button type="button" disabled={isBusy} onClick={() => change(r, { status: 'open' }, 'Reopened.')}>
                      <RotateCcw size={16} aria-hidden="true" /> {isBusy ? 'Reopening…' : 'Reopen'}
                    </button>
                  )}
                </div>

                {isFixing && (
                  <div className="report-fix" role="group" aria-label="Mark as fixed">
                    <label className="field">
                      <span>Resolution note (optional)</span>
                      <textarea ref={fixInput} rows={2} maxLength={RESOLUTION_NOTE_MAX} value={fixing.note}
                        aria-describedby={`report-fix-count-${r.id}`} onChange={(e) => setFixing({ id: r.id, note: e.target.value })} />
                      <small id={`report-fix-count-${r.id}`} className="muted">{fixing.note.length} of {RESOLUTION_NOTE_MAX} characters</small>
                    </label>
                    <div className="row wrap">
                      <button type="button" className="primary" disabled={isBusy} onClick={() => confirmFixed(r)}>
                        {isBusy ? 'Saving…' : 'Confirm fixed'}
                      </button>
                      <button type="button" className="ghost" disabled={isBusy} onClick={() => cancelFixed(r)}>Cancel</button>
                    </div>
                  </div>
                )}

                {r.history?.length > 0 && (
                  <details className="report-history">
                    <summary>History</summary>
                    <ol>
                      {r.history.map((h, k) => (
                        <li key={k}>
                          {HISTORY_WORD[h.status] || h.status} · <time dateTime={h.at}>{when(h.at)}</time>
                          {reporterName(h.by) && <> · {reporterName(h.by)}</>}
                          {h.resolution_note && <> · {h.resolution_note}</>}
                        </li>
                      ))}
                    </ol>
                  </details>
                )}
              </li>
            )
          })}
        </ul>
      )}
      {items?.length >= LIMIT && <p className="muted reports-empty">Showing the newest {LIMIT}.</p>}
    </section>
  )
}
