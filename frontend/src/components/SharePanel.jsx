import { useEffect, useId, useRef, useState } from 'react'
import { Copy, Link2, Printer } from 'lucide-react'
import { api, detail } from '../api'
import Poster from './Poster'

const ROLES = [
  { id: 'viewer', label: 'Viewer', hint: 'can get directions' },
  { id: 'editor', label: 'Editor', hint: 'can add walks and edit the map' },
]
const ROLE_NAME = { viewer: 'Viewer', editor: 'Editor', owner: 'Owner' }

// The full link someone opens: this site's address plus the join path the server returned.
export function fullLink(joinPath) {
  const jp = String(joinPath || '')
  const hash = jp.startsWith('#') ? jp : `#${jp}`
  return `${window.location.origin}${window.location.pathname}${hash}`
}

const when = (iso) => {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })
}

// Clipboard API first; a hidden textarea and execCommand when the API is unavailable (older browsers, insecure pages).
async function copyText(text) {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text)
      return true
    }
  } catch {
    /* fall through to the textarea copy */
  }
  try {
    const ta = document.createElement('textarea')
    ta.value = text
    ta.setAttribute('readonly', '')
    ta.style.position = 'fixed'
    ta.style.opacity = '0'
    document.body.appendChild(ta)
    ta.select()
    const ok = document.execCommand('copy')
    ta.remove()
    return ok
  } catch {
    return false
  }
}

// Owner only. Keyed by building in App, so switching buildings starts with a clean panel.
export default function SharePanel({ building, name }) {
  const uid = useId()
  const [role, setRole] = useState('viewer')
  const [label, setLabel] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [made, setMade] = useState(null) // { link, role } for the one link shown after creating; never stored
  const [copied, setCopied] = useState('')
  const [invites, setInvites] = useState(null) // null while loading
  const [listErr, setListErr] = useState('')
  const [confirmId, setConfirmId] = useState(null)
  const [revokingId, setRevokingId] = useState(null)
  const [notice, setNotice] = useState('')
  const [poster, setPoster] = useState(false)
  const linkRef = useRef(null)
  const posterBtn = useRef(null)

  const reload = () =>
    api
      .invites(building)
      .then((list) => {
        setInvites(Array.isArray(list) ? list : [])
        setListErr('')
      })
      .catch((e) => setListErr(detail(e)))

  useEffect(() => {
    reload()
  }, [building])

  const create = async (e) => {
    e.preventDefault()
    setBusy(true)
    setErr('')
    setCopied('')
    setNotice('')
    try {
      const r = await api.createInvite(building, { role, label: label.trim() })
      if (!r?.join_path) throw new Error('The server did not return a link. Try again.')
      setMade({ link: fullLink(r.join_path), role: r.role || role })
      setLabel('')
      await reload()
    } catch (x) {
      setErr(detail(x))
    } finally {
      setBusy(false)
    }
  }

  const copy = async () => {
    const ok = await copyText(made.link)
    setCopied(ok ? 'Copied to the clipboard.' : 'Copying did not work here. Select the link above and copy it.')
    if (!ok) {
      linkRef.current?.focus()
      linkRef.current?.select()
    }
  }

  const revoke = async (inv) => {
    setRevokingId(inv.id)
    setNotice('')
    try {
      await api.revokeInvite(building, inv.id)
      setConfirmId(null)
      setNotice(`Revoked the link${inv.label ? ` for ${inv.label}` : ''}. It no longer opens this building.`)
      await reload()
    } catch (x) {
      setListErr(detail(x))
    } finally {
      setRevokingId(null)
    }
  }

  const roleGroup = `${uid}-role`
  const labelId = `${uid}-label`

  return (
    <section className="card share-panel" aria-labelledby={`${uid}-title`}>
      <h3 id={`${uid}-title`}>
        <Link2 size={18} aria-hidden="true" /> Share this building
      </h3>
      <p className="muted">Make a link for someone to open {name || 'this building'} on their phone. They do not need an account.</p>

      <form className="share-form" onSubmit={create} noValidate>
        <fieldset className="role-choice" disabled={busy}>
          <legend>Role</legend>
          {ROLES.map((r) => (
            <label key={r.id} className="check">
              <input type="radio" name={roleGroup} value={r.id} checked={role === r.id} onChange={() => setRole(r.id)} />
              <span>
                <b>{r.label}</b> {r.hint}
              </span>
            </label>
          ))}
        </fieldset>
        <div className="field">
          <label htmlFor={labelId}>
            <span>Label (optional)</span>
          </label>
          <input
            id={labelId}
            type="text"
            value={label}
            maxLength={60}
            placeholder="e.g. Alex's phone"
            onChange={(e) => setLabel(e.target.value)}
            autoComplete="off"
            aria-describedby={role === 'viewer' ? `${labelId}-help` : undefined}
          />
          {role === 'viewer' && (
            <small id={`${labelId}-help`} className="field-help muted">
              For an entrance poster, label it &ldquo;Entrance poster&rdquo; so you can find it later.
            </small>
          )}
        </div>
        {err && <p role="alert" className="notice error">{err}</p>}
        <button type="submit" className="primary" disabled={busy}>
          {busy ? 'Creating…' : 'Create link'}
        </button>
      </form>

      {made && (
        <div className="share-made" role="region" aria-label="New invite link">
          <p>
            <b>Shown once, copy it now.</b> Anyone with this link can open {name || 'this building'} as a {ROLE_NAME[made.role] || made.role}.
            If you lose it, revoke it and create a new one.
          </p>
          <label className="field">
            <span>Invite link</span>
            <input ref={linkRef} type="text" readOnly value={made.link} onFocus={(e) => e.target.select()} />
          </label>
          <div className="row wrap">
            <button type="button" className="primary" onClick={copy}>
              <Copy size={16} aria-hidden="true" /> Copy link
            </button>
            <button type="button" className="ghost" onClick={() => { setMade(null); setCopied(''); setPoster(false) }}>
              Done
            </button>
          </div>
          <p role="status" className="muted">{copied}</p>
          {made.role === 'viewer' && (
            <div className="poster-offer">
              <p className="muted">
                <b>Anyone who scans the poster gets viewer access.</b> Revoking this invite in the list switches the poster off.
              </p>
              <button type="button" className="ghost" ref={posterBtn} onClick={() => setPoster(true)}>
                <Printer size={16} aria-hidden="true" /> Make a poster
              </button>
            </div>
          )}
        </div>
      )}
      {poster && made && (
        <Poster name={name} link={made.link} onClose={() => setPoster(false)} returnFocusTo={posterBtn} />
      )}

      {notice && <p role="status" className="notice">{notice}</p>}

      <div className="invites">
        <h4 id={`${uid}-list`}>Invite links</h4>
        {listErr && (
          <div className="notice error" role="alert">
            <span>{listErr}</span>{' '}
            <button type="button" className="ghost" onClick={reload}>Try again</button>
          </div>
        )}
        {!listErr && invites === null && <p role="status" className="muted">Loading links…</p>}
        {invites?.length === 0 && <p className="muted">No links yet. Create one above.</p>}
        {invites?.length > 0 && (
          <ul className="invite-list" aria-labelledby={`${uid}-list`}>
            {invites.map((inv) => {
              const who = inv.label || 'this link'
              return (
                <li key={inv.id} className={`invite-row ${inv.revoked ? 'is-revoked' : ''}`}>
                  <div className="invite-main">
                    <span>
                      <span className={`pill role-${inv.role}`}>{ROLE_NAME[inv.role] || inv.role}</span>{' '}
                      <b>{inv.label || 'No label'}</b>
                    </span>
                    <small className="muted">
                      Created {when(inv.created_at) || 'earlier'}
                      {inv.revoked ? ' · Revoked' : ''}
                    </small>
                  </div>
                  {!inv.revoked && confirmId !== inv.id && (
                    <button type="button" className="ghost danger" aria-label={`Revoke ${who}`} onClick={() => setConfirmId(inv.id)}>
                      Revoke
                    </button>
                  )}
                  {!inv.revoked && confirmId === inv.id && (
                    <div className="revoke-confirm" role="group" aria-label={`Confirm revoking ${who}`}>
                      <span>Revoke this link? It stops working straight away.</span>
                      <button type="button" className="danger" disabled={revokingId === inv.id} onClick={() => revoke(inv)}>
                        {revokingId === inv.id ? 'Revoking…' : 'Yes, revoke'}
                      </button>
                      <button type="button" className="ghost" autoFocus onClick={() => setConfirmId(null)}>
                        Keep it
                      </button>
                    </div>
                  )}
                </li>
              )
            })}
          </ul>
        )}
      </div>
    </section>
  )
}
