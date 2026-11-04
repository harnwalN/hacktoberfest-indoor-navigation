import { useEffect, useId, useRef, useState } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { Ban, Check, EyeOff, RotateCcw, ShieldCheck, Trash2, X } from 'lucide-react'
import { api, detail } from '../api'
import { useAuth } from './AuthContext'
import Uploader from './Uploader'
import MapReview from './MapReview'
import PosterPanel from './PosterPanel'
import ReportsSection from './ReportsSection'
import ServerPhoto from './ServerPhoto'
import ConfirmDialog from './ConfirmDialog'
import { SegButton } from './MapMotion'
import { useToast } from '../motion/Toast.jsx'
import Skeleton from '../motion/Skeleton.jsx'
import useDelayed from '../motion/useDelayed.js'
import { COLLAPSE, pick as reduced } from '../motion/dirbuild'
import '../styles/owner.css'

// The owner page (#/owner). Visible in the nav only to the owner; any other visitor who opens the URL gets NotAllowed.
// Sub-tabs keep their choice in the hash (#/owner?s=videos) so a link can open a section.
const SECTIONS = [
  { id: 'moderation', label: 'Moderation' },
  { id: 'videos', label: 'Videos & map' },
  { id: 'buildings', label: 'Buildings' },
  { id: 'reports', label: 'Reports' },
  { id: 'users', label: 'Users' },
]

const readSection = () => {
  const s = new URLSearchParams(window.location.hash.split('?')[1] || '').get('s')
  return SECTIONS.some((x) => x.id === s) ? s : null
}

// '' when the time is unknown: null, empty, unparsable, or the epoch (0), which the server sends for a missing date.
const when = (iso) => {
  if (iso === null || iso === undefined || iso === '' || iso === 0) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime()) || d.getTime() <= 0) return ''
  return d.toLocaleString(undefined, { day: 'numeric', month: 'short', hour: 'numeric', minute: '2-digit' })
}

export default function OwnerPage({ buildings, building, loaded, onChoose, onChanged }) {
  const { isOwner, checking } = useAuth()
  // With no buildings the first useful section is Buildings (create the first one).
  const [section, setSection] = useState(() => readSection() || (buildings.length ? 'moderation' : 'buildings'))
  const [focus, setFocus] = useState(null)

  // A change of the hash while the page is open (a link to #/owner?s=users from another tab's view, or a typed hash)
  // moves to that section. The section is read at mount too, above.
  useEffect(() => {
    const onHash = () => {
      const s = readSection()
      if (s) setSection(s)
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const pick = (id) => {
    setSection(id)
    const params = new URLSearchParams(window.location.hash.split('?')[1] || '')
    params.set('s', id)
    window.history.replaceState(null, '', `${window.location.pathname}${window.location.search}#/owner?${params.toString()}`)
  }

  if (checking) return <div className="card" role="status" aria-live="polite">Checking your access…</div>
  if (!isOwner) return <NotAllowed />

  const current = buildings.find((b) => b.id === building)
  // "Show on map" from Reports: the map section opens with that place selected.
  const showOnMap = (id) => {
    setFocus({ id, tick: Date.now() })
    pick('videos')
  }
  return (
    <div className="ow">
      <header className="ow-head">
        <h2><ShieldCheck size={20} aria-hidden="true" /> Owner</h2>
        <div className="view-switch ow-sections" role="group" aria-label="Owner sections">
          {SECTIONS.map((s) => (
            <SegButton key={s.id} id="owner-pill" active={section === s.id} onClick={() => pick(s.id)}>{s.label}</SegButton>
          ))}
        </div>
      </header>
      {section === 'moderation' && <ModerationSection />}
      {section === 'videos' && (
        !loaded ? null : !building ? <NoBuilding /> : (
          <>
            <Uploader building={building} onDone={onChanged} />
            <MapReview building={building} onChanged={onChanged} focus={focus} />
          </>
        )
      )}
      {section === 'buildings' && (
        <BuildingsSection buildings={buildings} building={building} onChoose={onChoose} onChanged={onChanged} current={current} />
      )}
      {section === 'reports' && (
        !building ? <NoBuilding /> : <ReportsPanel building={building} onShowNode={showOnMap} />
      )}
      {section === 'users' && <UsersSection />}
    </div>
  )
}

function NotAllowed() {
  const { signedIn, openSignIn } = useAuth()
  return (
    <div className="quiet-state" role="status">
      <h2>This page is not allowed</h2>
      <p>The owner page is only for the owner account.</p>
      {signedIn ? (
        <a className="button-link" href="#/navigate">Back to Directions</a>
      ) : (
        <button type="button" onClick={() => openSignIn('Sign in with the owner account to open this page.')}>Sign in</button>
      )}
    </div>
  )
}

function NoBuilding() {
  return <p className="card muted">No building selected. Create one under Buildings first.</p>
}

// Shared loading/error shell for the owner lists: a skeleton only after 180 ms, and a retry on error.
function useOwnerList(fetcher, deps) {
  const [items, setItems] = useState(null)
  const [error, setError] = useState('')
  const [tries, setTries] = useState(0)
  const showSkel = useDelayed(items === null && !error, 180)
  useEffect(() => {
    let live = true
    setError('')
    setItems(null) // a new filter or building shows the loading state, not the previous list
    fetcher()
      .then((list) => live && setItems(Array.isArray(list) ? list : []))
      .catch((e) => live && setError(detail(e)))
    return () => { live = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tries])
  return { items, setItems, error, showSkel, retry: () => { setItems(null); setTries((t) => t + 1) } }
}

const FILTERS = [
  { id: 'pending', label: 'Pending' },
  { id: 'rejected', label: 'Rejected' },
  { id: 'approved', label: 'Approved' },
  { id: 'hidden', label: 'Hidden' },
  { id: 'all', label: 'All' },
]
const ACTION_STATUS = { approve: 'approved', reject: 'rejected', hide: 'hidden' }
const ACTION_LABEL = { approve: 'Approve', reject: 'Reject', hide: 'Hide' }
const ACTION_DONE = { approve: 'Approved', reject: 'Rejected', hide: 'Hidden' }
const STATUS_LABEL = { pending: 'Pending', rejected: 'Rejected', approved: 'Approved', hidden: 'Hidden' }

// Moderation queue. Approve / Reject / Hide are optimistic: the card leaves (or changes its chip) at once, a toast
// reports the server's answer, and a failed call puts the card back where it was.
function ModerationSection() {
  const [filter, setFilter] = useState('pending')
  const toast = useToast()
  const { items, setItems, error, showSkel, retry } = useOwnerList(() => api.moderation(filter), [filter])
  const busy = useRef(new Set()) // keys of cards with a request in flight (their buttons are disabled)
  const [, force] = useState(0)
  const reduce = useReducedMotion()

  const key = (it) => `${it.building_id}/${it.video_id}`

  const act = (item, action) => {
    const k = key(item)
    if (busy.current.has(k)) return
    busy.current.add(k)
    const next = ACTION_STATUS[action]
    const leaves = filter !== 'all' && next !== filter
    const index = items ? items.findIndex((x) => key(x) === k) : -1
    // Optimistic: leave the list, or show the new chip in place (All).
    setItems((cur) => (cur ? (leaves ? cur.filter((x) => key(x) !== k) : cur.map((x) => (key(x) === k ? { ...x, status: next } : x))) : cur))
    force((n) => n + 1)
    api
      .moderate(item.building_id, item.video_id, action)
      .then((updated) => {
        toast.success(`${ACTION_DONE[action]}: ${item.filename || 'walk'}`)
        if (!leaves && updated && updated.status) setItems((cur) => (cur ? cur.map((x) => (key(x) === k ? { ...x, ...updated } : x)) : cur))
      })
      .catch((e) => {
        toast.error(`Could not ${ACTION_LABEL[action].toLowerCase()} ${item.filename || 'walk'}: ${detail(e)}`)
        setItems((cur) => {
          if (!cur) return cur
          if (leaves) {
            const copy = [...cur]
            copy.splice(Math.max(0, Math.min(index, copy.length)), 0, item)
            return copy
          }
          return cur.map((x) => (key(x) === k ? item : x))
        })
      })
      .finally(() => {
        busy.current.delete(k)
        force((n) => n + 1)
      })
  }

  return (
    <section className="ow-mod" aria-labelledby="mod-title">
      <div className="row spread">
        <h3 id="mod-title">Moderation queue</h3>
        <div className="view-switch" role="group" aria-label="Filter by status">
          {FILTERS.map((f) => (
            <SegButton key={f.id} id="mod-pill" active={filter === f.id} onClick={() => setFilter(f.id)}>{f.label}</SegButton>
          ))}
        </div>
      </div>
      {error && (
        <div className="banner error" role="alert">
          <span>Could not load the queue: {error}</span>
          <button type="button" className="ghost" onClick={retry}>Try again</button>
        </div>
      )}
      {!error && items === null && (
        <div role="status" aria-live="polite">
          {showSkel ? <Skeleton className="sk-card" /> : null}
          <span className="sr-only">Loading the queue…</span>
        </div>
      )}
      {items && items.length === 0 && (
        <p className="muted">{filter === 'pending' ? 'Nothing waiting. New walks from contributors show up here.' : `No ${filter === 'all' ? '' : filter + ' '}walks.`}</p>
      )}
      {items && items.length > 0 && (
        <ul className="ow-cards">
          <AnimatePresence initial={false}>
            {items.map((it) => {
              const k = key(it)
              const frames = (it.frames || []).slice(0, 6)
              const isBusy = busy.current.has(k)
              const actions = Object.keys(ACTION_STATUS).filter((a) => ACTION_STATUS[a] !== it.status)
              return (
                <motion.li key={k} className="ow-card" style={{ overflow: 'hidden' }}
                  initial={reduce ? false : { opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0, height: 'auto' }}
                  exit={{ opacity: 0, height: 0, paddingTop: 0, paddingBottom: 0, marginBottom: 0, transition: reduced(reduce, COLLAPSE) }}
                  transition={{ duration: 0.22 }}>
                  <div className="ow-frames" data-count={frames.length}>
                    {frames.length === 0 && <div className="nophoto" />}
                    {frames.map((u, i) => (
                      <ServerPhoto key={`${k}-${i}`} src={u} alt="" loading="lazy" className="ow-frame" placeholder={<div className="nophoto" />} />
                    ))}
                  </div>
                  <div className="ow-card-body">
                    <div className="row spread">
                      <b className="ow-file">{it.filename || 'Walk'}</b>
                      <span className={`ow-chip ${it.status === 'pending' ? 'review' : it.status}`}>{STATUS_LABEL[it.status] || it.status}</span>
                    </div>
                    <small className="muted">
                      {it.building_name || it.building_id} · {it.uploader_email || 'unknown uploader'} · {when(it.created_at) || 'Date unknown'}
                    </small>
                    {it.reason && <p className="ow-bot">Bot reason: {it.reason}</p>}
                    <div className="row wrap ow-actions">
                      {actions.map((a) => (
                        <button key={a} type="button" className={a === 'approve' ? 'primary' : a === 'reject' ? 'danger-soft' : 'ghost'}
                          disabled={isBusy} onClick={() => act(it, a)}>
                          {a === 'approve' && <Check size={16} aria-hidden="true" />}
                          {a === 'reject' && <X size={16} aria-hidden="true" />}
                          {a === 'hide' && <EyeOff size={16} aria-hidden="true" />}
                          {ACTION_LABEL[a]}
                        </button>
                      ))}
                    </div>
                  </div>
                </motion.li>
              )
            })}
          </AnimatePresence>
        </ul>
      )}
    </section>
  )
}

// Buildings: every building (empty ones too), create, open, delete (in-page confirm) and the entrance poster.
function BuildingsSection({ buildings, building, onChoose, onChanged, current }) {
  const toast = useToast()
  const { requireSignIn } = useAuth()
  const [name, setName] = useState('')
  const [error, setError] = useState('')
  const [creating, setCreating] = useState(false)
  const [doomed, setDoomed] = useState(null) // building waiting for the confirm dialog
  const [deleting, setDeleting] = useState(false)
  const nameId = useId()

  const create = async () => {
    const n = name.trim()
    if (!n) return
    if (!requireSignIn('Sign in to create a building. Browsing stays open.')) return
    setCreating(true)
    setError('')
    try {
      const b = await api.createBuilding(n)
      setName('')
      toast.success(`Created "${n}"`)
      onChanged()
      onChoose(b.building_id)
    } catch (e) {
      setError(detail(e))
    } finally {
      setCreating(false)
    }
  }

  const confirmDelete = async () => {
    setDeleting(true)
    try {
      await api.deleteBuilding(doomed.id)
      toast.success(`Deleted "${doomed.name}"`)
      setDoomed(null)
      onChanged()
    } catch (e) {
      setDoomed(null)
      toast.error(`Could not delete "${doomed.name}": ${detail(e)}`)
    } finally {
      setDeleting(false)
    }
  }

  return (
    <div className="ow-buildings">
      <section className="card">
        <h3>New building</h3>
        <div className="row">
          <label className="sr-only" htmlFor={nameId}>New building name</label>
          <input id={nameId} placeholder="e.g. Engineering Hall" value={name} onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && create()} />
          <button type="button" className="primary" onClick={create} disabled={creating || !name.trim()}>
            {creating ? 'Creating…' : 'Create'}
          </button>
        </div>
        {error && <p className="notice error" role="alert">{error}</p>}
      </section>

      <section className="card">
        <h3>All buildings</h3>
        {buildings.length === 0 && <p className="muted">No buildings yet. Create one above.</p>}
        <ul className="ow-list">
          {buildings.map((b) => (
            <li key={b.id} className={`ow-building ${b.id === building ? 'is-current' : ''}`}>
              <div className="ow-upload-main">
                <b>{b.name}</b>
                <small className="muted">
                  {b.nodes ? `${b.nodes} places` : 'No places yet'}{b.id === building ? ' · open in the owner tools' : ''}
                </small>
              </div>
              <div className="row wrap">
                {b.id !== building && <button type="button" className="ghost" onClick={() => onChoose(b.id)}>Open</button>}
                <button type="button" className="ghost danger" aria-label={`Delete ${b.name}`} onClick={() => setDoomed(b)}>
                  <Trash2 size={16} aria-hidden="true" /> Delete
                </button>
              </div>
            </li>
          ))}
        </ul>
      </section>

      {current && <PosterPanel key={current.id} building={current.id} name={current.name} />}

      <ConfirmDialog
        open={!!doomed}
        title={doomed ? `Delete ${doomed.name}?` : ''}
        confirmLabel="Delete building"
        danger
        busy={deleting}
        onConfirm={confirmDelete}
        onClose={() => setDoomed(null)}
      >
        {doomed && <>Its map, videos, photos and reports are deleted. This cannot be undone.</>}
      </ConfirmDialog>
    </div>
  )
}

// Reports for one building. The reports list is reused as it is; "Show on map" opens the map section on that place.
function ReportsPanel({ building, onShowNode }) {
  const [nodes, setNodes] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => {
    let live = true
    setNodes(null)
    setError('')
    api.getMap(building)
      .then((m) => live && setNodes(m?.nodes || []))
      .catch((e) => live && setError(detail(e)))
    return () => { live = false }
  }, [building])
  if (error) return <div className="banner error" role="alert">Could not load the building: {error}</div>
  if (!nodes) return <div className="card" role="status">Loading reports…</div>
  return <ReportsSection building={building} nodes={nodes} onShowNode={onShowNode} />
}

// Users: uploads and status per account, with Block / Unblock. Blocked users get 403 on upload.
function UsersSection() {
  const toast = useToast()
  const { account } = useAuth()
  const { items, setItems, error, showSkel, retry } = useOwnerList(() => api.ownerUsers(), [])
  const [busy, setBusy] = useState('')

  const toggle = async (u) => {
    if (busy) return
    setBusy(u.email)
    const blocked = !u.blocked
    try {
      const r = await api.setBlocked(u.email, blocked)
      setItems((cur) => cur && cur.map((x) => (x.email === u.email ? { ...x, blocked: r?.blocked ?? blocked } : x)))
      toast.success(blocked ? `Blocked ${u.email}` : `Unblocked ${u.email}`)
    } catch (e) {
      toast.error(`Could not ${blocked ? 'block' : 'unblock'} ${u.email}: ${detail(e)}`)
    } finally {
      setBusy('')
    }
  }

  return (
    <section className="card" aria-labelledby="users-title">
      <h3 id="users-title">Users</h3>
      {error && (
        <div className="banner error" role="alert">
          <span>Could not load users: {error}</span>
          <button type="button" className="ghost" onClick={retry}>Try again</button>
        </div>
      )}
      {!error && items === null && (
        <div role="status" aria-live="polite">
          {showSkel ? <Skeleton className="sk-card" /> : null}
          <span className="sr-only">Loading users…</span>
        </div>
      )}
      {items && items.length === 0 && <p className="muted">No contributors yet.</p>}
      {items && items.length > 0 && (
        <div className="table-scroll" tabIndex={0} role="region" aria-label="Users, scrolls sideways">
          <table className="ow-table">
            <thead>
              <tr><th>Email</th><th>Uploads</th><th>Live</th><th>Rejected</th><th>Last seen</th><th><span className="sr-only">Action</span></th></tr>
            </thead>
            <tbody>
              {items.map((u) => (
                <tr key={u.email} className={u.blocked ? 'is-blocked' : ''}>
                  <td>{u.email} {u.blocked && <span className="ow-chip rejected"><Ban size={12} aria-hidden="true" /> Blocked</span>}</td>
                  <td>{u.uploads ?? 0}</td>
                  <td>{u.live ?? 0}</td>
                  <td>{u.rejected ?? 0}</td>
                  <td>{when(u.last_seen) || '—'}</td>
                  <td>
                    {/* Owner accounts cannot be blocked (the server answers 409). Hidden for them: the row's is_owner flag,
                        or the signed-in owner's own row. */}
                    {!(u.is_owner || (account && u.email === account.email)) && (
                      <button type="button" className={u.blocked ? 'ghost' : 'danger-soft'} disabled={!!busy} onClick={() => toggle(u)}>
                        {u.blocked ? <><RotateCcw size={14} aria-hidden="true" /> Unblock</> : <><Ban size={14} aria-hidden="true" /> Block</>}
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
