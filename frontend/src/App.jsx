import { useCallback, useEffect, useRef, useState } from 'react'
import { Building2, Compass, Map as MapIcon, Upload, Video, X } from 'lucide-react'
import { api, detail } from './api'
import {
  clearAllAccess,
  clearToken,
  consumeJoinLink,
  getToken,
  hasOwnerKey,
  hasStoredAccess,
  onAccessNeeded,
  onOwnerKeyRemoved,
  setCurrentBuilding,
} from './access'
import Uploader from './components/Uploader'
import MapReview from './components/MapReview'
import Navigator from './components/Navigator'
import Recorder from './components/Recorder'
import AccessScreen from './components/AccessScreen'
import SharePanel from './components/SharePanel'
import SignOutMenu from './components/SignOutMenu'

const TABS = [
  { id: 'navigate', label: 'Directions', icon: Compass },
  { id: 'review', label: 'Map', icon: MapIcon, edit: true },
  { id: 'record', label: 'Record', icon: Video, edit: true },
  { id: 'build', label: 'Build', icon: Upload, edit: true },
]

// Tabs a role can use. Viewers get Directions only. With no known role (not accessible, or not loaded yet) it is
// Directions only too, so nothing beyond it is ever offered without a role.
const tabsFor = (role) => TABS.filter((t) => !t.edit || role === 'owner' || role === 'editor')

const KEY_REMOVED_NOTICE = 'Your saved owner key no longer works, so it was removed from this device.'

// Hash routing: #/navigate?b=<building>, so a phone link opens straight into navigation.
function readHash() {
  const [path, query] = window.location.hash.replace(/^#\/?/, '').split('?')
  const params = new URLSearchParams(query || '')
  return { tab: TABS.some((t) => t.id === path) ? path : null, building: params.get('b') }
}

// A join link (#/join?b=&t=) is consumed once per page load, before the first render, so its token is out of the
// address bar before anything reads the hash.
const BOOT_JOIN = consumeJoinLink()

function loadSaved(key) {
  try {
    return localStorage.getItem(key)
  } catch {
    return null
  }
}

function saveBuilding(id) {
  try {
    localStorage.setItem('building', id)
  } catch {
    /* private mode */
  }
}

export default function App() {
  const initial = readHash()
  const [tab, setTab] = useState(initial.tab || 'navigate')
  const [building, setBuilding] = useState(initial.building || loadSaved('building') || '')
  const [buildings, setBuildings] = useState([])
  const [loaded, setLoaded] = useState(false)
  const [fetchFailed, setFetchFailed] = useState(false)
  // An explicit access problem: a 401 from a building call, or a join link that was missing parts.
  const [access, setAccess] = useState(() => (BOOT_JOIN && !BOOT_JOIN.ok ? { building: BOOT_JOIN.building, reason: 'bad-link' } : null))
  const [newName, setNewName] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  // The building the user is on. Empty means "first accessible building" once the list arrives.
  const effBuilding = building || buildings[0]?.id || ''
  const current = buildings.find((b) => b.id === effBuilding)
  const role = current?.role || null
  // A saved building id can be stale (a new server, or a building removed). With no buildings on the server only the
  // "New building" card is shown, never a view for a building that does not exist.
  const hasBuilding = !!effBuilding && buildings.length > 0
  const ownerKey = hasOwnerKey()
  const stored = hasStoredAccess()
  setCurrentBuilding(effBuilding)

  // A valid owner key on a server with no buildings yet is still the owner: the shell offers all four tabs. The tabs
  // whose content needs a building are disabled until one exists, and Build (New building) is the tab that is open.
  const noBuildings = loaded && !fetchFailed && !buildings.length
  const tabRole = role || (noBuildings && ownerKey ? 'owner' : null)
  const allowed = (loaded ? tabsFor(tabRole) : TABS).map((t) => t.id)
  let shownTab = allowed.includes(tab) ? tab : 'navigate'
  if (noBuildings && allowed.includes('build')) shownTab = 'build'

  // Which screen needs an invite. An explicit 401 wins. Otherwise: an empty list without an owner key, or a building
  // the list does not include, both need one.
  let needed = access
  if (!needed && loaded && !fetchFailed) {
    if (!buildings.length && !ownerKey) needed = { building: effBuilding, reason: stored ? 'invalid' : 'none' }
    else if (buildings.length && effBuilding && !current)
      needed = { building: effBuilding, reason: ownerKey ? 'missing' : getToken(effBuilding) ? 'invalid' : 'none' }
  }

  const latest = useRef('')
  useEffect(() => {
    latest.current = effBuilding
  }, [effBuilding])

  const refresh = useCallback(
    () =>
      api
        .buildings()
        .then((list) => {
          setBuildings(Array.isArray(list) ? list : [])
          setFetchFailed(false)
        })
        .catch((e) => {
          // A 401 is handled by the access screen (see onAccessNeeded below); anything else is an error banner.
          if (e?.response?.status !== 401) {
            setFetchFailed(true)
            setError(detail(e))
          }
        })
        .finally(() => setLoaded(true)),
    [],
  )
  useEffect(() => {
    refresh()
  }, [refresh])

  // The server refused a stored owner key on the building list. The key is already gone; say so once.
  useEffect(() => onOwnerKeyRemoved(() => setNotice(KEY_REMOVED_NOTICE)), [])

  useEffect(
    () =>
      onAccessNeeded(({ building: b, hadToken, hadOwnerKey }) => {
        // A late reply for a building the user has already left must not pull them back to its access screen.
        if (b && b !== latest.current) return
        setAccess({ building: b || latest.current, reason: hadOwnerKey ? 'owner' : hadToken ? 'invalid' : 'none' })
      }),
    [],
  )

  useEffect(() => {
    const [path, query] = window.location.hash.replace(/^#\/?/, '').split('?')
    const params = new URLSearchParams(path === shownTab ? query || '' : '')
    if (effBuilding) params.set('b', effBuilding)
    const qs = params.toString()
    window.location.hash = `/${shownTab}${qs ? `?${qs}` : ''}`
    if (effBuilding) saveBuilding(effBuilding)
    // tab is a dependency so a hash for a tab this role cannot use is rewritten to the one it falls back to.
  }, [tab, shownTab, effBuilding])

  useEffect(() => {
    const onHash = () => {
      const join = consumeJoinLink()
      if (join) {
        if (join.ok) {
          // A join link for a building the user already has replaces the old token (setToken overwrites).
          setAccess(null)
          setNotice('')
          setLoaded(false)
          setTab('navigate')
          setBuilding(join.building)
          refresh()
        } else {
          setAccess({ building: join.building, reason: 'bad-link' })
        }
        return
      }
      const h = readHash()
      if (h.tab) setTab(h.tab)
      if (h.building) setBuilding(h.building)
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [refresh])

  const choose = (id) => {
    setNotice('')
    setAccess(null)
    setBuilding(id)
  }

  // An invite or owner key was just saved (or a building was picked from the access screen): check again from the server.
  const granted = (id) => {
    setAccess(null)
    setNotice('')
    setError('')
    if (id) {
      setBuilding(id)
      setTab('navigate')
    }
    setLoaded(false)
    refresh()
  }

  const signOut = (scope) => {
    if (scope === 'all') clearAllAccess()
    else clearToken(effBuilding)
    setAccess(null)
    setNotice(scope === 'all' ? 'Signed out on this device. Invite links and the owner key were removed.' : 'Removed this building\'s invite from this device.')
    setLoaded(false)
    refresh()
  }

  const create = async () => {
    if (!newName.trim()) return
    try {
      const b = await api.createBuilding(newName.trim())
      setNewName('')
      setAccess(null)
      setLoaded(false)
      setBuilding(b.building_id)
      setTab('build')
      await refresh()
    } catch (e) {
      setError(detail(e))
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark"><Compass size={20} /></span>
          <span>Waypoint</span>
        </div>
        <div className="topbar-tools">
          <div className="building-picker">
            <Building2 size={16} />
            <select aria-label="Building" value={current ? effBuilding : ''} onChange={(e) => choose(e.target.value)}>
              {!current && <option value="">{!loaded ? 'Loading…' : buildings.length ? 'Choose a building' : 'No buildings yet'}</option>}
              {buildings.map((b) => (
                <option key={b.id} value={b.id}>
                  {b.name}
                </option>
              ))}
            </select>
          </div>
          {stored && <SignOutMenu building={effBuilding} onSignOut={signOut} />}
        </div>
      </header>

      {error && (
        <div className="banner error banner-dismiss" role="alert">
          <span>{error}</span>
          <button type="button" className="ghost icon-btn" onClick={() => setError('')} aria-label="Dismiss error">
            <X size={18} aria-hidden="true" />
          </button>
        </div>
      )}
      {notice && (
        <p className="notice notice-dismiss" role="status">
          <span>{notice}</span>
          <button type="button" className="ghost icon-btn" onClick={() => setNotice('')} aria-label="Dismiss notice">
            <X size={18} aria-hidden="true" />
          </button>
        </p>
      )}

      <main className={`content ${shownTab === 'navigate' ? 'is-nav' : ''}`}>
        {needed ? (
          <AccessScreen
            key={`${needed.building}:${needed.reason}`}
            building={needed.building}
            reason={needed.reason}
            buildings={buildings}
            hasStored={stored}
            onGranted={granted}
            onOpen={choose}
            onSignOut={signOut}
          />
        ) : !loaded ? (
          <div className="card loading-card" role="status" aria-live="polite">Loading your buildings…</div>
        ) : fetchFailed && !buildings.length ? (
          <div className="quiet-state" role="status">
            <h2>Could not load your buildings</h2>
            <button type="button" onClick={() => { setLoaded(false); refresh() }}>Try again</button>
          </div>
        ) : (
          <>
            {ownerKey && (shownTab === 'build' || !buildings.length) && (
              <section className="card new-building">
                <h3>New building</h3>
                {!buildings.length && <p className="muted">Create your first building to unlock Directions, Map and Record.</p>}
                <div className="row">
                  <input aria-label="New building name" placeholder="e.g. Engineering Hall" value={newName} onChange={(e) => setNewName(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && create()} />
                  <button onClick={create}>Create</button>
                </div>
              </section>
            )}
            {hasBuilding && shownTab === 'build' && <Uploader building={effBuilding} onDone={refresh} />}
            {hasBuilding && shownTab === 'build' && role === 'owner' && (
              <SharePanel key={effBuilding} building={effBuilding} name={current?.name} />
            )}
            {hasBuilding && shownTab === 'review' && <MapReview building={effBuilding} onChanged={refresh} />}
            {hasBuilding && shownTab === 'navigate' && <Navigator building={effBuilding} />}
            {hasBuilding && shownTab === 'record' && <Recorder building={effBuilding} onUploaded={() => setTab('build')} />}
          </>
        )}
      </main>

      {!needed && (
        <nav className="tabbar" aria-label="Main">
          {TABS.filter((t) => allowed.includes(t.id)).map(({ id, label, icon: Icon }) => (
            <button key={id} className={shownTab === id ? 'active' : ''} aria-current={shownTab === id ? 'page' : undefined}
              disabled={noBuildings && id !== 'build'} onClick={() => setTab(id)}>
              <Icon size={20} />
              <span>{label}</span>
            </button>
          ))}
        </nav>
      )}
    </div>
  )
}
