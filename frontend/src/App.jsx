import { lazy, Suspense, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { AnimatePresence, motion, PresenceContext, useReducedMotion } from 'motion/react'
import { Building2, Compass, Map as MapIcon, ShieldCheck, Upload, Video, X } from 'lucide-react'
import { api, detail } from './api'
import Navigator from './components/Navigator'
import AccountChip from './components/AccountChip'
import ErrorBoundary from './components/ErrorBoundary'
import Landing from './components/Landing'
import SignInPage from './components/SignInPage'
import { useAuth } from './components/AuthContext'
import { useOnline } from './offline'
import Skeleton from './motion/Skeleton.jsx'
import { spring, variants } from './motion/tokens.js'
import useDelayed from './motion/useDelayed.js'

// Heavy pages load on first visit (their own chunks: three.js sits in the Map chunk). Landing and Directions stay in
// the main bundle. The service worker precaches these chunks (see vite.config.js and public/sw.js).
const MapReview = lazy(() => import('./components/MapReview'))
const ContributePage = lazy(() => import('./components/ContributePage'))
const OwnerPage = lazy(() => import('./components/OwnerPage'))
const Recorder = lazy(() => import('./components/Recorder'))

// Shown while a lazy page loads. Delayed 180 ms, so a chunk that is already cached never flashes a placeholder.
function PageLoading() {
  const show = useDelayed(true, 180)
  return (
    <div className="card" role="status" aria-live="polite">
      {show ? <Skeleton className="sk-card" /> : null}
      <span className="sr-only">Loading…</span>
    </div>
  )
}

// Visitors and contributors see Directions, Map, Record and Contribute. The owner sees Owner in place of Contribute.
// 'build' is the Contribute tab's id (old links keep working); 'owner' is the owner page.
const TABS = [
  { id: 'navigate', label: 'Directions', icon: Compass },
  { id: 'review', label: 'Map', icon: MapIcon },
  { id: 'record', label: 'Record', icon: Video },
  { id: 'build', label: 'Contribute', icon: Upload },
]
const OWNER_TAB = { id: 'owner', label: 'Owner', icon: ShieldCheck }
// Last known role, so the fourth slot does not flip while the session is checked on reload.
const ROLE_KEY = 'nav_role'
const KNOWN_TABS = [...TABS.map((t) => t.id), 'owner']
// Tab slot for the sliding pill and the slide direction: Contribute and Owner share the fourth slot.
const slotOf = (id) => (id === 'owner' ? 3 : TABS.findIndex((t) => t.id === id))

// Runs when a tab's page mounts. With mode="wait" that is after the old page has left, and useLayoutEffect runs before
// the browser paints, so the scroll reset to the top is never seen. Renders nothing (no DOM node, no .content > * rule).
function ScrollTopOnMount() {
  useLayoutEffect(() => {
    window.scrollTo({ top: 0, behavior: 'instant' })
  }, [])
  return null
}

// Friendly aliases for the tab routes (#/directions = #/navigate, #/map = #/review). The old names keep working.
const ALIASES = { directions: 'navigate', map: 'review' }

// Hash routing: #/navigate?b=<building>, so a phone link opens straight into navigation.
// Route: '' or #/landing is the landing page, #/sign-in is the sign-in page, a tab id (or alias) is the app.
// Any other path keeps its old behaviour: it falls back to the Directions tab.
function readHash() {
  const raw = window.location.hash.replace(/^#\/?/, '').split('?')[0]
  const query = window.location.hash.includes('?') ? window.location.hash.split('?').slice(1).join('?') : ''
  const params = new URLSearchParams(query)
  const path = ALIASES[raw] || raw
  let route = 'app'
  if (raw === '' || raw === 'landing') route = 'landing'
  else if (raw === 'sign-in') route = 'sign-in'
  return { route, tab: KNOWN_TABS.includes(path) ? path : null, building: params.get('b') }
}

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

// Live media query (true/false), so the tab nav can move between the header and the bottom bar without a reload.
function useMediaQuery(query) {
  const [matches, setMatches] = useState(() => window.matchMedia?.(query).matches ?? true)
  useEffect(() => {
    const mq = window.matchMedia?.(query)
    if (!mq) return
    const onChange = () => setMatches(mq.matches)
    onChange()
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [query])
  return matches
}

export default function App() {
  const initial = readHash()
  const [route, setRoute] = useState(initial.route)
  const [tab, setTab] = useState(initial.tab || 'navigate')
  const online = useOnline()
  const [building, setBuilding] = useState(initial.building || loadSaved('building') || '')
  const [buildings, setBuildings] = useState([])
  const [loaded, setLoaded] = useState(false)
  const [fetchFailed, setFetchFailed] = useState(false)
  // Skeleton on the building picker only while the list really loads, and only after 180 ms (no flash on fast loads).
  const showPickerSkel = useDelayed(!loaded, 180)
  const [error, setError] = useState('')
  const { isOwner, signedIn, checking } = useAuth()
  // The fourth slot is Contribute for visitors and Owner for the owner. Both nav and the header pill use this list.
  // While the session is still being checked, the slot keeps the role saved on the last visit (no Owner -> Contribute -> Owner).
  const slotOwner = checking ? loadSaved(ROLE_KEY) === 'owner' : isOwner
  const navTabs = [TABS[0], TABS[1], TABS[2], slotOwner ? OWNER_TAB : TABS[3]]
  // Tab nav placement. 900px and up: a pill inside the header (DOM order: brand, tabs, account). Below 900px: the
  // bottom bar after the page. The landing page has no tab nav below 1160px (it links to every place itself).
  const wide = useMediaQuery('(min-width: 900px)')
  const xwide = useMediaQuery('(min-width: 1160px)')

  // The building the user is on. Empty means "first building" once the list arrives. A saved id that is not in the list
  // (removed, or not visible to this viewer) falls back to the first building.
  const knownBuilding = buildings.some((b) => b.id === building)
  const effBuilding = (loaded && building && !knownBuilding ? buildings[0]?.id : building || buildings[0]?.id) || ''
  // A saved building id can be stale (a new server, or a building removed). With no buildings on the server no building
  // view is shown, never one for a building that does not exist.
  const hasBuilding = !!effBuilding && buildings.length > 0

  // The owner's Build tab is the owner page. With no buildings yet, the owner lands on Buildings (create the first one)
  // and visitors on Contribute, which says there is nothing to upload to. Directions and Map need a building first.
  const noBuildings = loaded && !fetchFailed && !buildings.length
  const requested = tab === 'build' && isOwner ? 'owner' : tab
  const shownTab = noBuildings ? (isOwner ? 'owner' : 'build') : KNOWN_TABS.includes(requested) ? requested : 'navigate'
  // The tab that is highlighted: none on the landing page.
  const activeTab = route === 'app' ? shownTab : null
  const navOn = route === 'app' || xwide
  const navInHeader = navOn && wide
  const navBottom = navOn && !wide

  // Direction for the tab slide: 1 when the new tab is to the right of the old one, -1 when left. Recomputed only
  // when the tab changes, so a re-render during the exit animation never flips it.
  const navDir = useRef({ tab: shownTab, dir: 1 })
  if (navDir.current.tab !== shownTab) {
    navDir.current = { tab: shownTab, dir: slotOf(shownTab) >= slotOf(navDir.current.tab) ? 1 : -1 }
  }
  const dir = navDir.current.dir
  const reduceMotion = useReducedMotion()

  // Once the user has moved between tabs, the page's blocks no longer rise in (see styles.css, .content.no-rise): the
  // rise is for the first load only, so a tab switch settles within the 350 ms budget.
  const lastShown = useRef(shownTab)
  const switched = useRef(false)
  if (lastShown.current !== shownTab) {
    lastShown.current = shownTab
    switched.current = true
  }

  // Only on the app routes. The landing and sign-in pages keep their own hash (#/ and #/sign-in).
  useEffect(() => {
    if (route !== 'app') return
    const [path, query] = window.location.hash.replace(/^#\/?/, '').split('?')
    const params = new URLSearchParams(path === shownTab ? query || '' : '')
    if (effBuilding) params.set('b', effBuilding)
    const qs = params.toString()
    window.location.hash = `/${shownTab}${qs ? `?${qs}` : ''}`
    if (effBuilding) saveBuilding(effBuilding)
    // tab is a dependency so a hash for a tab that does not exist is rewritten to the one it falls back to.
  }, [route, tab, shownTab, effBuilding])

  const refresh = () =>
    api
      .buildings()
      .then((list) => {
        setBuildings(Array.isArray(list) ? list : [])
        setFetchFailed(false)
      })
      .catch((e) => {
        setFetchFailed(true)
        setError(detail(e))
      })
      .finally(() => setLoaded(true))

  // Loaded again when the viewer changes (sign in or out, owner check done): the owner sees every building, a visitor only
  // the ones with a map.
  const viewerKey = `${signedIn ? 'in' : 'out'}:${isOwner ? 'owner' : 'member'}`
  useEffect(() => {
    refresh()
  }, [viewerKey])

  // Remember the role once it is known (after the session check).
  useEffect(() => {
    if (checking) return
    try {
      localStorage.setItem(ROLE_KEY, isOwner ? 'owner' : 'member')
    } catch {
      /* private mode */
    }
  }, [checking, isOwner])

  useEffect(() => {
    const onHash = () => {
      const h = readHash()
      setRoute(h.route)
      if (h.tab) setTab(h.tab)
      if (h.building) setBuilding(h.building)
      if (h.route === 'landing' || h.route === 'sign-in') window.scrollTo({ top: 0, behavior: 'instant' })
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const choose = (id) => {
    setBuilding(id)
    saveBuilding(id)
  }

  // A tab tap only sets the tab. The scroll to the top happens in ScrollTopOnMount, when the new page mounts (after
  // the old one has faded out), so the old page never jumps while it is still visible.
  const pickTab = (id) => {
    if (id === shownTab && route === 'app') return
    setRoute('app')
    setTab(id)
  }

  // Sign-in page: its own full-screen layout, no app chrome. Hooks above still run, so this early return is safe.
  if (route === 'sign-in') return <SignInPage />

  // The one tab nav, rendered either in the header (wide) or after the page (phone). Never both.
  const tabNav = (
    <nav className="tabbar" aria-label="Main">
      {navTabs.map(({ id, label, icon: Icon }) => (
        <button key={id} className={activeTab === id ? 'active' : ''} aria-current={activeTab === id ? 'page' : undefined}
          disabled={noBuildings && id !== 'build' && id !== 'owner'} onClick={() => pickTab(id)}>
          {/* Reduced motion: a static pill on the active tab, no layoutId, so it never glides (no transform at any frame). */}
          {activeTab === id && (reduceMotion
            ? <span className="tab-pill" aria-hidden="true" />
            : <motion.span layoutId="tab-pill" className="tab-pill" aria-hidden="true" transition={spring.snappy} />)}
          {/* Plain span on purpose: whileTap here would give it tabindex="0" (an extra tab stop). Press feedback is CSS :active. */}
          <span className="tab-icon">
            <Icon size={20} />
          </span>
          <span>{label}</span>
        </button>
      ))}
    </nav>
  )

  return (
    <div className={`app${route === 'landing' ? ' navy navy-bg' : ''}`}>
      <header className={`topbar${route === 'landing' ? ' topbar-landing' : ''}`}>
        <a href="#/" className="brand" aria-label="Waypoint home">
          <span className="brand-mark"><Compass size={18} aria-hidden="true" /></span>
          <span className="brand-word">Waypoint</span>
        </a>
        {navInHeader && tabNav}
        <div className="topbar-tools">
          <AccountChip />
          <div className={`building-picker${showPickerSkel ? ' is-busy' : ''}`}>
            <Building2 size={16} />
            {showPickerSkel && <Skeleton className="picker-skel" />}
            <select aria-label="Building" value={buildings.some((b) => b.id === effBuilding) ? effBuilding : ''} onChange={(e) => choose(e.target.value)}>
              {!buildings.some((b) => b.id === effBuilding) && <option value="">{!loaded ? 'Loading…' : buildings.length ? 'Choose a building' : 'No buildings yet'}</option>}
              {buildings.map((b) => (
                <option key={b.id} value={b.id}>
                  {b.name}
                </option>
              ))}
            </select>
          </div>
        </div>
      </header>

      {!online && (
        <p className="notice" role="status">
          <span>You are offline. Saved maps and photos still work.</span>
        </p>
      )}
      {error && (
        <div className="banner error banner-dismiss" role="alert">
          <span>{error}</span>
          <button type="button" className="ghost icon-btn" onClick={() => setError('')} aria-label="Dismiss error">
            <X size={18} aria-hidden="true" />
          </button>
        </div>
      )}

      {route === 'landing' ? (
        <Landing
          signedIn={signedIn}
          loaded={loaded}
          fetchFailed={fetchFailed}
          buildings={buildings}
          building={effBuilding}
        />
      ) : (
      <AnimatePresence mode="wait" initial={false} custom={dir}>
        <motion.main key={shownTab} custom={dir} variants={reduceMotion ? variants.tabFade : variants.tabPage} initial="enter" animate="center" exit="exit"
          className={`content ${shownTab === 'navigate' ? 'is-nav' : ''} ${switched.current ? 'no-rise' : ''}`}>
          {/* AnimatePresence initial={false} (above) stops the page itself from entering on first load. Its presence context
              must not reach the page contents, or every motion child (e.g. Directions' .place-picked swap) would skip its
              initial animation too. null restores each child's own initial behaviour. */}
          <PresenceContext.Provider value={null}>
            <ScrollTopOnMount />
            {!loaded ? (
              <div className="card loading-card" role="status" aria-live="polite">Loading your buildings…</div>
            ) : fetchFailed && !buildings.length ? (
              <div className="quiet-state" role="status">
                <h2>Could not load your buildings</h2>
                <button type="button" onClick={() => { setLoaded(false); refresh() }}>Try again</button>
              </div>
            ) : (
              <>
                {/* Lazy pages: a failed chunk load (offline, first visit with no signal) shows a message here, not a blank app. */}
                <ErrorBoundary key={shownTab} title="This page could not load">
                  <Suspense fallback={<PageLoading />}>
                    {shownTab === 'owner' && (
                      <OwnerPage buildings={buildings} building={effBuilding} loaded={loaded} onChoose={choose} onChanged={refresh} />
                    )}
                    {shownTab === 'build' && (
                      <ContributePage building={effBuilding} buildingName={buildings.find((b) => b.id === effBuilding)?.name} hasBuilding={hasBuilding}
                        onRecord={() => setTab('record')} />
                    )}
                    {hasBuilding && shownTab === 'review' && <MapReview building={effBuilding} readOnly />}
                    {hasBuilding && shownTab === 'record' && <Recorder building={effBuilding} onUploaded={() => setTab(isOwner ? 'owner' : 'build')} />}
                  </Suspense>
                </ErrorBoundary>
                {hasBuilding && shownTab === 'navigate' && <Navigator building={effBuilding} />}
              </>
            )}
          </PresenceContext.Provider>
        </motion.main>
      </AnimatePresence>
      )}

      {navBottom && tabNav}
    </div>
  )
}
