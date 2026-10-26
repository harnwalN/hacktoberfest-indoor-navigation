import { useEffect, useState } from 'react'
import { Building2, Compass, Map as MapIcon, Upload, Video } from 'lucide-react'
import { api, detail } from './api'
import Uploader from './components/Uploader'
import MapReview from './components/MapReview'
import Navigator from './components/Navigator'
import Recorder from './components/Recorder'

const TABS = [
  { id: 'navigate', label: 'Directions', icon: Compass },
  { id: 'review', label: 'Map', icon: MapIcon },
  { id: 'record', label: 'Record', icon: Video },
  { id: 'build', label: 'Build', icon: Upload },
]

// Hash routing: #/navigate?b=<building>, so a phone link opens straight into navigation.
function readHash() {
  const [path, query] = window.location.hash.replace(/^#\/?/, '').split('?')
  const params = new URLSearchParams(query || '')
  return { tab: TABS.some((t) => t.id === path) ? path : null, building: params.get('b') }
}

function loadSaved(key) {
  try {
    return localStorage.getItem(key)
  } catch {
    return null
  }
}

export default function App() {
  const initial = readHash()
  const [tab, setTab] = useState(initial.tab || 'navigate')
  const [building, setBuilding] = useState(initial.building || loadSaved('building') || '')
  const [buildings, setBuildings] = useState([])
  const [newName, setNewName] = useState('')
  const [error, setError] = useState('')

  const refresh = () => api.buildings().then(setBuildings).catch((e) => setError(detail(e)))
  useEffect(() => {
    refresh()
  }, [])
  useEffect(() => {
    if (!building && buildings.length) setBuilding(buildings[0].id)
  }, [buildings, building])
  useEffect(() => {
    window.location.hash = `/${tab}${building ? `?b=${building}` : ''}`
    try {
      if (building) localStorage.setItem('building', building)
    } catch {
      /* private mode */
    }
  }, [tab, building])
  useEffect(() => {
    const onHash = () => {
      const h = readHash()
      if (h.tab) setTab(h.tab)
      if (h.building) setBuilding(h.building)
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const create = async () => {
    if (!newName.trim()) return
    try {
      const b = await api.createBuilding(newName.trim())
      setNewName('')
      await refresh()
      setBuilding(b.building_id)
      setTab('build')
    } catch (e) {
      setError(detail(e))
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <Compass size={22} />
          <span>Waypoint</span>
        </div>
        <div className="building-picker">
          <Building2 size={16} />
          <select value={building} onChange={(e) => setBuilding(e.target.value)}>
            {!buildings.length && <option value="">No buildings yet</option>}
            {buildings.map((b) => (
              <option key={b.id} value={b.id}>
                {b.name}
              </option>
            ))}
          </select>
        </div>
      </header>

      {error && (
        <div className="banner error" onClick={() => setError('')}>
          {error}
        </div>
      )}

      <main className={`content ${tab === 'navigate' ? 'is-nav' : ''}`}>
        {(tab === 'build' || !buildings.length) && (
          <section className="card new-building">
            <h3>New building</h3>
            <div className="row">
              <input placeholder="e.g. Engineering Hall" value={newName} onChange={(e) => setNewName(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && create()} />
              <button onClick={create}>Create</button>
            </div>
          </section>
        )}
        {building && tab === 'build' && <Uploader building={building} onDone={refresh} />}
        {building && tab === 'review' && <MapReview building={building} onChanged={refresh} />}
        {building && tab === 'navigate' && <Navigator building={building} />}
        {building && tab === 'record' && <Recorder building={building} onUploaded={() => setTab('build')} />}
      </main>

      <nav className="tabbar">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button key={id} className={tab === id ? 'active' : ''} onClick={() => setTab(id)}>
            <Icon size={20} />
            <span>{label}</span>
          </button>
        ))}
      </nav>
    </div>
  )
}
