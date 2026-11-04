import { useEffect, useState } from 'react'
import { motion } from 'motion/react'
import { CheckCircle2, CircleAlert, Clock, Circle as CircleIcon, UploadCloud, Video, XCircle } from 'lucide-react'
import { api, detail, MAX_FLOOR, validFloor } from '../api'
import { useAuth } from './AuthContext'
import { VideoFileCard } from './Uploader'
import Skeleton from '../motion/Skeleton.jsx'
import useDelayed from '../motion/useDelayed.js'
import { useToast } from '../motion/Toast.jsx'
import { variants, stagger } from '../motion/tokens.js'
import '../styles/owner.css'

// The contributor page (signed in, not owner): upload a walk, record one, and see what happened to your uploads.
// Processing internals (queue, Inspect, Rebuild, job controls) are owner-only and are not here.
export default function ContributePage({ building, buildingName, hasBuilding, onRecord }) {
  const { signedIn, openSignIn } = useAuth()
  const [uploadsKey, setUploadsKey] = useState(0)

  if (!signedIn) {
    return (
      <section className="card ow-prompt">
        <h2>Contribute a walk</h2>
        <p className="muted">
          Walk through a building with your phone and upload the video. Sign in with your email to upload. Browsing stays open.
        </p>
        <button type="button" className="primary big" onClick={() => openSignIn('Sign in to contribute a walk. Browsing stays open.')}>
          Sign in to contribute
        </button>
      </section>
    )
  }

  return (
    <>
      <UploadWalk building={building} buildingName={buildingName} hasBuilding={hasBuilding} onUploaded={() => setUploadsKey((k) => k + 1)} />
      <section className="card ow-record">
        <div className="row spread">
          <div>
            <h3><Video size={18} aria-hidden="true" /> Record a walk</h3>
            <p className="muted">Record on this phone. The gyro log comes with it.</p>
          </div>
          <button type="button" className="ghost" onClick={onRecord}>Open Record</button>
        </div>
      </section>
      <MyUploads key={uploadsKey} />
    </>
  )
}

// Upload a walk: the video, an optional sensor log (JSON), and the floor it starts on. Lands in the building chosen in the picker.
function UploadWalk({ building, buildingName, hasBuilding, onUploaded }) {
  const toast = useToast()
  const [file, setFile] = useState(null)
  const [sensorFile, setSensorFile] = useState(null)
  const [startFloor, setStartFloor] = useState('')
  const [pct, setPct] = useState(null) // null when idle, 0..1 while uploading
  const [error, setError] = useState('')

  const upload = async () => {
    setError('')
    let sensors = null
    try {
      sensors = sensorFile ? JSON.parse(await sensorFile.text()) : null
    } catch {
      setError('The sensor log is not valid JSON. Choose the file from the Record tab or Sensor Logger.')
      return
    }
    setPct(0)
    try {
      await api.uploadVideo(building, file, { sensors, startFloor: validFloor(startFloor) ?? '', onProgress: setPct })
      toast.success('Walk uploaded. It shows under My uploads.')
      setFile(null)
      setSensorFile(null)
      setStartFloor('')
      onUploaded()
    } catch (e) {
      setError(detail(e))
    } finally {
      setPct(null)
    }
  }

  if (!hasBuilding) {
    return (
      <section className="card">
        <h2>Upload a walk</h2>
        <p className="muted">There are no buildings to upload to yet. An owner adds them.</p>
      </section>
    )
  }

  return (
    <section className="card">
      <h2>Upload a walk</h2>
      <p className="muted">Uploading to <b>{buildingName || 'this building'}</b>. Choose the building in the picker at the top.</p>
      <label className="field">
        <span>Walk video (.mov, .mp4, .webm…)</span>
        <input type="file" accept="video/*" aria-describedby="walk-video-help" onChange={(e) => setFile(e.target.files[0] || null)} />
        <small id="walk-video-help" className="muted">Stabilization on, phone upright, pause about a second at signs and junctions.</small>
      </label>
      {file && <VideoFileCard key={`${file.name}-${file.size}-${file.lastModified}`} file={file} />}
      <label className="field">
        <span>Phone sensor log (optional, JSON)</span>
        <input type="file" accept="application/json,.json" onChange={(e) => setSensorFile(e.target.files[0] || null)} />
      </label>
      <label className="field">
        <span>Starts on floor</span>
        <input type="number" inputMode="numeric" min={1} max={MAX_FLOOR} placeholder="e.g. 1" value={startFloor} onChange={(e) => setStartFloor(e.target.value)} />
        {startFloor !== '' && validFloor(startFloor) == null && (
          <small className="error">Floor must be a whole number from 1 to {MAX_FLOOR}; it will be ignored.</small>
        )}
      </label>
      {error && <div className="banner error" role="alert">{error}</div>}
      <button className={`primary upload-btn ${pct !== null ? 'is-uploading' : ''}`} disabled={!file || pct !== null} onClick={upload}>
        {pct !== null && <span className="btn-fill" aria-hidden="true" style={{ transform: `translateX(${Math.round((pct - 1) * 100)}%)` }} />}
        <UploadCloud size={18} aria-hidden="true" />
        {pct !== null ? `Uploading ${Math.round(pct * 100)}%` : 'Upload walk'}
      </button>
    </section>
  )
}

// Friendly status for a contributor's upload. Unknown statuses show as they are, never as an empty chip.
const STATUS = {
  processing: { label: 'Processing', tone: 'processing', Icon: Clock },
  live: { label: 'Live', tone: 'live', Icon: CheckCircle2 },
  needs_review: { label: 'Needs review', tone: 'review', Icon: CircleAlert },
  rejected: { label: 'Rejected', tone: 'rejected', Icon: XCircle },
}

export function statusChip(status) {
  return STATUS[status] || { label: status || 'Unknown', tone: 'other', Icon: CircleIcon }
}

const when = (iso) => {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleString(undefined, { day: 'numeric', month: 'short', hour: 'numeric', minute: '2-digit' })
}

// The contributor's own uploads, newest first as the server sends them. Loads on mount; remounted (new key) after an upload.
function MyUploads() {
  const [items, setItems] = useState(null)
  const [error, setError] = useState('')
  const [tries, setTries] = useState(0)
  const showSkel = useDelayed(items === null && !error, 180)

  useEffect(() => {
    let live = true
    setError('')
    api
      .myUploads()
      .then((list) => live && setItems(Array.isArray(list) ? list : []))
      .catch((e) => live && setError(detail(e)))
    return () => { live = false }
  }, [tries])

  return (
    <section className="card ow-uploads" aria-labelledby="my-uploads-title">
      <h3 id="my-uploads-title">My uploads</h3>
      {error && (
        <div className="banner error" role="alert">
          <span>Could not load your uploads: {error}</span>
          <button type="button" className="ghost" onClick={() => setTries((t) => t + 1)}>Try again</button>
        </div>
      )}
      {!error && items === null && (
        <div role="status" aria-live="polite">
          {showSkel ? <Skeleton className="sk-card" /> : null}
          <span className="sr-only">Loading your uploads…</span>
        </div>
      )}
      {items && items.length === 0 && <p className="muted">No uploads yet. Walks you upload show here with their status.</p>}
      {items && items.length > 0 && (
        <motion.ul className="ow-list" initial="hidden" animate="show">
          {items.map((u, i) => {
            const chip = statusChip(u.status)
            return (
              <motion.li key={`${u.building_id}/${u.video_id}`} className="ow-upload" variants={variants.listItem} transition={{ delay: stagger(i) }}>
                <div className="ow-upload-main">
                  <b>{u.filename || 'Walk'}</b>
                  <small className="muted">{u.building_name || u.building_id}{u.created_at ? ` · ${when(u.created_at)}` : ''}</small>
                  {(u.status === 'rejected' || u.status === 'needs_review') && u.reason && (
                    <small className={`ow-reason ${u.status}`}>{u.reason}</small>
                  )}
                </div>
                <span className={`ow-chip ${chip.tone}`}>
                  <chip.Icon size={14} aria-hidden="true" />
                  {chip.label}
                </span>
              </motion.li>
            )
          })}
        </motion.ul>
      )}
    </section>
  )
}
