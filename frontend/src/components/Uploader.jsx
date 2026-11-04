import { useEffect, useId, useRef, useState } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import FlowNumber from './FlowNumber'
import { ChevronDown, Cpu, RefreshCw, Search, Trash2, UploadCloud } from 'lucide-react'
import { api, detail, MAX_FLOOR, validFloor } from '../api'
import { useAuth } from './AuthContext'
import JobProgress from './JobProgress'
import QueuePanel from './QueuePanel'
import VideoDebug from './VideoDebug'
import { CARD_IN, COLLAPSE, FADE, pick as reduced, rowIn } from '../motion/dirbuild'

export const formatSize = (b) => (b >= 1048576 ? `${(b / 1048576).toFixed(b >= 10 * 1048576 ? 0 : 1)} MB` : `${Math.max(1, Math.round(b / 1024))} KB`)

// The chosen walk, shown under its picker: name, size and the first frame. The object URL lives only as long as the
// file is chosen (revoked on change and on unmount). The card is keyed by the file, so each choice slides in again.
export function VideoFileCard({ file }) {
  const [url, setUrl] = useState('')
  const reduce = useReducedMotion()
  useEffect(() => {
    const u = URL.createObjectURL(file)
    setUrl(u)
    return () => URL.revokeObjectURL(u)
  }, [file])
  return (
    <motion.div className="file-card" initial={reduce ? false : { opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={reduced(reduce, CARD_IN)}>
      {url && <video className="file-thumb" src={`${url}#t=0.1`} muted playsInline preload="metadata" aria-hidden="true" tabIndex={-1} />}
      <div className="file-meta">
        <b>{file.name}</b>
        <small>{formatSize(file.size)}</small>
      </div>
    </motion.div>
  )
}

// Status of a walk. A change cross-fades the old pill out and the new one in. Turning "done" pops the pill once.
// The first render is not a change, so existing "done" walks do not pop on page load.
function StatusPill({ status }) {
  const reduce = useReducedMotion()
  const seen = useRef(status)
  const [pop, setPop] = useState(false)
  useEffect(() => {
    if (seen.current !== status && status === 'done') setPop(true)
    seen.current = status
  }, [status])
  return (
    <span className="pill-stack">
      <AnimatePresence initial={false}>
        <motion.span key={status} className={`pill ${status}${pop ? ' just-done' : ''}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          exit={{ opacity: 0 }} transition={reduced(reduce, FADE)} onAnimationEnd={() => setPop(false)}>
          {status}
        </motion.span>
      </AnimatePresence>
    </span>
  )
}

// Recording tips: a disclosure button (aria-expanded) with a turning chevron and a height collapse.
function RecordingTips() {
  const [open, setOpen] = useState(false)
  const id = useId()
  const reduce = useReducedMotion()
  return (
    <div className="tips">
      <button type="button" className="tips-toggle" aria-expanded={open} aria-controls={open ? id : undefined} onClick={() => setOpen((o) => !o)}>
        <ChevronDown size={20} aria-hidden="true" className="tips-chevron" />
        Recording tips
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div key="tips" id={id} className="tips-body" initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }} transition={reduced(reduce, COLLAPSE)}>
            <ul>
              <li>1080p, stabilization on, phone upright at chest height, camera pointing where you walk.</li>
              <li>Walk at a steady pace and turn slowly at junctions.</li>
              <li>
                <b>Pause ~1 s</b> at every door plaque, junction and staircase — pauses become sharp keyframes.
              </li>
              <li>Glancing at a plaque and back is fine: it is detected as a glance, not a turn.</li>
              <li>Walk corridors both ways when you can, and come back past known places to close loops.</li>
            </ul>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default function Uploader({ building, onDone }) {
  const [health, setHealth] = useState(null)
  const [map, setMap] = useState(null)
  const [file, setFile] = useState(null)
  const [sensorFile, setSensorFile] = useState(null)
  const [model, setModel] = useState('')
  const [priority, setPriority] = useState(false)
  const [startFloor, setStartFloor] = useState('')
  const [uploadPct, setUploadPct] = useState(null)
  const [jobId, setJobId] = useState(null)
  // True for the job of a building's first walk (no video done yet): only that one gets the confetti.
  const [firstWalk, setFirstWalk] = useState(false)
  const [inspect, setInspect] = useState(null)
  const [error, setError] = useState('')
  const { requireSignIn } = useAuth()
  const reduce = useReducedMotion()

  const reload = () => api.getMap(building).then(setMap).catch((e) => setError(detail(e)))
  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null))
    reload()
  }, [building])

  const gemmaModels = (health?.gemma?.models || []).filter((m) => m.startsWith('gemma'))

  const upload = async () => {
    if (!requireSignIn('Sign in to upload a walk. Browsing stays open.')) return
    setError('')
    try {
      const sensors = sensorFile ? JSON.parse(await sensorFile.text()) : null
      setUploadPct(0)
      const r = await api.uploadVideo(building, file, { sensors, model, priority, startFloor: validFloor(startFloor) ?? '', onProgress: setUploadPct })
      setFirstWalk(!!map && !(map.videos || []).some((v) => v.status === 'done'))
      setJobId(r.job_id)
      setInspect(null)
    } catch (e) {
      setError(detail(e))
    } finally {
      setUploadPct(null)
    }
  }

  const rebuild = async () => {
    if (!requireSignIn('Sign in to rebuild the map. Browsing stays open.')) return
    if (!confirm('Rebuild the whole map from all videos? Manual edits will be lost (Gemma answers are cached).')) return
    try {
      const r = await api.rebuild(building)
      setFirstWalk(false)
      setJobId(r.job_id)
    } catch (e) {
      setError(detail(e))
    }
  }

  return (
    <>
      <section className="card">
        <h2>Add a walkthrough</h2>
        <div className="status-stack">
          <AnimatePresence>
            <motion.div key={health?.gemma?.ok ? 'ok' : 'bad'} className={`status-line ${health?.gemma?.ok ? 'ok' : 'bad'}`}
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={reduced(reduce, FADE)}>
              <Cpu size={16} />
              {health?.gemma?.ok
                ? `Gemma ready locally (${health.gemma.selected}) · runs offline`
                : `Gemma offline: ${health?.gemma?.error || 'checking…'} — mapping will use motion only`}
            </motion.div>
          </AnimatePresence>
        </div>
        <label className="field">
          <span>Video (any format: .mov, .mp4, .webm…)</span>
          <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files[0])} />
        </label>
        {file && <VideoFileCard key={`${file.name}-${file.size}-${file.lastModified}`} file={file} />}
        <label className="field">
          <span>Phone sensor log (optional, from the Record tab or Sensor Logger)</span>
          <input type="file" accept="application/json,.json" onChange={(e) => setSensorFile(e.target.files[0])} />
        </label>
        {gemmaModels.length > 1 && (
          <label className="field">
            <span>Gemma model</span>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              <option value="">Default ({health.gemma.selected})</option>
              {gemmaModels.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>
        )}
        <label className="field">
          <span>Starts on floor (keeps floors from stacking; leave blank to inherit from places it passes)</span>
          <input type="number" min={1} max={MAX_FLOOR} placeholder="e.g. 1" value={startFloor} onChange={(e) => setStartFloor(e.target.value)} />
          {startFloor !== '' && validFloor(startFloor) == null && <small className="error">Floor must be a whole number from 1 to {MAX_FLOOR}; it will be ignored.</small>}
        </label>
        <label className="check">
          <input type="checkbox" className="tick-box" checked={priority} onChange={(e) => setPriority(e.target.checked)} />
          Process next (pauses the current video; it resumes afterwards)
        </label>
        {error && <div className="banner error" role="alert">{error}</div>}
        <button className={`primary upload-btn ${uploadPct !== null ? 'is-uploading' : ''}`} disabled={!file || uploadPct !== null} onClick={upload}>
          {uploadPct !== null && <span className="btn-fill" aria-hidden="true" style={{ transform: `translateX(${Math.round((uploadPct - 1) * 100)}%)` }} />}
          <UploadCloud size={18} />
          {uploadPct !== null ? `Uploading ${Math.round(uploadPct * 100)}%` : 'Upload & process'}
        </button>
        <RecordingTips />
      </section>

      <QueuePanel building={building} selected={jobId} onSelect={setJobId} />

      {jobId && (
        <JobProgress
          jobId={jobId}
          celebrate={firstWalk}
          onFinished={(job) => {
            reload()
            onDone?.()
            if (job.video_id) setInspect(job.video_id)
          }}
        />
      )}

      {inspect && <VideoDebug building={building} videoId={inspect} />}

      <section className="card">
        <div className="row spread">
          <h3>Processed videos</h3>
          <button className="ghost" onClick={rebuild} disabled={!map?.videos?.length}>
            <RefreshCw size={16} /> Rebuild map
          </button>
        </div>
        {!map?.videos?.length && <p className="muted">None yet.</p>}
        <ul className="video-list">
          <AnimatePresence>
          {map?.videos?.map((v, k) => (
            // Rows stagger in; a deleted row stays until its height has closed, then it goes.
            <motion.li key={v.id} style={{ overflow: 'hidden' }} initial={reduce ? false : { opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0, height: 'auto', transition: reduced(reduce, rowIn(k)) }}
              exit={{ opacity: 0, height: 0, paddingTop: 0, paddingBottom: 0, transition: reduced(reduce, COLLAPSE) }}>
              <span>
                {v.filename} <small className="muted">· <FlowNumber value={v.duration} suffix="s" /> · <FlowNumber value={v.keyframes} /> keyframes{v.has_sensors ? ' · gyro' : ''}</small>
              </span>
              <StatusPill status={v.status} />
              <label className="floor-input" title="Floor this walk starts on">
                Floor
                <input type="number" min={1} max={MAX_FLOOR} placeholder="1"
                  defaultValue={v.start_floor ?? (v.floor_offset >= 1 ? v.floor_offset : '')}
                  onBlur={(e) => {
                    const shown = String(v.start_floor ?? (v.floor_offset >= 1 ? v.floor_offset : ''))
                    const f = validFloor(e.target.value)
                    if (e.target.value === shown) return
                    if (f == null) { setError(`Floor must be a whole number from 1 to ${MAX_FLOOR}`); return }
                    if (!requireSignIn('Sign in to change a walk\'s floor. Browsing stays open.')) return
                    api.setVideoFloor(building, v.id, f).then(reload).catch((err) => setError(detail(err)))
                  }} />
              </label>
              {v.status === 'done' && (
                <button className="ghost" onClick={() => setInspect(v.id)}>
                  <Search size={14} /> Inspect
                </button>
              )}
              <button className="ghost danger" title="Remove this video and its data"
                onClick={async () => {
                  if (!requireSignIn('Sign in to remove a walk. Browsing stays open.')) return
                  if (!confirm(`Remove ${v.filename}? Its processing stops and its places, photos and files are deleted. Places other videos also pass through are kept.`)) return
                  try {
                    const r = await api.deleteVideo(building, v.id)
                    if (inspect === v.id) setInspect(null)
                    setError('')
                    alert(`Removed ${v.filename}: ${r.removed_places} places, ${r.deleted_files} photos deleted.`)
                    reload()
                    onDone?.()
                  } catch (err) {
                    setError(detail(err))
                  }
                }}>
                <Trash2 size={14} />
              </button>
            </motion.li>
          ))}
          </AnimatePresence>
        </ul>
      </section>
    </>
  )
}
