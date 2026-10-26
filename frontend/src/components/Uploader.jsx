import { useEffect, useState } from 'react'
import { Cpu, RefreshCw, Search, UploadCloud } from 'lucide-react'
import { api, detail } from '../api'
import JobProgress from './JobProgress'
import QueuePanel from './QueuePanel'
import VideoDebug from './VideoDebug'

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
  const [inspect, setInspect] = useState(null)
  const [error, setError] = useState('')

  const reload = () => api.getMap(building).then(setMap).catch((e) => setError(detail(e)))
  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null))
    reload()
  }, [building])

  const gemmaModels = (health?.gemma?.models || []).filter((m) => m.startsWith('gemma'))

  const upload = async () => {
    setError('')
    try {
      const sensors = sensorFile ? JSON.parse(await sensorFile.text()) : null
      setUploadPct(0)
      const r = await api.uploadVideo(building, file, { sensors, model, priority, startFloor, onProgress: setUploadPct })
      setJobId(r.job_id)
      setInspect(null)
    } catch (e) {
      setError(detail(e))
    } finally {
      setUploadPct(null)
    }
  }

  const rebuild = async () => {
    if (!confirm('Rebuild the whole map from all videos? Manual edits will be lost (Gemma answers are cached).')) return
    const r = await api.rebuild(building)
    setJobId(r.job_id)
  }

  return (
    <>
      <section className="card">
        <h2>Add a walkthrough</h2>
        <div className={`status-line ${health?.gemma?.ok ? 'ok' : 'bad'}`}>
          <Cpu size={16} />
          {health?.gemma?.ok
            ? `Gemma ready locally (${health.gemma.selected}) · runs offline`
            : `Gemma offline: ${health?.gemma?.error || 'checking…'} — mapping will use motion only`}
        </div>
        <label className="field">
          <span>Video (any format: .mov, .mp4, .webm…)</span>
          <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files[0])} />
        </label>
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
          <input type="number" placeholder="e.g. 1" value={startFloor} onChange={(e) => setStartFloor(e.target.value)} />
        </label>
        <label className="check">
          <input type="checkbox" checked={priority} onChange={(e) => setPriority(e.target.checked)} />
          Process next (pauses the current video; it resumes afterwards)
        </label>
        {error && <div className="banner error">{error}</div>}
        <button className="primary" disabled={!file || uploadPct !== null} onClick={upload}>
          <UploadCloud size={18} />
          {uploadPct !== null ? `Uploading ${Math.round(uploadPct * 100)}%` : 'Upload & process'}
        </button>
        <details className="tips">
          <summary>Recording tips</summary>
          <ul>
            <li>1080p, stabilization on, phone upright at chest height, camera pointing where you walk.</li>
            <li>Walk at a steady pace and turn slowly at junctions.</li>
            <li>
              <b>Pause ~1 s</b> at every door plaque, junction and staircase — pauses become sharp keyframes.
            </li>
            <li>Glancing at a plaque and back is fine: it is detected as a glance, not a turn.</li>
            <li>Walk corridors both ways when you can, and come back past known places to close loops.</li>
          </ul>
        </details>
      </section>

      <QueuePanel building={building} selected={jobId} onSelect={setJobId} />

      {jobId && (
        <JobProgress
          jobId={jobId}
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
          {map?.videos?.map((v) => (
            <li key={v.id}>
              <span>
                {v.filename} <small className="muted">· {v.duration}s · {v.keyframes} keyframes{v.has_sensors ? ' · gyro' : ''}</small>
              </span>
              <span className={`pill ${v.status}`}>{v.status}</span>
              <label className="floor-input" title="Floor this walk starts on">
                Floor
                <input type="number" defaultValue={v.start_floor ?? v.floor_offset ?? 0}
                  onBlur={(e) => e.target.value !== '' && api.setVideoFloor(building, v.id, e.target.value).then(reload).catch((err) => setError(detail(err)))} />
              </label>
              {v.status === 'done' && (
                <button className="ghost" onClick={() => setInspect(v.id)}>
                  <Search size={14} /> Inspect
                </button>
              )}
            </li>
          ))}
        </ul>
      </section>
    </>
  )
}
