import { useEffect, useState } from 'react'
import { ListOrdered, Pause, Play, SkipForward, X } from 'lucide-react'
import { api, detail } from '../api'

// Processing queue: one video runs at a time. Pause/resume any job, or "Run next" to jump the queue
// (the running video pauses and resumes right after; Gemma answers are cached so nothing is lost).
export default function QueuePanel({ building, selected, onSelect }) {
  const [jobs, setJobs] = useState([])
  const [error, setError] = useState('')

  useEffect(() => {
    let timer
    const poll = async () => {
      try {
        setJobs(await api.queue(building))
        setError('')
      } catch (e) {
        setError(detail(e))
      }
      timer = setTimeout(poll, 2000)
    }
    poll()
    return () => clearTimeout(timer)
  }, [building])

  const act = async (id, action) => {
    try {
      await api.jobAction(id, action)
      setJobs(await api.queue(building))
    } catch (e) {
      setError(detail(e))
    }
  }

  const active = jobs.filter((j) => ['running', 'queued', 'paused'].includes(j.status))
  const finished = jobs.filter((j) => !['running', 'queued', 'paused'].includes(j.status)).slice(0, 5)
  if (!jobs.length) return null

  return (
    <section className="card">
      <h3>
        <ListOrdered size={18} /> Queue <small className="muted">{active.length} active</small>
      </h3>
      {error && (
        <div className="banner error banner-dismiss" role="alert">
          <span>{error}</span>
          <button type="button" className="ghost icon-btn" onClick={() => setError('')} aria-label="Dismiss error">
            <X size={18} aria-hidden="true" />
          </button>
        </div>
      )}
      <ul className="queue">
        {[...active, ...finished].map((j, k) => (
          <li key={j.id} className={`${j.status} ${selected === j.id ? 'sel' : ''}`}>
            <button type="button" className="ghost q-main" onClick={() => onSelect?.(j.id)} aria-pressed={selected === j.id}>
              <span className="q-name">{j.filename || j.kind}</span>
              <span className={`pill ${j.status}`}>{j.status === 'running' ? `${j.stage} ${Math.round(j.pct)}%` : j.status}</span>
            </button>
            {j.status === 'running' && (
              <div className="bar slim">
                <div style={{ width: `${j.pct}%` }} />
              </div>
            )}
            <div className="q-actions">
              {(j.status === 'running' || j.status === 'queued') && (
                <button className="ghost" onClick={() => act(j.id, 'pause')} title="Pause">
                  <Pause size={14} /> Pause
                </button>
              )}
              {(j.status === 'paused' || j.status === 'error') && (
                <button className="ghost" onClick={() => act(j.id, 'resume')} title="Resume">
                  <Play size={14} /> Resume
                </button>
              )}
              {(j.status === 'queued' || j.status === 'paused') && k > 0 && (
                <button className="ghost" onClick={() => act(j.id, 'run-next')} title="Process this now">
                  <SkipForward size={14} /> Run next
                </button>
              )}
            </div>
          </li>
        ))}
      </ul>
    </section>
  )
}
