import { useEffect, useRef, useState } from 'react'
import { CheckCircle2, Clock, Loader2, XCircle } from 'lucide-react'
import { api, detail } from '../api'
import { Confetti } from './Motion'

const STAGES = [
  ['normalize', 'Normalize video'],
  ['motion', 'Camera motion & turns'],
  ['keyframes', 'Pick sharp keyframes'],
  ['gemma', 'Gemma reads signs'],
  ['mapping', 'Build graph & close loops'],
]

const WAIT = {
  running: 'Your map will be ready here when this finishes.',
  queued: 'Waiting its turn in the queue.',
  paused: 'Paused. It carries on from where it stopped.',
  error: 'Processing stopped. The details are below.',
}

export default function JobProgress({ jobId, onFinished }) {
  const [job, setJob] = useState(null)
  const [error, setError] = useState('')
  const logRef = useRef(null)
  const finished = useRef(false)
  const sawRunning = useRef(false)

  useEffect(() => {
    finished.current = false
    let timer
    const poll = async () => {
      try {
        const j = await api.job(jobId)
        if (j.status === 'running' || j.status === 'queued') sawRunning.current = true
        setJob(j)
        if (j.status === 'done' || j.status === 'error') {
          if (!finished.current) {
            finished.current = true
            onFinished?.(j)
          }
          return
        }
      } catch (e) {
        setError(detail(e))
      }
      timer = setTimeout(poll, 1000)
    }
    poll()
    return () => clearTimeout(timer)
  }, [jobId])

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [job?.log?.length])

  if (!job) return <section className="card">{error || 'Starting…'}</section>
  const current = STAGES.findIndex(([s]) => s === job.stage)

  return (
    <section className="card">
      <div className="row spread">
        <h3>Processing {job.kind === 'rebuild' ? '(rebuild)' : ''}</h3>
        <span className={`pill ${job.status}`}>{job.status}</span>
      </div>
      {/* Fixed slot: holds a short waiting note until the job finishes, then the "ready" banner takes its place
          at the same height, so nothing below it moves. */}
      <div className="job-slot">
        {job.status === 'done' ? (
          <div className="success-pop" role="status">
            {sawRunning.current && <Confetti count={26} spread={170} />}
            <svg className="success-check" viewBox="0 0 52 52" aria-hidden="true" focusable="false">
              <circle cx="26" cy="26" r="24" className="sc-disc" />
              <path d="M15 27 L23 35 L38 18" pathLength="1" className="sc-tick" />
            </svg>
            <div>
              <b>Map is ready</b>
              <small>The new places are on your map.</small>
            </div>
          </div>
        ) : (
          <p className={`job-wait ${job.status === 'error' ? 'error' : ''}`}>
            {job.status === 'error' ? <XCircle size={20} aria-hidden="true" /> : <Clock size={20} aria-hidden="true" />}
            {WAIT[job.status] || WAIT.running}
          </p>
        )}
      </div>
      <div className="bar" role="progressbar" aria-label="Processing progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(job.pct)}>
        <div style={{ transform: `translateX(${Math.min(0, Math.max(-100, job.pct - 100))}%)` }} className={job.status} />
      </div>
      <ol className="stages">
        {STAGES.map(([id, label], k) => {
          const done = job.status === 'done' || k < current
          const active = k === current && job.status === 'running'
          return (
            <li key={id} className={done ? 'done' : active ? 'active' : ''} style={{ '--i': k }}>
              {done ? <CheckCircle2 size={16} className="tick" /> : active ? <Loader2 size={16} className="spin" /> : job.status === 'error' && k === current ? <XCircle size={16} /> : <span className="dot" />}
              {label}
            </li>
          )
        })}
      </ol>
      {job.error && <div className="banner error">{job.error}</div>}
      <pre className="log" ref={logRef}>
        {job.log.join('\n')}
      </pre>
    </section>
  )
}
