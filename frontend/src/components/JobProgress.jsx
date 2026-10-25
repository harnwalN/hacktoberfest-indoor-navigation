import { useEffect, useRef, useState } from 'react'
import { CheckCircle2, Loader2, XCircle } from 'lucide-react'
import { api, detail } from '../api'

const STAGES = [
  ['normalize', 'Normalize video'],
  ['motion', 'Camera motion & turns'],
  ['keyframes', 'Pick sharp keyframes'],
  ['gemma', 'Gemma reads signs'],
  ['mapping', 'Build graph & close loops'],
]

export default function JobProgress({ jobId, onFinished }) {
  const [job, setJob] = useState(null)
  const [error, setError] = useState('')
  const logRef = useRef(null)
  const finished = useRef(false)

  useEffect(() => {
    finished.current = false
    let timer
    const poll = async () => {
      try {
        const j = await api.job(jobId)
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
      <div className="bar">
        <div style={{ width: `${job.pct}%` }} className={job.status} />
      </div>
      <ol className="stages">
        {STAGES.map(([id, label], k) => {
          const done = job.status === 'done' || k < current
          const active = k === current && job.status === 'running'
          return (
            <li key={id} className={done ? 'done' : active ? 'active' : ''}>
              {done ? <CheckCircle2 size={16} /> : active ? <Loader2 size={16} className="spin" /> : job.status === 'error' && k === current ? <XCircle size={16} /> : <span className="dot" />}
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
