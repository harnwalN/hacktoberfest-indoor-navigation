import { useEffect, useMemo, useState } from 'react'
import { Eye } from 'lucide-react'
import { api, detail } from '../api'

const W = 800
const H = 180
const PAD = 28

// "What the AI saw": integrated heading over time with detected events, and every keyframe with
// the structured JSON Gemma returned for it.
export default function VideoDebug({ building, videoId }) {
  const [d, setD] = useState(null)
  const [sel, setSel] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    setD(null)
    setSel(null)
    api.videoDebug(building, videoId).then(setD).catch((e) => setError(detail(e)))
  }, [building, videoId])

  const chart = useMemo(() => {
    if (!d?.t?.length) return null
    const tMax = d.t[d.t.length - 1] || 1
    const hMin = Math.min(...d.heading) - 20
    const hMax = Math.max(...d.heading) + 20
    const x = (t) => PAD + ((W - 2 * PAD) * t) / tMax
    const y = (h) => H - PAD - ((H - 2 * PAD) * (h - hMin)) / (hMax - hMin || 1)
    const path = d.t.map((t, k) => `${k ? 'L' : 'M'}${x(t).toFixed(1)},${y(d.heading[k]).toFixed(1)}`).join('')
    const ticks = []
    for (let h = Math.ceil(hMin / 90) * 90; h <= hMax; h += 90) ticks.push(h)
    return { x, y, path, ticks, tMax }
  }, [d])

  if (error) return <section className="card banner error">{error}</section>
  if (!d) return <section className="card">Loading analysis…</section>

  const kf = sel != null ? d.keyframes[sel] : null
  return (
    <section className="card">
      <h3>
        <Eye size={18} /> What the pipeline saw <small className="muted">heading from {d.source === 'imu' ? 'phone gyro' : 'optical flow'}</small>
      </h3>
      {chart && (
        <svg viewBox={`0 0 ${W} ${H}`} className="heading-chart">
          {d.events.map((e, k) => (
            <rect key={k} x={chart.x(e.start)} width={Math.max(2, chart.x(e.end) - chart.x(e.start))} y={PAD / 2}
              height={H - 1.5 * PAD} className={`ev ${e.type}`}>
              <title>
                {e.type} {e.start.toFixed(1)}–{e.end.toFixed(1)}s {e.delta ? `${e.delta > 0 ? '+' : ''}${e.delta}°` : ''}
              </title>
            </rect>
          ))}
          {chart.ticks.map((h) => (
            <g key={h}>
              <line x1={PAD} x2={W - PAD} y1={chart.y(h)} y2={chart.y(h)} className="grid" />
              <text x={4} y={chart.y(h) + 4} className="axis">
                {h}°
              </text>
            </g>
          ))}
          <path d={chart.path} className="heading-line" />
          {d.keyframes.map((k, idx) => (
            <circle key={k.frame_id} cx={chart.x(k.t)} cy={chart.y(k.heading)} r={sel === idx ? 6 : 3.5}
              className={`kf ${k.analysis?.has_sign ? 'sign' : ''}`} onClick={() => setSel(idx)} />
          ))}
          <text x={W - PAD} y={H - 6} className="axis" textAnchor="end">
            {chart.tMax.toFixed(0)}s
          </text>
        </svg>
      )}
      <div className="legend">
        <span className="ev turn">turn</span>
        <span className="ev glance">glance</span>
        <span className="ev dwell">pause</span>
        <span className="kf-dot sign">sign read by Gemma</span>
      </div>
      <div className="strip">
        {d.keyframes.map((k, idx) => (
          <button key={k.frame_id} className={`thumb ${sel === idx ? 'sel' : ''}`} onClick={() => setSel(idx)}>
            <img src={k.url} loading="lazy" alt="" />
            <span>
              {k.t.toFixed(1)}s · {k.kind}
              {k.soft ? ' · soft' : ''}
            </span>
            {k.analysis?.has_sign && <b>{(k.analysis.room_numbers[0] || k.analysis.texts[0] || '').slice(0, 14)}</b>}
          </button>
        ))}
      </div>
      {kf && (
        <div className="kf-detail">
          <img src={kf.url} alt="" />
          <pre>
            {JSON.stringify(
              { t: kf.t, kind: kf.kind, heading: kf.heading, sharpness: kf.score, gemma: kf.analysis },
              null,
              2,
            )}
          </pre>
        </div>
      )}
    </section>
  )
}
