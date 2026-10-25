import { useEffect, useRef, useState } from 'react'
import { Circle, Square, UploadCloud } from 'lucide-react'
import { api, detail } from '../api'

// Records the walkthrough in the phone browser AND logs the gyroscope on the same clock, so the
// backend can use real rotation rates for turns instead of inferring them from pixels.
// iOS requires HTTPS (use the cloudflared URL) and a tap to grant motion access.
function pickMime() {
  const options = ['video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm']
  return options.find((m) => window.MediaRecorder?.isTypeSupported?.(m)) || ''
}

export default function Recorder({ building, onUploaded }) {
  const videoRef = useRef(null)
  const recRef = useRef(null)
  const chunks = useRef([])
  const samples = useRef([])
  const t0 = useRef(0)
  const [stream, setStream] = useState(null)
  const [recording, setRecording] = useState(false)
  const [result, setResult] = useState(null)
  const [motion, setMotion] = useState('unknown')
  const [elapsed, setElapsed] = useState(0)
  const [status, setStatus] = useState('')

  useEffect(() => () => stream?.getTracks().forEach((t) => t.stop()), [stream])

  useEffect(() => {
    if (!recording) return
    const id = setInterval(() => setElapsed((performance.now() - t0.current) / 1000), 250)
    return () => clearInterval(id)
  }, [recording])

  const onMotion = (e) => {
    if (!recRef.current || recRef.current.state !== 'recording') return
    const r = e.rotationRate
    const g = e.accelerationIncludingGravity
    if (!r || !g) return
    samples.current.push({ t: (performance.now() - t0.current) / 1000, rr: [r.alpha || 0, r.beta || 0, r.gamma || 0], g: [g.x || 0, g.y || 0, g.z || 0] })
  }

  const start = async () => {
    setStatus('')
    try {
      if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
        setMotion((await DeviceMotionEvent.requestPermission()) === 'granted' ? 'granted' : 'denied')
      } else {
        setMotion(typeof DeviceMotionEvent !== 'undefined' ? 'granted' : 'unavailable')
      }
      const s = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 } },
        audio: false,
      })
      setStream(s)
      videoRef.current.srcObject = s
      await videoRef.current.play()
    } catch (e) {
      setStatus(`Camera/motion error: ${e.message}. Use the HTTPS (cloudflared) URL on iPhone.`)
    }
  }

  const record = () => {
    chunks.current = []
    samples.current = []
    setResult(null)
    const mimeType = pickMime()
    const rec = new MediaRecorder(stream, mimeType ? { mimeType, videoBitsPerSecond: 8_000_000 } : undefined)
    rec.ondataavailable = (e) => e.data.size && chunks.current.push(e.data)
    rec.onstop = () => {
      const type = rec.mimeType || 'video/mp4'
      const blob = new Blob(chunks.current, { type })
      const ext = type.includes('webm') ? 'webm' : 'mp4'
      setResult({ file: new File([blob], `walk-${Date.now()}.${ext}`, { type }), sensors: { samples: samples.current, offset_sec: 0, source: 'web-devicemotion' } })
      window.removeEventListener('devicemotion', onMotion)
    }
    window.addEventListener('devicemotion', onMotion)
    t0.current = performance.now()
    rec.start(1000)
    recRef.current = rec
    setRecording(true)
  }

  const stop = () => {
    recRef.current?.stop()
    setRecording(false)
  }

  const upload = async () => {
    setStatus('Uploading…')
    try {
      await api.uploadVideo(building, result.file, {
        sensors: result.sensors.samples.length > 20 ? result.sensors : null,
        onProgress: (p) => setStatus(`Uploading ${Math.round(p * 100)}%`),
      })
      setStatus('Uploaded. Processing has started on the laptop.')
      onUploaded?.()
    } catch (e) {
      setStatus(detail(e))
    }
  }

  return (
    <section className="card recorder">
      <h2>Record a walkthrough</h2>
      <p className="muted">Hold the phone upright, point it where you walk, pause ~1 s at signs and junctions.</p>
      <div className="viewfinder">
        <video ref={videoRef} playsInline muted />
        {recording && <div className="rec-badge">● REC {elapsed.toFixed(0)}s · {samples.current.length} gyro samples</div>}
      </div>
      <p className="muted">Motion sensors: {motion}</p>
      {!stream && <button className="primary big" onClick={start}>Enable camera & motion</button>}
      {stream && !recording && <button className="primary big" onClick={record}><Circle size={18} /> Start recording</button>}
      {recording && <button className="danger big" onClick={stop}><Square size={18} /> Stop</button>}
      {result && !recording && (
        <button className="primary big" onClick={upload}>
          <UploadCloud size={18} /> Upload ({(result.file.size / 1e6).toFixed(0)} MB, {result.sensors.samples.length} gyro samples)
        </button>
      )}
      {status && <div className="banner">{status}</div>}
    </section>
  )
}
