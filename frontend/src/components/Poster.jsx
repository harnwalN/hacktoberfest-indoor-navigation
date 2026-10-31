import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { Printer, X } from 'lucide-react'

// The QR encoder (qrcode-generator, MIT, no dependencies) is loaded only when a poster is opened, so the main bundle
// does not carry it. Encoding happens here in the browser: the link never goes to an outside service.
let qrLib = null
const loadQr = () => (qrLib ||= import('qrcode-generator').then((m) => m.default))

const QUIET = 4 // quiet zone, in modules, around the code (the standard minimum)

// The code as one SVG path: a unit square for each dark module. Built from numbers only, so no markup is ever
// assembled from the link text.
export function encodeQr(qrcode, text) {
  const qr = qrcode(0, 'M')
  qr.addData(text)
  qr.make()
  const n = qr.getModuleCount()
  let d = ''
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      if (qr.isDark(r, c)) d += `M${c + QUIET} ${r + QUIET}h1v1h-1z`
    }
  }
  return { size: n + QUIET * 2, path: d, modules: n }
}

// Entrance poster for a viewer link: the building name, a large code for the full join link, the call to action and
// the link written out below it. It is shown in a full-screen layer over the app; only the poster prints.
export default function Poster({ name, link, onClose, returnFocusTo }) {
  const [code, setCode] = useState(null)
  const [err, setErr] = useState('')
  const dialog = useRef(null)
  const closeRef = useRef(onClose)
  closeRef.current = onClose
  const title = name || 'this building'

  useEffect(() => {
    let live = true
    loadQr()
      .then((qrcode) => {
        if (live) setCode(encodeQr(qrcode, link))
      })
      .catch(() => {
        if (live) setErr('The code could not be made. Copy the link instead.')
      })
    return () => {
      live = false
    }
  }, [link])

  // A modal layer: the app underneath is inert while it is open, Escape closes it, and focus goes back on close.
  useEffect(() => {
    const root = document.getElementById('root')
    const wasInert = root ? root.inert : false
    if (root) root.inert = true
    const onKey = (e) => {
      if (e.key === 'Escape') closeRef.current()
    }
    document.addEventListener('keydown', onKey)
    dialog.current?.querySelector('button')?.focus()
    return () => {
      document.removeEventListener('keydown', onKey)
      if (root) root.inert = wasInert
      returnFocusTo?.current?.focus()
    }
  }, [returnFocusTo])

  return createPortal(
    <div className="poster-root" ref={dialog} role="dialog" aria-modal="true" aria-label={`Entrance poster for ${title}`}>
      <div className="poster-bar no-print">
        <p>
          <b>Anyone who scans this poster gets viewer access to {title}.</b> Revoke the invite in the list to switch the poster off.
        </p>
        <div className="row wrap">
          <button type="button" className="primary" onClick={() => window.print()}>
            <Printer size={16} aria-hidden="true" /> Print poster
          </button>
          <button type="button" className="ghost" onClick={() => closeRef.current()}>
            <X size={16} aria-hidden="true" /> Close
          </button>
        </div>
      </div>

      <article className="poster-sheet">
        <h1 className="poster-name">{name || 'Building'}</h1>
        {code ? (
          <svg className="poster-qr" viewBox={`0 0 ${code.size} ${code.size}`} role="img" aria-label="QR code for the invite link"
            shapeRendering="crispEdges" data-modules={code.modules}>
            <rect width={code.size} height={code.size} fill="#ffffff" />
            <path d={code.path} fill="#000000" />
          </svg>
        ) : err ? (
          <p className="notice error poster-error" role="alert">{err}</p>
        ) : (
          <p className="poster-wait no-print" role="status">Making the code…</p>
        )}
        <p className="poster-cta">Scan for directions inside {title}</p>
        <p className="poster-link">{link}</p>
      </article>
    </div>,
    document.body,
  )
}
