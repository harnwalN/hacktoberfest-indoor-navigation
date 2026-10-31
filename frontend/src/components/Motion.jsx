import { useEffect, useId, useState } from 'react'

// Shared motion helpers and decorative art. Everything here is decorative (aria-hidden) and animated with
// transform/opacity through CSS (styles.css, "motion" section); prefers-reduced-motion switches it all off.

export const reducedMotion = () => typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

const CONFETTI = ['var(--accent)', 'var(--stairs)', 'var(--room-fg)', 'var(--landmark-fg)', 'var(--noexit-fg)', 'var(--accent-deep)']

// A small burst of paper bits from the middle of its (relatively positioned) parent. Deterministic, so a
// re-render never reshuffles it; removes itself when finished.
export function Confetti({ count = 22, spread = 150, className = '' }) {
  const [live, setLive] = useState(() => !reducedMotion())
  useEffect(() => {
    if (!live) return undefined
    const t = setTimeout(() => setLive(false), 1900)
    return () => clearTimeout(t)
  }, [live])
  if (!live) return null
  return (
    <span className={`confetti ${className}`.trim()} aria-hidden="true">
      {Array.from({ length: count }, (_, i) => {
        const a = (i / count) * Math.PI * 2 + ((i * 37) % 11) * 0.05
        const d = spread * (0.55 + ((i * 53) % 10) / 22)
        return (
          <i
            key={i}
            style={{
              '--dx': `${Math.round(Math.cos(a) * d)}px`, '--dy': `${Math.round(Math.sin(a) * d * 0.85 - 30)}px`,
              '--rot': `${((i * 97) % 360) - 180}deg`, '--c': CONFETTI[i % CONFETTI.length],
              '--w': `${6 + ((i * 7) % 5)}px`, '--h': i % 3 === 0 ? '6px' : `${10 + ((i * 3) % 4)}px`,
              '--delay': `${(i % 6) * 14}ms`, borderRadius: i % 4 === 0 ? '50%' : '2px',
            }}
          />
        )
      })}
    </span>
  )
}

// Start screen motif: a dotted route draws itself from "you are here" to a pin that drops in; a dot walks it.
export function PlannerArt() {
  const id = useId().replace(/:/g, '')
  const d = 'M40 80 C 88 14, 150 118, 196 66 S 268 26, 300 44'
  return (
    <svg className="art planner-art" viewBox="0 0 340 110" aria-hidden="true" focusable="false">
      <defs>
        <mask id={`m${id}`} maskUnits="userSpaceOnUse" x="0" y="0" width="340" height="110">
          <path d={d} className="pa-draw" pathLength="1" fill="none" stroke="#fff" strokeWidth="8" strokeLinecap="round" />
        </mask>
      </defs>
      <circle cx="58" cy="60" r="46" className="pa-blob pa-blob--a" fill="var(--accent-soft)" />
      <circle cx="276" cy="52" r="38" className="pa-blob pa-blob--b" fill="var(--stairs-bg)" />
      <g className="pa-tile pa-tile--room">
        <rect x="222" y="80" width="52" height="26" rx="13" fill="var(--room-bg)" />
        <text x="248" y="98" textAnchor="middle" fontSize="13" fontWeight="700" fill="var(--room-fg)" fontFamily="var(--font)">201</text>
      </g>
      <g className="pa-tile pa-tile--stairs">
        <rect x="206" y="6" width="30" height="30" rx="10" fill="var(--stairs-bg)" />
        <path d="M214 28 H220 V22 H226 V16 H230" fill="none" stroke="var(--stairs-fg)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
      </g>
      <path d={d} fill="none" stroke="var(--accent)" strokeWidth="4" strokeLinecap="round" strokeDasharray="0.1 9" mask={`url(#m${id})`} />
      <circle cx="40" cy="80" r="8" fill="var(--surface)" stroke="var(--accent)" strokeWidth="4" />
      <g className="pa-pin">
        <path d="M300 70 C300 70 282 52 282 41 A18 18 0 0 1 318 41 C318 52 300 70 300 70 Z" fill="var(--accent)" />
        <circle cx="300" cy="41" r="6.5" fill="var(--surface)" />
      </g>
      <circle className="pa-walker" r="6" fill="var(--accent-deep)" style={{ offsetPath: `path('${d}')` }} />
    </svg>
  )
}

// Friendly empty state: a little building and a pin bobbing over it.
export function EmptyArt() {
  return (
    <svg className="art empty-art" viewBox="0 0 200 130" aria-hidden="true" focusable="false">
      <ellipse cx="100" cy="118" rx="62" ry="7" fill="var(--well)" />
      <rect x="52" y="44" width="96" height="72" rx="14" fill="var(--surface)" stroke="var(--border)" strokeWidth="3" />
      <rect x="66" y="58" width="18" height="16" rx="5" fill="var(--room-bg)" />
      <rect x="91" y="58" width="18" height="16" rx="5" fill="var(--stairs-bg)" />
      <rect x="116" y="58" width="18" height="16" rx="5" fill="var(--exit-bg)" />
      <rect x="86" y="86" width="28" height="30" rx="8" fill="var(--accent-soft)" />
      <path d="M100 100 V106" stroke="var(--accent)" strokeWidth="3" strokeLinecap="round" />
      <g className="ea-pin">
        <path d="M100 42 C100 42 84 26 84 17 A16 16 0 0 1 116 17 C116 26 100 42 100 42 Z" fill="var(--accent)" />
        <circle cx="100" cy="17" r="5.5" fill="var(--surface)" />
      </g>
      <circle className="ea-spark ea-spark--a" cx="40" cy="36" r="4" fill="var(--stairs)" />
      <circle className="ea-spark ea-spark--b" cx="160" cy="30" r="3" fill="var(--room-fg)" />
    </svg>
  )
}

// Skeleton for the start screen while the map loads.
export function PlannerSkeleton() {
  return (
    <div className="skeleton-planner" aria-hidden="true">
      <div className="sk sk-title" />
      <div className="sk sk-title short" />
      <div className="sk sk-card" />
      <div className="sk sk-btn" />
    </div>
  )
}
