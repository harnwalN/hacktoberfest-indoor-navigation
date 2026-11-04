import { ArrowRight } from 'lucide-react'

// The landing page (#/ or #/landing). Presentational: App passes the building list it already loads.
// Every link is a hash link, so App's router does the work (no state passed back up).
function scrollToHow() {
  const el = document.getElementById('how')
  if (!el) return
  const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
  el.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' })
  el.focus({ preventScroll: true })
}

export default function Landing({ signedIn, loaded, fetchFailed, buildings }) {
  // Contributing needs an identity for the upload, so signed-out visitors go to sign-in first.
  const contribute = signedIn ? '#/build' : '#/sign-in'

  return (
    <main className="content landing" id="landing-main">
      <section className="hero" aria-labelledby="hero-title">
        <p className="eyebrow-chip">UC SAN DIEGO · INDOOR NAVIGATION</p>
        <h1 id="hero-title" className="display">
          Find any room, <em>step by step.</em>
        </h1>
        <p className="lead">
          Routes through campus buildings, built from walk videos that people record and anyone can check. Pick a
          building, say where you are, and follow the steps.
        </p>
        <div className="pill-row">
          <a className="pill pill-solid" href="#/navigate">
            Get directions <ArrowRight size={16} aria-hidden="true" />
          </a>
          <a className="pill" href={contribute}>Contribute a walk</a>
          <button type="button" className="pill" onClick={scrollToHow}>How it works</button>
        </div>
      </section>

      <section className="chooser" aria-labelledby="chooser-title">
        <h2 id="chooser-title" className="section-title">Choose a building</h2>
        {!loaded ? (
          <p className="muted" role="status">Loading buildings…</p>
        ) : fetchFailed && !buildings.length ? (
          <p className="muted" role="status">Could not load buildings. Try again in a moment.</p>
        ) : !buildings.length ? (
          <p className="muted" role="status">No buildings with maps yet.</p>
        ) : (
          <ul className="building-list">
            {buildings.map((b) => (
              <li key={b.id}>
                <a className="pill" href={`#/navigate?b=${encodeURIComponent(b.id)}`}>
                  {b.name}
                  {b.campus ? <span className="pill-sub">{b.campus}</span> : null}
                </a>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section id="how" tabIndex={-1} className="glass-card" aria-labelledby="how-title">
        <div className="glass-head">
          <p className="eyebrow-chip eyebrow-soft">HOW IT WORKS</p>
          <h2 id="how-title" className="display card-title">Walk it once. Everyone finds the way.</h2>
        </div>
        <ol className="how-steps">
          <li><span className="step-num">01</span><span>Record a walk</span></li>
          <li><span className="step-num">02</span><span>We map it</span></li>
          <li><span className="step-num">03</span><span>Anyone navigates</span></li>
        </ol>
        <a className="pill pill-solid" href="#/review">
          Explore <ArrowRight size={16} aria-hidden="true" />
        </a>
      </section>
    </main>
  )
}
