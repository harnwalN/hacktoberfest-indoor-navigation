import { useEffect, useRef, useState } from 'react'
import { ArrowRight, Briefcase, Compass, GraduationCap, Mail } from 'lucide-react'
import { api, detail } from '../api'
import { useAuth } from './AuthContext'

const EMAIL_OK = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

// Who is signing in. Only the helper text changes: the backend takes the same email either way.
const AUDIENCES = {
  uc: {
    label: 'UC San Diego',
    icon: GraduationCap,
    help: 'Use your UC San Diego email. We send a one-time sign-in link. No password needed.',
    placeholder: 'you@ucsd.edu',
  },
  partner: {
    label: 'Join as a partner',
    icon: Briefcase,
    help: 'Use the email your organisation gave you. We send a one-time sign-in link. No password needed.',
    placeholder: 'you@example.com',
  },
}

// The full-page sign-in (#/sign-in). Same magic-link flow as the sign-in sheet: request a link, open it in this browser.
export default function SignInPage() {
  const { signedIn, account, signOut } = useAuth()
  const [who, setWho] = useState('uc')
  const [email, setEmail] = useState('')
  const [status, setStatus] = useState('idle') // idle | sending | sent | error
  const [err, setErr] = useState('')
  const inFlight = useRef(false)
  const input = useRef(null)
  const sentHead = useRef(null)
  const aud = AUDIENCES[who]

  // After a successful send the form is replaced by the sent note: move focus to its heading so a keyboard or
  // screen reader user lands on the result, not on the top of the page.
  useEffect(() => {
    if (status === 'sent') sentHead.current?.focus()
  }, [status])

  const send = async (e) => {
    e.preventDefault()
    if (inFlight.current) return
    const addr = email.trim()
    if (!EMAIL_OK.test(addr)) {
      setStatus('error')
      setErr('Enter a valid email address.')
      input.current?.focus()
      return
    }
    inFlight.current = true
    setStatus('sending')
    setErr('')
    try {
      await api.requestLink(addr)
      setStatus('sent')
    } catch (x) {
      setStatus('error')
      setErr(detail(x))
    } finally {
      inFlight.current = false
    }
  }

  return (
    <div className="signin-page">
      <section className="signin-brand" aria-label="About Waypoint">
        <div className="rings" aria-hidden="true">
          <span /><span /><span />
        </div>
        <a href="#/" className="brand brand-light" aria-label="Waypoint home">
          <span className="brand-mark"><Compass size={18} aria-hidden="true" /></span>
          <span className="brand-word">Waypoint</span>
        </a>
        <div className="signin-brand-copy">
          <p className="brand-headline">
            Find your way, <em>one walk at a time.</em>
          </p>
          <p className="brand-body">
            Students, staff and visitors map campus buildings by walking through them. Sign in to add a walk.
            Finding your way never needs an account.
          </p>
        </div>
        <p className="mono-foot">WAYPOINT · INDOOR NAVIGATION · UC SAN DIEGO</p>
      </section>

      <section className="signin-panel">
        <div className="signin-form">
          {signedIn ? (
            <>
              <p className="eyebrow-amber">SIGNED IN</p>
              <h1 className="display">You are <em>signed in.</em></h1>
              <p className="muted">Signed in as {account?.email}.</p>
              <a className="btn-black" href="#/navigate">
                Continue to directions <ArrowRight size={16} aria-hidden="true" />
              </a>
              <button type="button" className="ghost fine-action" onClick={signOut}>Sign out</button>
            </>
          ) : (
            <>
              <div className="segmented" role="group" aria-label="Who is signing in">
                {Object.entries(AUDIENCES).map(([key, a]) => {
                  const Icon = a.icon
                  return (
                    <button
                      key={key}
                      type="button"
                      className={who === key ? 'seg-on' : ''}
                      aria-pressed={who === key}
                      onClick={() => setWho(key)}
                    >
                      <Icon size={16} aria-hidden="true" />
                      <span>{a.label}</span>
                    </button>
                  )
                })}
              </div>

              <p className="eyebrow-amber">SIGN IN</p>
              <h1 className="display">Welcome to <em>Waypoint.</em></h1>
              <p className="muted" id="signin-help">{aud.help}</p>

              {status === 'sent' ? (
                <div role="status" aria-live="polite" className="sent-note">
                  <h2 ref={sentHead} tabIndex={-1} className="sent-title">Check your email for a sign-in link.</h2>
                  <p className="muted">Open the link in this browser to finish signing in.</p>
                  <button type="button" className="ghost fine-action" onClick={() => { setStatus('idle'); setEmail('') }}>
                    Use a different email
                  </button>
                </div>
              ) : (
                <form onSubmit={send} noValidate>
                  <label className="field-label" htmlFor="signin-page-email">Email</label>
                  <input
                    id="signin-page-email"
                    ref={input}
                    type="email"
                    className="field"
                    autoComplete="email"
                    inputMode="email"
                    autoFocus
                    placeholder={aud.placeholder}
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    aria-invalid={status === 'error' ? 'true' : undefined}
                    aria-describedby={err ? 'signin-page-err signin-help' : 'signin-help'}
                  />
                  {err && <p id="signin-page-err" className="notice error" role="alert">{err}</p>}
                  <button type="submit" className="btn-black" disabled={status === 'sending'}>
                    <Mail size={16} aria-hidden="true" />
                    <span>{status === 'sending' ? 'Sending…' : 'Email me a link'}</span>
                    {status !== 'sending' && <ArrowRight size={16} aria-hidden="true" />}
                  </button>
                </form>
              )}

              <p className="fine">
                We only use your email to send you a sign-in link. There is no password. You control your uploads: ask the
                building owner to remove a walk at any time. Finding your way never needs an account.
              </p>
              <a className="back-link" href="#/">Back to home</a>
            </>
          )}
        </div>
      </section>
    </div>
  )
}
