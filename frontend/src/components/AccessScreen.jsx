import { useEffect, useId, useRef, useState } from 'react'
import { KeyRound } from 'lucide-react'
import { api, detail } from '../api'
import { parseInviteLink, setOwnerKey, setToken } from '../access'
import { SignOutOptions } from './SignOutMenu'

// Copy for each reason the app needs an invite. "none": nothing stored for this building. "invalid": a stored invite
// was refused (revoked or expired). "owner": the stored owner key was refused. "missing": the owner key works but
// the building is not on this server. "bad-link": a join link was missing its building or code.
const COPY = {
  none: {
    title: 'This building is invite-only',
    body: 'You need an invite link to open this building. Ask the person who runs it to send you one.',
  },
  invalid: {
    title: "This invite isn't working",
    body: 'The invite for this building was revoked or is no longer valid. Ask the owner for a new link, then paste it below.',
  },
  owner: {
    title: "The owner key isn't working",
    body: 'The owner key saved on this device was refused. Enter it again, or sign out on this device.',
  },
  missing: {
    title: "We couldn't find that building",
    body: 'Pick a building you can open from the list, or paste an invite link below.',
  },
  'bad-link': {
    title: 'That invite link is incomplete',
    body: 'Paste the whole link you were sent. It contains the building and a code after #/join.',
  },
}
const NO_BUILDING = {
  title: 'You need an invite link',
  body: 'Open the invite link you were sent, or paste it below. You have no building open on this device yet.',
}

const linkMessage = (err) => {
  if (err?.response?.status === 401) return "That invite link isn't valid. It may have been revoked. Ask for a new one."
  return detail(err)
}
// Only a 200 from the owner check saves the key. 401 means the key is wrong; 429 means wait. Neither is saved.
const ownerMessage = (err) => {
  if (err?.response?.status === 401) return "That owner key isn't valid."
  if (err?.response?.status === 429) return 'Too many tries, wait a minute.'
  return detail(err)
}

export default function AccessScreen({ building, reason, buildings, hasStored, onGranted, onOpen, onSignOut }) {
  const uid = useId()
  const heading = useRef(null)
  const [link, setLink] = useState('')
  const [linkErr, setLinkErr] = useState('')
  const [linkBusy, setLinkBusy] = useState(false)
  const [ownerOpen, setOwnerOpen] = useState(reason === 'owner')
  const [ownerText, setOwnerText] = useState('')
  const [ownerErr, setOwnerErr] = useState('')
  const [ownerBusy, setOwnerBusy] = useState(false)

  // The screen takes focus when it appears, so a screen reader and a keyboard user land on the explanation.
  useEffect(() => {
    heading.current?.focus()
  }, [])

  const copy = reason === 'none' && !building ? NO_BUILDING : COPY[reason] || COPY.none
  const others = buildings.filter((b) => b.id !== building)

  const openLink = async (e) => {
    e.preventDefault()
    setLinkErr('')
    const parsed = parseInviteLink(link)
    if (!parsed) {
      setLinkErr("That doesn't look like an invite link. Paste the whole link, which contains #/join?b= and a code.")
      return
    }
    setLinkBusy(true)
    try {
      // Check the pasted invite on its own before saving it, so a bad link never replaces a working one.
      await api.checkAccess(parsed.building, parsed.token)
      setToken(parsed.building, parsed.token)
      setLink('')
      onGranted(parsed.building)
    } catch (err) {
      setLinkErr(linkMessage(err))
      setLinkBusy(false)
    }
  }

  const saveOwner = async (e) => {
    e.preventDefault()
    const key = ownerText.trim()
    if (!key) {
      setOwnerErr('Enter the owner key.')
      return
    }
    setOwnerErr('')
    setOwnerBusy(true)
    try {
      await api.checkOwnerKey(key)
      setOwnerKey(key)
      setOwnerText('')
      onGranted('')
    } catch (err) {
      setOwnerErr(ownerMessage(err))
      setOwnerBusy(false)
    }
  }

  const linkErrId = `${uid}-link-err`
  const ownerPanelId = `${uid}-owner`
  const ownerErrId = `${uid}-owner-err`

  return (
    <section className="card access-screen" aria-labelledby={`${uid}-title`}>
      <h2 id={`${uid}-title`} tabIndex={-1} ref={heading}>{copy.title}</h2>
      <p>{copy.body}</p>

      <form className="access-form" onSubmit={openLink} noValidate>
        <label className="field">
          <span>Paste an invite link</span>
          <input
            type="text"
            value={link}
            onChange={(e) => {
              setLink(e.target.value)
              if (linkErr) setLinkErr('')
            }}
            aria-invalid={linkErr ? 'true' : undefined}
            aria-describedby={linkErr ? linkErrId : undefined}
            autoComplete="off"
            autoCapitalize="off"
            autoCorrect="off"
            spellCheck={false}
            placeholder="Paste the whole link"
          />
        </label>
        {linkErr && <p id={linkErrId} role="alert" className="notice error">{linkErr}</p>}
        <button type="submit" className="primary" disabled={linkBusy || !link.trim()}>
          {linkBusy ? 'Checking the link…' : 'Open building'}
        </button>
      </form>

      {others.length > 0 && (
        <div className="access-others">
          <p className="muted">Or open a building you already have access to:</p>
          <div className="row wrap">
            {others.map((b) => (
              <button key={b.id} type="button" className="ghost" onClick={() => onOpen(b.id)}>
                {b.name || b.id}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="access-owner">
        <button
          type="button"
          className="ghost"
          aria-expanded={ownerOpen}
          aria-controls={ownerPanelId}
          onClick={() => setOwnerOpen((o) => !o)}
        >
          <KeyRound size={16} aria-hidden="true" /> I own this server
        </button>
        <form id={ownerPanelId} className="access-form" onSubmit={saveOwner} noValidate hidden={!ownerOpen}>
          <label className="field">
            <span>Owner key</span>
            <input
              type="password"
              value={ownerText}
              onChange={(e) => {
                setOwnerText(e.target.value)
                if (ownerErr) setOwnerErr('')
              }}
              aria-invalid={ownerErr ? 'true' : undefined}
              aria-describedby={ownerErr ? ownerErrId : undefined}
              autoComplete="off"
              autoCapitalize="off"
              autoCorrect="off"
              spellCheck={false}
            />
          </label>
          {ownerErr && <p id={ownerErrId} role="alert" className="notice error">{ownerErr}</p>}
          <button type="submit" disabled={ownerBusy || !ownerText.trim()}>
            {ownerBusy ? 'Checking the key…' : 'Save owner key'}
          </button>
        </form>
      </div>

      {hasStored && (
        <div className="access-signout">
          <p className="muted">On a shared device? Sign out to remove the invite links and owner key saved here.</p>
          <SignOutOptions building={building} onSignOut={onSignOut} />
        </div>
      )}
    </section>
  )
}
