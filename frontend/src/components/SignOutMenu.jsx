import { useId, useRef, useState } from 'react'
import { LogOut } from 'lucide-react'
import { getToken } from '../access'

// The two sign-out choices. "building" forgets this building's invite; "all" forgets every invite and the owner key.
export function SignOutOptions({ building, onSignOut }) {
  const hasBuilding = !!(building && getToken(building))
  return (
    <div className="signout-options">
      {hasBuilding && (
        <button type="button" className="ghost" onClick={() => onSignOut('building')}>
          Forget this building
        </button>
      )}
      <button type="button" className="ghost danger" onClick={() => onSignOut('all')}>
        Sign out on this device
      </button>
    </div>
  )
}

// Small disclosure in the top bar. Native buttons only: Enter/Space toggles it, Escape closes it and returns focus.
export default function SignOutMenu({ building, onSignOut }) {
  const [open, setOpen] = useState(false)
  const toggle = useRef(null)
  const id = useId()

  const close = () => {
    setOpen(false)
    toggle.current?.focus()
  }

  return (
    <div className="signout" onKeyDown={(e) => e.key === 'Escape' && open && close()}>
      <button
        ref={toggle}
        type="button"
        className="ghost icon-btn signout-btn"
        title="Sign out on this device"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((o) => !o)}
      >
        <LogOut size={18} aria-hidden="true" />
        <span className="sr-only">Sign out</span>
      </button>
      <div id={id} className="signout-panel" hidden={!open}>
        <p>Removes the invite links and owner key saved on this device.</p>
        <SignOutOptions
          building={building}
          onSignOut={(scope) => {
            setOpen(false)
            onSignOut(scope)
          }}
        />
      </div>
    </div>
  )
}
