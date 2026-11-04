import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { api } from '../api'
import { clearSession, getSessionAccount, getSessionToken, saveSession, takeTokenFromUrl } from '../session'
import SignInDialog from './SignInDialog'

// A sign-in link token is read once, when the module loads, before StrictMode can render twice (the URL is cleaned
// right away, so a second read would find nothing).
const LINK_TOKEN = takeTokenFromUrl()
let linkVerify = null
// One verify call per page load, shared by StrictMode's double effect. A link token works once.
const verifyLinkOnce = () => {
  if (!linkVerify) linkVerify = api.verify(LINK_TOKEN)
  return linkVerify
}

const AuthContext = createContext(null)

// Identity only: browsing never needs this. Write flows call requireSignIn() before they send anything.
export function AuthProvider({ children }) {
  const [account, setAccount] = useState(() => (getSessionToken() ? getSessionAccount() : null))
  const [verifying, setVerifying] = useState(!!LINK_TOKEN)
  // True until the stored session (or a new sign-in link) has been checked against /api/auth/me, so is_owner is known.
  const [checking, setChecking] = useState(() => !!LINK_TOKEN || !!getSessionToken())
  const [dialog, setDialog] = useState(null) // null, or { reason } when the sign-in sheet is open

  // Opened by a sign-in link: exchange the token for a session, or say the link did not work.
  useEffect(() => {
    if (!LINK_TOKEN) return
    verifyLinkOnce()
      .then(({ session_token, account: acct }) => {
        saveSession(session_token, acct)
        setAccount(acct)
        // /api/auth/me carries is_owner; the verify answer may not, so ask once more before the owner page decides.
        return api
          .me(session_token)
          .then((full) => {
            saveSession(session_token, full)
            setAccount(full)
          })
          .catch(() => {})
      })
      .catch(() => setDialog({ reason: 'That sign-in link is invalid or has expired. Ask for a new one below.' }))
      .finally(() => {
        setVerifying(false)
        setChecking(false)
      })
  }, [])

  // A stored session is checked once. A 401 means it is gone; a network error keeps it for the next try.
  useEffect(() => {
    const token = getSessionToken()
    if (!token || LINK_TOKEN) return
    api
      .me(token)
      .then((acct) => {
        saveSession(token, acct)
        setAccount(acct)
      })
      .catch((e) => {
        if (e?.response?.status === 401) {
          clearSession()
          setAccount(null)
        }
      })
      .finally(() => setChecking(false))
  }, [])

  // Any write sent with the session that the server refuses: the session is cleared and the sheet asks again.
  useEffect(() => {
    const onExpired = () => {
      setAccount(null)
      setDialog({ reason: 'Your sign-in ended. Sign in again to continue.' })
    }
    window.addEventListener('auth:expired', onExpired)
    return () => window.removeEventListener('auth:expired', onExpired)
  }, [])

  const openSignIn = (reason = '') => setDialog({ reason })
  const closeSignIn = useCallback(() => setDialog(null), [])

  // Returns true when the caller may go on. Otherwise it opens the sheet with why, and the caller stops.
  const requireSignIn = (reason) => {
    if (account) return true
    setDialog({ reason })
    return false
  }

  const signOut = () => {
    clearSession()
    setAccount(null)
  }

  // is_owner comes from /api/auth/me. A stored account from before the owner split has none, so it is false until checked.
  const isOwner = !!account?.is_owner
  const value = { account, signedIn: !!account, isOwner, checking, verifying, openSignIn, requireSignIn, signOut }

  return (
    <AuthContext.Provider value={value}>
      {children}
      <SignInDialog open={!!dialog} reason={dialog?.reason || ''} onClose={closeSignIn} />
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth needs AuthProvider')
  return ctx
}
