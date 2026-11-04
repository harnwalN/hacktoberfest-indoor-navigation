// The contributor session, kept in localStorage so it survives a reload.
// Keys: "session_token" (the bearer string) and "session_account" (JSON {id, email}).
// Sent as "Authorization: Bearer <session_token>" on writes that need an identity only (walk upload, building
// creation). Reads (maps, buildings, routes, reports) never send it.
const TOKEN_KEY = 'session_token'
const ACCOUNT_KEY = 'session_account'

export function getSessionToken() {
  try {
    return localStorage.getItem(TOKEN_KEY) || null
  } catch {
    return null
  }
}

export function getSessionAccount() {
  try {
    const v = JSON.parse(localStorage.getItem(ACCOUNT_KEY) || 'null')
    return v && typeof v.email === 'string' ? v : null
  } catch {
    return null
  }
}

export function saveSession(token, account) {
  try {
    localStorage.setItem(TOKEN_KEY, token)
    localStorage.setItem(ACCOUNT_KEY, JSON.stringify(account))
  } catch {
    /* private mode: the session lasts this page load only */
  }
}

export function clearSession() {
  try {
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(ACCOUNT_KEY)
  } catch {
    /* nothing stored */
  }
}

// Reads ?token= from the query, or from the hash query (#/navigate?token=), and removes it from the address bar at
// once, so the one-time token is not left in history or on screen. Returns the token or null.
export function takeTokenFromUrl() {
  if (typeof window === 'undefined') return null
  const search = new URLSearchParams(window.location.search)
  const hashQuery = window.location.hash.includes('?') ? window.location.hash.split('?').slice(1).join('?') : ''
  const hashParams = new URLSearchParams(hashQuery)
  const token = search.get('token') || hashParams.get('token')
  if (!token) return null
  search.delete('token')
  hashParams.delete('token')
  const qs = search.toString()
  const hs = hashParams.toString()
  const hashPath = window.location.hash.includes('?') ? window.location.hash.split('?')[0] : window.location.hash
  const hash = hs ? `${hashPath}?${hs}` : hashPath
  window.history.replaceState(null, '', `${window.location.pathname}${qs ? `?${qs}` : ''}${hash}`)
  return token
}
