// Invite access on this device: per-building invite tokens and the owner key, kept in localStorage.
// Every storage call is guarded (private mode can throw). Nothing here logs or shows a secret.

const TOKENS_KEY = 'waypoint.tokens'
const OWNER_KEY = 'waypoint.ownerKey'
const MAX_TOKENS = 50
// Building ids come from a slug (letters, digits, dashes). Tokens must not contain whitespace or commas, since
// they travel comma-separated in X-Access-Tokens.
const SAFE_ID = /^[^\s/?#&=,]{1,128}$/
const SAFE_TOKEN = /^[^\s,]{1,512}$/

// Used only while localStorage is unavailable, so this page load still works.
const memory = { tokens: {}, ownerKey: '' }
let storageBroken = false

function storage() {
  try {
    return window.localStorage
  } catch {
    return null
  }
}

function cleanTokens(obj) {
  const out = {}
  if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return out
  for (const [id, t] of Object.entries(obj)) {
    if (SAFE_ID.test(id) && typeof t === 'string' && SAFE_TOKEN.test(t)) out[id] = t
  }
  return out
}

function readTokens() {
  const s = storage()
  if (s && !storageBroken) {
    try {
      return cleanTokens(JSON.parse(s.getItem(TOKENS_KEY) || '{}'))
    } catch {
      /* unreadable: fall back to what this page holds */
    }
  }
  return { ...memory.tokens }
}

function writeTokens(tokens) {
  memory.tokens = tokens
  try {
    storage()?.setItem(TOKENS_KEY, JSON.stringify(tokens))
  } catch {
    storageBroken = true
  }
}

export function getToken(building) {
  return readTokens()[building] || ''
}

export function setToken(building, token) {
  if (!SAFE_ID.test(building) || !SAFE_TOKEN.test(token)) return
  writeTokens({ ...readTokens(), [building]: token })
}

export function clearToken(building) {
  const next = readTokens()
  delete next[building]
  writeTokens(next)
}

// Every stored token, comma-separated, for GET /buildings (max 50).
export function tokenHeader() {
  return Object.values(readTokens()).slice(0, MAX_TOKENS).join(',')
}

export function getOwnerKey() {
  const s = storage()
  if (s && !storageBroken) {
    try {
      const v = s.getItem(OWNER_KEY)
      return v && SAFE_TOKEN.test(v) ? v : ''
    } catch {
      /* fall back to memory */
    }
  }
  return memory.ownerKey
}

export function setOwnerKey(key) {
  if (!SAFE_TOKEN.test(key)) return
  memory.ownerKey = key
  try {
    storage()?.setItem(OWNER_KEY, key)
  } catch {
    storageBroken = true
  }
}

export function hasOwnerKey() {
  return !!getOwnerKey()
}

// Removes only the owner key (the invites stay). Used when the server refuses it on the building list.
export function clearOwnerKey() {
  memory.ownerKey = ''
  try {
    storage()?.removeItem(OWNER_KEY)
  } catch {
    storageBroken = true
  }
}

// Tells the app that a stored owner key was removed for us, so it can say so once.
const ownerKeyListeners = new Set()
export function onOwnerKeyRemoved(fn) {
  ownerKeyListeners.add(fn)
  return () => ownerKeyListeners.delete(fn)
}
export function notifyOwnerKeyRemoved() {
  ownerKeyListeners.forEach((fn) => fn())
}

export function hasStoredAccess() {
  return Object.keys(readTokens()).length > 0 || hasOwnerKey()
}

// "Sign out on this device": removes every token and the owner key.
export function clearAllAccess() {
  writeTokens({})
  memory.ownerKey = ''
  try {
    storage()?.removeItem(OWNER_KEY)
  } catch {
    storageBroken = true
  }
}

// The building the app is showing. Jobs have no building in their URL, so they use this one.
let currentBuilding = ''
export const setCurrentBuilding = (id) => {
  currentBuilding = id || ''
}
export const getCurrentBuilding = () => currentBuilding

// The API tells the app when a request was refused for lack of access (401), so it can show the access screen.
const accessListeners = new Set()
export function onAccessNeeded(fn) {
  accessListeners.add(fn)
  return () => accessListeners.delete(fn)
}
export function notifyAccessNeeded(info) {
  accessListeners.forEach((fn) => fn(info))
}

// Pulls { building, token } out of an invite link. Accepts the full URL, a #/join?… fragment, or just join?…
// Returns null for anything else. The pasted text is never echoed back, since it may contain the token.
export function parseInviteLink(text) {
  const raw = String(text || '').trim()
  const at = raw.search(/join\?/)
  if (at < 0) return null
  const params = new URLSearchParams(raw.slice(at + 5).split(/\s/)[0])
  const building = (params.get('b') || '').trim()
  const token = (params.get('t') || '').trim()
  if (!SAFE_ID.test(building) || !SAFE_TOKEN.test(token)) return null
  return { building, token }
}

// A join link (#/join?b=<id>&t=<token>) stores its token, then the token is removed from the address bar before
// anything else happens, so it does not stay in history. Returns null when the hash is not a join link.
export function consumeJoinLink() {
  const hash = window.location.hash || ''
  if (!/^#\/?join(\?|$)/.test(hash)) return null
  const q = hash.indexOf('?')
  const params = new URLSearchParams(q >= 0 ? hash.slice(q + 1) : '')
  const building = (params.get('b') || '').trim()
  const token = (params.get('t') || '').trim()
  const idOk = SAFE_ID.test(building)
  const to = idOk ? `#/navigate?b=${encodeURIComponent(building)}` : '#/navigate'
  window.history.replaceState(window.history.state, '', `${window.location.pathname}${window.location.search}${to}`)
  if (!idOk || !SAFE_TOKEN.test(token)) return { ok: false, building: idOk ? building : '' }
  setToken(building, token)
  return { ok: true, building }
}
