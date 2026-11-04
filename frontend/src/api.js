import axios from 'axios'
import { clearSession, getSessionToken } from './session'

// Relative base URL: Vite proxies in dev, FastAPI serves the built app in prod/cloudflared.
const http = axios.create({ baseURL: '/api' })

// A write that needs a contributor identity. The stored session (if any) goes as a bearer header and the config is
// marked, so a 401 on it means the session is no longer valid: it is cleared and "auth:expired" tells the app.
const authed = (cfg = {}) => {
  const token = getSessionToken()
  if (!token) return { ...cfg, needsSession: true }
  return { ...cfg, needsSession: true, headers: { ...(cfg.headers || {}), Authorization: `Bearer ${token}` } }
}
http.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err?.response?.status === 401 && err.config?.needsSession) {
      clearSession()
      window.dispatchEvent(new Event('auth:expired'))
    }
    return Promise.reject(err)
  },
)

// An identical building list request already in flight is shared, so two mount-time calls (React runs effects twice
// in development) make one request, not two.
// The list carries the bearer when a session is stored: the server shows an owner every building (also empty ones) and
// visitors only buildings with a map. A 401 here does not clear the session (the read is not a write).
let listInFlight = null
let listToken = null
function buildingsList() {
  const token = getSessionToken()
  if (!listInFlight || listToken !== token) {
    listToken = token
    const req = http
      .get('/buildings', token ? { headers: { Authorization: `Bearer ${token}` } } : undefined)
      .then((r) => r.data)
      .finally(() => {
        if (listInFlight === req) listInFlight = null
      })
    listInFlight = req
  }
  return listInFlight
}

// The one place the account shape is decided. /api/auth/me answers {account:{id,email,moderator_of}, is_owner, blocked};
// the flat {id,email,is_owner} shape is still accepted. Returns the flat object that is stored and used everywhere:
// {...account, is_owner, blocked}. Null for a missing answer.
export function normaliseAccount(data) {
  if (!data || typeof data !== 'object') return null
  const nested = data.account && typeof data.account === 'object' ? data.account : null
  const base = nested || data
  return { ...base, is_owner: !!data.is_owner, blocked: !!data.blocked }
}

const enc = encodeURIComponent

export const api = {
  health: () => http.get('/health').then((r) => r.data),
  buildings: () => buildingsList(),
  createBuilding: (name) => http.post('/buildings', { name }, authed()).then((r) => r.data),
  // Sign-in by email link. requestLink always answers the same way (no account enumeration); verify exchanges the
  // one-time link token for a session; me checks a stored session (401 when it is no longer valid).
  requestLink: (email) => http.post('/auth/request-link', { email }).then((r) => r.data),
  verify: (token) =>
    http.post('/auth/verify', { token }).then((r) => ({ ...r.data, account: normaliseAccount(r.data?.account) })),
  me: (token) => http.get('/auth/me', { headers: { Authorization: `Bearer ${token}` } }).then((r) => normaliseAccount(r.data)),
  getMap: (b) => http.get(`/buildings/${b}/map`).then((r) => r.data),
  saveMap: (b, map) => http.put(`/buildings/${b}/map`, map, authed()).then((r) => r.data.map),
  uploadVideo: (b, file, { sensors, model, priority, startFloor, onProgress } = {}) => {
    const form = new FormData()
    form.append('video', file, file.name || 'walk.mp4')
    if (sensors) form.append('sensors', new Blob([JSON.stringify(sensors)], { type: 'application/json' }), 'sensors.json')
    if (model) form.append('model', model)
    if (priority) form.append('priority', 'true')
    if (startFloor !== '' && startFloor != null) form.append('start_floor', String(startFloor))
    return http
      .post(`/buildings/${b}/videos`, form, authed({
        onUploadProgress: (e) => onProgress?.(e.total ? e.loaded / e.total : 0),
      }))
      .then((r) => r.data)
  },
  setVideoFloor: (b, v, startFloor) => http.patch(`/buildings/${b}/videos/${v}`, { start_floor: Number(startFloor) }, authed()).then((r) => r.data),
  deleteVideo: (b, v) => http.delete(`/buildings/${b}/videos/${v}`, authed()).then((r) => r.data),
  consolidate: (b) => http.post(`/buildings/${b}/consolidate`, undefined, authed()).then((r) => r.data),
  rebuild: (b) => http.post(`/buildings/${b}/rebuild`, undefined, authed()).then((r) => r.data),
  job: (id) => http.get(`/jobs/${id}`).then((r) => r.data),
  queue: (b) => http.get(`/buildings/${b}/queue`).then((r) => r.data),
  jobAction: (id, action) => http.post(`/jobs/${id}/${action}`, undefined, authed()).then((r) => r.data),
  videoDebug: (b, v) => http.get(`/buildings/${b}/videos/${v}/debug`).then((r) => r.data),
  resolveMerge: (b, id, accept) => http.post(`/buildings/${b}/merges/${id}`, { accept }, authed()).then((r) => r.data),
  mergeNodes: (b, keep, remove) => http.post(`/buildings/${b}/merge`, { keep, remove }, authed()).then((r) => r.data),
  addPhoto: (b, file, label) => {
    const form = new FormData()
    form.append('image', file)
    form.append('label', label || '')
    return http.post(`/buildings/${b}/frames/manual`, form, authed()).then((r) => r.data)
  },
  // avoidStairs: only elevators, ramps and flat walks. Always sent, so the server never guesses. The answer carries step_free.
  route: (b, start, target, { avoidStairs = false } = {}) =>
    http
      .post(`/buildings/${b}/route`, { start_node_id: start, target_node_id: target, avoid_stairs: avoidStairs === true })
      .then((r) => r.data),
  // "Something's wrong" from Directions.
  report: (b, body) => http.post(`/buildings/${enc(b)}/reports`, body).then((r) => r.data),
  // status: open | resolved | all.
  reports: (b, { status = 'open', limit = 100 } = {}) =>
    http.get(`/buildings/${enc(b)}/reports`, { params: { status, limit } }).then((r) => r.data),
  updateReport: (b, id, body) => http.patch(`/buildings/${enc(b)}/reports/${enc(id)}`, body).then((r) => r.data),
  // Contributor: own uploads with their status (processing | live | needs_review | rejected).
  myUploads: () => http.get('/me/uploads', authed()).then((r) => r.data),
  // Owner tools. status: pending | rejected | approved | hidden | all.
  moderation: (status) => http.get('/owner/moderation', authed({ params: { status } })).then((r) => r.data),
  moderate: (b, v, action, note) =>
    http.post(`/owner/videos/${enc(b)}/${enc(v)}/moderation`, { action, ...(note ? { note } : {}) }, authed()).then((r) => r.data),
  ownerUsers: () => http.get('/owner/users', authed()).then((r) => r.data),
  setBlocked: (email, blocked) => http.post(`/owner/users/${enc(email)}/block`, { blocked }, authed()).then((r) => r.data),
  deleteBuilding: (b) => http.delete(`/buildings/${enc(b)}`, authed()).then((r) => r.data),
}

// The message to show for a failed call.
export const detail = (err) => {
  const status = err?.response?.status
  if (status === 429) return 'Too many tries, wait a minute.'
  const d = err?.response?.data?.detail
  return (typeof d === 'string' && d) || err?.message || 'Something went wrong. Try again.'
}

// The message for a failed manual photo add. 422: moderation rejected the photo, its reason is in `detail`.
// 503: the moderation check is unavailable right now.
export const photoError = (err) => {
  const status = err?.response?.status
  if (status === 422) {
    const d = err.response.data?.detail
    return (typeof d === 'string' && d) || 'The photo was rejected and not added.'
  }
  if (status === 503) return 'The photo check is unavailable right now. Try again later.'
  return detail(err)
}

// Frame images. A server-provided URL always wins; otherwise the path is built from the building and frame ids.
export const frameUrl = (b, frameId, signed) => signed || (frameId ? `/bfiles/${b}/frames/${frameId}` : null)

// Mirrors backend config.MAX_FLOOR (valid start floors are 1..MAX_FLOOR).
export const MAX_FLOOR = 9
// Returns an integer in 1..MAX_FLOOR, or null if blank/out of range.
export const validFloor = (v) => {
  const n = Number(v)
  return v !== '' && v != null && Number.isInteger(n) && n >= 1 && n <= MAX_FLOOR ? n : null
}
