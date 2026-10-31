import axios from 'axios'
import {
  clearOwnerKey, getCurrentBuilding, getOwnerKey, getToken, hasOwnerKey, notifyAccessNeeded, notifyOwnerKeyRemoved, tokenHeader,
} from './access'

// Relative base URL: Vite proxies in dev, FastAPI serves the built app in prod/cloudflared.
const http = axios.create({ baseURL: '/api' })

const BUILDING_PATH = /^\/buildings\/([^/?#]+)/
const buildingOf = (path) => {
  const m = BUILDING_PATH.exec(path || '')
  return m ? decodeURIComponent(m[1]) : ''
}
const hasExplicitAuth = (config) => {
  const h = config.headers
  return !!h && (typeof h.get === 'function' ? !!h.get('Authorization') : !!h.Authorization)
}

// Sends the right credential for each route. The owner key wins when stored; otherwise the building's invite token.
http.interceptors.request.use((config) => {
  // A caller that checks one specific key (pasting an invite, entering an owner key) sends only that key, never the
  // stored invite list, so a stored token cannot make a wrong owner key look valid.
  if (hasExplicitAuth(config)) {
    config.headers.delete?.('X-Access-Tokens')
    return config
  }
  const path = config.url || ''
  const method = String(config.method || 'get').toLowerCase()
  // skipOwnerKey: the list is asked again with the invites alone (see buildings below).
  const owner = config.skipOwnerKey ? '' : getOwnerKey()
  let auth = ''
  if (path === '/buildings') {
    if (method === 'get') {
      const xt = tokenHeader()
      if (xt) config.headers.set('X-Access-Tokens', xt)
    }
    auth = owner
  } else if (BUILDING_PATH.test(path)) {
    auth = owner || getToken(buildingOf(path))
  } else if (path.startsWith('/jobs/')) {
    auth = owner || getToken(getCurrentBuilding())
  }
  if (auth) config.headers.set('Authorization', `Bearer ${auth}`)
  return config
})

// A 401 means the app needs an invite (or a valid owner key). Tell the app which building, and whether a key
// was already stored, so it can explain what to do. Errors are passed on untouched.
http.interceptors.response.use(
  (r) => r,
  (err) => {
    const cfg = err?.config || {}
    if (err?.response?.status === 401 && !cfg.skipAccessNotify) {
      const building = buildingOf(cfg.url) || getCurrentBuilding()
      notifyAccessNeeded({ building, hadToken: !!getToken(building), hadOwnerKey: !!getOwnerKey() })
    }
    return Promise.reject(err)
  },
)

const enc = encodeURIComponent

// GET /buildings. The server answers 401 only when the Authorization header (a stored owner key) is refused. Then that
// key is removed from this device and the list is asked once more with the invites alone, so an invite still works.
// That retry is the last attempt: it cannot loop.
function loadBuildings() {
  const sentOwner = hasOwnerKey()
  return http
    .get('/buildings', { skipAccessNotify: sentOwner })
    .then((r) => r.data)
    .catch((err) => {
      if (err?.response?.status !== 401 || !sentOwner) throw err
      clearOwnerKey()
      notifyOwnerKeyRemoved()
      return http.get('/buildings', { skipOwnerKey: true }).then((r) => r.data)
    })
}

// An identical list request already in flight is shared, so two mount-time calls (React runs effects twice in
// development) make one request, not two.
let listInFlight = null
function buildingsList() {
  const key = `${getOwnerKey()}\n${tokenHeader()}`
  if (listInFlight && listInFlight.key === key) return listInFlight.promise
  const promise = loadBuildings().finally(() => {
    if (listInFlight?.promise === promise) listInFlight = null
  })
  listInFlight = { key, promise }
  return promise
}

export const api = {
  health: () => http.get('/health').then((r) => r.data),
  buildings: () => buildingsList(),
  createBuilding: (name) => http.post('/buildings', { name }).then((r) => r.data),
  // Checks one invite token without using anything stored. Resolves with { role } or rejects with 401.
  checkAccess: (b, token) =>
    http.get(`/buildings/${enc(b)}/access`, { headers: { Authorization: `Bearer ${token}` }, skipAccessNotify: true }).then((r) => r.data),
  // Checks one owner key without using anything stored.
  checkOwnerKey: (key) =>
    http.get('/buildings', { headers: { Authorization: `Bearer ${key}` }, skipAccessNotify: true }).then((r) => r.data),
  invites: (b) => http.get(`/buildings/${enc(b)}/invites`).then((r) => r.data),
  createInvite: (b, { role, label }) =>
    http.post(`/buildings/${enc(b)}/invites`, label ? { role, label } : { role }).then((r) => r.data),
  revokeInvite: (b, id) => http.delete(`/buildings/${enc(b)}/invites/${enc(id)}`).then((r) => r.data),
  getMap: (b) => http.get(`/buildings/${b}/map`).then((r) => r.data),
  saveMap: (b, map) => http.put(`/buildings/${b}/map`, map).then((r) => r.data.map),
  uploadVideo: (b, file, { sensors, model, priority, startFloor, onProgress } = {}) => {
    const form = new FormData()
    form.append('video', file, file.name || 'walk.mp4')
    if (sensors) form.append('sensors', new Blob([JSON.stringify(sensors)], { type: 'application/json' }), 'sensors.json')
    if (model) form.append('model', model)
    if (priority) form.append('priority', 'true')
    if (startFloor !== '' && startFloor != null) form.append('start_floor', String(startFloor))
    return http
      .post(`/buildings/${b}/videos`, form, {
        onUploadProgress: (e) => onProgress?.(e.total ? e.loaded / e.total : 0),
      })
      .then((r) => r.data)
  },
  setVideoFloor: (b, v, startFloor) => http.patch(`/buildings/${b}/videos/${v}`, { start_floor: Number(startFloor) }).then((r) => r.data),
  deleteVideo: (b, v) => http.delete(`/buildings/${b}/videos/${v}`).then((r) => r.data),
  consolidate: (b) => http.post(`/buildings/${b}/consolidate`).then((r) => r.data),
  rebuild: (b) => http.post(`/buildings/${b}/rebuild`).then((r) => r.data),
  job: (id) => http.get(`/jobs/${id}`).then((r) => r.data),
  queue: (b) => http.get(`/buildings/${b}/queue`).then((r) => r.data),
  jobAction: (id, action) => http.post(`/jobs/${id}/${action}`).then((r) => r.data),
  videoDebug: (b, v) => http.get(`/buildings/${b}/videos/${v}/debug`).then((r) => r.data),
  resolveMerge: (b, id, accept) => http.post(`/buildings/${b}/merges/${id}`, { accept }).then((r) => r.data),
  mergeNodes: (b, keep, remove) => http.post(`/buildings/${b}/merge`, { keep, remove }).then((r) => r.data),
  addPhoto: (b, file, label) => {
    const form = new FormData()
    form.append('image', file)
    form.append('label', label || '')
    return http.post(`/buildings/${b}/frames/manual`, form).then((r) => r.data)
  },
  // avoidStairs: only elevators, ramps and flat walks. Always sent, so the server never guesses. The answer carries step_free.
  route: (b, start, target, { avoidStairs = false } = {}) =>
    http
      .post(`/buildings/${b}/route`, { start_node_id: start, target_node_id: target, avoid_stairs: avoidStairs === true })
      .then((r) => r.data),
  // "Something's wrong" from Directions (any role that can read the building).
  report: (b, body) => http.post(`/buildings/${enc(b)}/reports`, body).then((r) => r.data),
  // Editors and owners. status: open | resolved | all.
  reports: (b, { status = 'open', limit = 100 } = {}) =>
    http.get(`/buildings/${enc(b)}/reports`, { params: { status, limit } }).then((r) => r.data),
  updateReport: (b, id, body) => http.patch(`/buildings/${enc(b)}/reports/${enc(id)}`, body).then((r) => r.data),
  // The caller's role; editors and owners also get open_reports (the count shown as a badge).
  myAccess: (b) => http.get(`/buildings/${enc(b)}/access`).then((r) => r.data),
}

// The message to show for a failed call. It never includes request headers, so a token cannot leak through an error.
export const detail = (err) => {
  const status = err?.response?.status
  if (status === 401) return 'You need an invite link to open this building.'
  if (status === 403) return "Your invite doesn't allow that."
  if (status === 429) return 'Too many tries, wait a minute.'
  const d = err?.response?.data?.detail
  return (typeof d === 'string' && d) || err?.message || 'Something went wrong. Try again.'
}

// Frame images. The server signs them (?exp=..&sig=..) and <img> cannot send the bearer header, so a server-provided
// URL always wins and its query string is never touched.
export const frameUrl = (b, frameId, signed) => signed || (frameId ? `/bfiles/${b}/frames/${frameId}` : null)

// Mirrors backend config.MAX_FLOOR (valid start floors are 1..MAX_FLOOR).
export const MAX_FLOOR = 9
// Returns an integer in 1..MAX_FLOOR, or null if blank/out of range.
export const validFloor = (v) => {
  const n = Number(v)
  return v !== '' && v != null && Number.isInteger(n) && n >= 1 && n <= MAX_FLOOR ? n : null
}
