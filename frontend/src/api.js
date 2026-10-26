import axios from 'axios'

// Relative base URL: Vite proxies in dev, FastAPI serves the built app in prod/cloudflared.
const http = axios.create({ baseURL: '/api' })

const detail = (err) => err?.response?.data?.detail || err.message

export const api = {
  health: () => http.get('/health').then((r) => r.data),
  buildings: () => http.get('/buildings').then((r) => r.data),
  createBuilding: (name) => http.post('/buildings', { name }).then((r) => r.data),
  getMap: (b) => http.get(`/buildings/${b}/map`).then((r) => r.data),
  saveMap: (b, map) => http.put(`/buildings/${b}/map`, map).then((r) => r.data.map),
  uploadVideo: (b, file, { sensors, model, priority, onProgress } = {}) => {
    const form = new FormData()
    form.append('video', file, file.name || 'walk.mp4')
    if (sensors) form.append('sensors', new Blob([JSON.stringify(sensors)], { type: 'application/json' }), 'sensors.json')
    if (model) form.append('model', model)
    if (priority) form.append('priority', 'true')
    return http
      .post(`/buildings/${b}/videos`, form, {
        onUploadProgress: (e) => onProgress?.(e.total ? e.loaded / e.total : 0),
      })
      .then((r) => r.data)
  },
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
  route: (b, start, target) =>
    http.post(`/buildings/${b}/route`, { start_node_id: start, target_node_id: target }).then((r) => r.data),
}

export { detail }

export const frameUrl = (b, frameId) => (frameId ? `/bfiles/${b}/frames/${frameId}` : null)
