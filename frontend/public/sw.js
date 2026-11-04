// Waypoint offline cache: a plain service worker (no library). Only production builds register it (src/offline.js).
//
// Shell: /assets/* (hashed, immutable), the manifest and the icon are cache-first. The lazy pages' chunks are precached at install. The document (/, /index.html,
// navigations) is answered from cache at once and refreshed in the background, so a new deploy shows up on the next visit.
//
// The building list (/api/buildings) and building map JSON (/api/buildings/<b>/map) and keyframe photos (/bfiles/<b>/frames/<file>) are network-first.
// A successful answer is stored; the cached copy is used only when the network cannot answer (a network error or a 5xx).
// A 4xx is never answered from cache. A 401/403 on the map, or a 401/403/404 on a photo, removes that building's cached
// data (the server refused it). Each fresh map also drops cached photos it no longer lists.
//
// Never stored: anything whose URL has a query parameter, except the old photo signature (exp, sig) that is still
// accepted. Only the response body is stored.

const SHELL = 'waypoint-shell-v1'
const DATA = 'waypoint-data-v1'
const KEEP = [SHELL, DATA]
const LIST_PATH = '/api/buildings'
const MAP_PATH = /^\/api\/buildings\/([^/]+)\/map$/
const FRAME_PATH = /^\/bfiles\/([^/]+)\/frames\/([^/]+)$/
const MAX_PREFETCH = 200

// True when every query parameter is one the cache is allowed to hold (none for the map, exp and sig for photos).
function queryAllowed(url, allowed) {
  for (const k of url.searchParams.keys()) if (!allowed.includes(k)) return false
  return true
}

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

// Removes the cached map and photos of one building (raw path segment) where test(pathname) is true.
async function dropEntries(cache, test) {
  for (const req of await cache.keys()) {
    if (test(new URL(req.url).pathname)) await cache.delete(req)
  }
}
const isBuilding = (b) => (p) => p === `/api/buildings/${b}/map` || p.startsWith(`/bfiles/${b}/frames/`)

// Called with a fresh map body: photos of this building that the map no longer lists are dropped.
async function pruneFrames(cache, b, text) {
  const live = new Set()
  const re = new RegExp(`/bfiles/${escapeRe(b)}/frames/[^"\\s\\\\]+`, 'g')
  for (const m of text.matchAll(re)) live.add(new URL(m[0], self.location.origin).href)
  for (const req of await cache.keys()) {
    const u = new URL(req.url)
    if (u.pathname.startsWith(`/bfiles/${b}/frames/`) && !live.has(u.href)) await cache.delete(req)
  }
}

const jsonError = (status, detail) =>
  new Response(JSON.stringify({ detail }), { status, headers: { 'Content-Type': 'application/json' } })

// The building list (GET /api/buildings) and a building map. Both answer from the network first; the saved copy
// answers only when the network cannot (no connection, or a 5xx). A refused request (401/403) drops what it saved.
async function jsonFromNetwork(req, url, { building = '', missing }) {
  const cache = await caches.open(DATA)
  const key = url.href
  let res
  try {
    res = await fetch(req)
  } catch {
    const hit = await cache.match(key)
    return hit || jsonError(503, missing)
  }
  if (res.status === 401 || res.status === 403) {
    if (building) await dropEntries(cache, isBuilding(building)) // access refused: nothing for this building stays
    else await cache.delete(key)
    return res
  }
  if (res.ok) {
    const text = await res.clone().text()
    await cache.put(key, new Response(text, { headers: { 'Content-Type': 'application/json' } }))
    if (building) await pruneFrames(cache, building, text)
    return res
  }
  if (res.status >= 500) {
    const hit = await cache.match(key)
    if (hit) return hit
    return res
  }
  await cache.delete(key) // 404 and other 4xx: gone, so never served from cache
  return res
}

const mapFromNetwork = (req, url, b) =>
  jsonFromNetwork(req, url, { building: b, missing: "This building's map is not saved on this device yet. Connect once to save it." })
const listFromNetwork = (req, url) =>
  jsonFromNetwork(req, url, { missing: 'Your building list is not saved on this device yet. Connect once to save it.' })

async function frameFromNetwork(req, url, b) {
  const cache = await caches.open(DATA)
  let res
  try {
    res = await fetch(req)
  } catch {
    const hit = await cache.match(url.href)
    return hit || Response.error()
  }
  if (res.ok) {
    await cache.put(url.href, res.clone())
  } else if (res.status === 401 || res.status === 403) {
    await dropEntries(cache, isBuilding(b)) // refused: drop the building, not just this photo
  } else if (res.status === 404 || res.status < 500) {
    await cache.delete(url.href) // missing or refused: never served from cache again
  } else {
    const hit = await cache.match(url.href)
    if (hit) return hit
  }
  return res
}

async function shellAsset(req, url) {
  const cache = await caches.open(SHELL)
  const hit = await cache.match(req)
  if (hit) return hit
  const res = await fetch(req)
  if (res.ok) await cache.put(req, res.clone())
  return res
}

async function shellDocument(url) {
  const cache = await caches.open(SHELL)
  const key = url.origin + url.pathname
  const fresh = fetch(new Request(key)).then(async (res) => {
    if (res.ok) await cache.put(key, res.clone())
    return res
  })
  const hit = await cache.match(key)
  if (hit) {
    fresh.catch(() => {}) // refresh in the background; the cached copy answers now
    return hit
  }
  try {
    return await fresh
  } catch {
    return Response.error()
  }
}

// Install: precache the lazy pages' chunks (the list the build writes to /lazy-chunks.json). A missing list or a failed
// chunk never blocks the install; that chunk is then cached on its first load, as before.
const LAZY_LIST = '/lazy-chunks.json'
self.addEventListener('install', (event) => {
  event.waitUntil(
    (async () => {
      try {
        const list = await (await fetch(LAZY_LIST, { cache: 'no-store' })).json()
        const cache = await caches.open(SHELL)
        await Promise.all(
          list.map(async (url) => {
            try {
              const res = await fetch(url)
              if (res.ok) await cache.put(url, res)
            } catch {
              /* offline during install: cached on first use */
            }
          }),
        )
      } catch {
        /* no list (dev or an old build) */
      }
    })(),
  )
  self.skipWaiting()
})

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      for (const k of await caches.keys()) if (!KEEP.includes(k)) await caches.delete(k)
      await self.clients.claim()
    })(),
  )
})

self.addEventListener('fetch', (event) => {
  const req = event.request
  if (req.method !== 'GET') return
  const url = new URL(req.url)
  if (url.origin !== self.location.origin) return

  if (url.pathname === LIST_PATH && queryAllowed(url, [])) {
    event.respondWith(listFromNetwork(req, url))
    return
  }
  const map = MAP_PATH.exec(url.pathname)
  if (map && queryAllowed(url, [])) {
    event.respondWith(mapFromNetwork(req, url, map[1]))
    return
  }
  const frame = FRAME_PATH.exec(url.pathname)
  if (frame && queryAllowed(url, ['exp', 'sig'])) {
    event.respondWith(frameFromNetwork(req, url, frame[1]))
    return
  }
  if (url.pathname.startsWith('/assets/') || url.pathname === '/manifest.webmanifest' || url.pathname === '/icon.svg') {
    event.respondWith(shellAsset(req, url).catch(() => Response.error()))
    return
  }
  if (req.mode === 'navigate' || url.pathname === '/' || url.pathname === '/index.html') {
    event.respondWith(shellDocument(url))
  }
})

// The page asks for the photos of a route it just showed, so they can be shown with no signal.
self.addEventListener('message', (event) => {
  const d = event.data || {}
  if (d.type !== 'cache-frames' || !Array.isArray(d.urls)) return
  event.waitUntil(cacheFrames(d.urls.slice(0, MAX_PREFETCH)))
})

async function cacheFrames(list) {
  const cache = await caches.open(DATA)
  for (const raw of list) {
    if (typeof raw !== 'string') continue
    let url
    try {
      url = new URL(raw, self.location.origin)
    } catch {
      continue
    }
    if (url.origin !== self.location.origin || !FRAME_PATH.test(url.pathname) || !queryAllowed(url, ['exp', 'sig'])) continue
    try {
      const res = await fetch(url.href)
      if (res.ok) await cache.put(url.href, res)
    } catch {
      /* offline: the photo is simply not kept */
    }
  }
}
