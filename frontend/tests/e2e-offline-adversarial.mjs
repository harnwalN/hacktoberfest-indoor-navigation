// Offline adversarial: a 401 must not leave a readable saved map. Run: node tests/e2e-offline-adversarial.mjs  (needs `playwright` resolvable and a production build served
// at BASE, default http://localhost:5403: `npm run build`, then `npx vite preview --port 5403 --strictPort`).
// The service worker only registers in production builds. The API is mocked with context.route, so the network the
// service worker sees is the same mock. The test goes online once, then sets the context offline and checks what loads.
import { chromium } from 'playwright'

const BASE = process.env.BASE || 'http://localhost:5403'
const DESKTOP = { width: 1280, height: 800 }
const DATA_CACHE = 'waypoint-data-v1'
const F1 = '/bfiles/b1/frames/f1.png?exp=9999999999&sig=aaa111'
const F2 = '/bfiles/b1/frames/f2.png?exp=9999999999&sig=bbb222'
// 1x1 transparent PNG, the body of every photo the mock serves.
const PNG = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==', 'base64')

const N = (id, label, type, extra = {}) => ({ id, label, type, floor: 1, x: 0, y: 0, verified: true, texts: [], sign_texts: [], description: '', frame_url: null, ...extra })
const nodes = [N('r112', 'Room 112', 'room', { frame_url: F1 }), N('hp1', 'Hallway point', 'hallway', { verified: false }), N('bh', 'Room 210', 'room', { frame_url: F2 })]
const E = (a, b, len) => ({ id: `${a}-${b}`, from_node: a, to_node: b, heading_out: 90, heading_in: 90, vertical: 'none', length_m: len, frames_fwd: [], source: 'walk', verified: true })
const MAP = { building_id: 'b1', name: 'B1', nodes, edges: [E('r112', 'hp1', 12), E('hp1', 'bh', 5)], suggested_merges: [], videos: [{ id: 'v1', filename: 'a.mp4' }] }
const ROUTE = {
  node_path: ['r112', 'hp1', 'bh'],
  step_free: true,
  steps: [
    { step_index: 0, node_id: 'r112', label: 'Room 112', frame_url: F1, action: 'start', vertical: 'none', floor: 1, instruction_text: 'Start at Room 112', turn_degrees: 0 },
    { step_index: 1, node_id: 'bh', label: 'Room 210', frame_url: F2, action: 'destination_reached', vertical: 'none', floor: 1, instruction_text: 'Arrive at Room 210', turn_degrees: 0 },
  ],
}

const results = []
const check = (name, ok, info = '') => {
  results.push({ name, ok: !!ok, info })
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok ? '' : '  -> ' + info}`)
}
const realErrors = (errs) => errs.filter((e) => !/Failed to load resource/.test(e))

// A context with the mock API. No sign-in. st.map401 makes the map endpoint answer 401.
async function session(browser) {
  const ctx = await browser.newContext({ viewport: DESKTOP })
  const st = { offline: false, mapHits: 0, frameHits: 0 }
  const errors = []
  await ctx.route('**/api/**', async (route) => {
    if (st.offline) return route.abort('internetdisconnected') // context.setOffline does not stop routed requests, so the mock does
    const req = route.request()
    const p = new URL(req.url()).pathname.replace(/^\/api/, '')
    const json = (b, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(b) })
    if (p === '/buildings' && req.method() === 'GET') return json([{ id: 'b1', name: 'B1' }])
    if (p === '/buildings/b1/map') { st.mapHits += 1; return st.map401 ? json({ detail: 'Unauthorized' }, 401) : json(MAP) }
    if (p === '/buildings/b1/route' && req.method() === 'POST') return json(ROUTE)
    return json({ detail: 'nope' }, 404)
  })
  await ctx.route('**/bfiles/**', (route) => {
    if (st.offline) return route.abort('internetdisconnected')
    st.frameHits += 1
    return route.fulfill({ status: 200, contentType: 'image/png', body: PNG })
  })
  return { ctx, st, errors }
}

const cachedUrls = (page) => page.evaluate(async (name) => {
  const c = await caches.open(name)
  return (await c.keys()).map((r) => new URL(r.url).pathname)
}, DATA_CACHE)

const waitController = async (page) => {
  await page.waitForFunction(() => !!navigator.serviceWorker.controller, null, { timeout: 15000 }).catch(() => {})
  return page.evaluate(() => !!navigator.serviceWorker.controller)
}

const browser = await chromium.launch()
const OFFLINE_MAP_OPTION = '#from-list .place-option:has-text("Room 112")'

async function warm(page) {
  await page.goto(`${BASE}/#/navigate?b=b1`)
  await page.waitForSelector('#from', { timeout: 15000 }).catch(() => {})
  await waitController(page)
  await page.reload()
  await page.waitForSelector('#from', { timeout: 15000 }).catch(() => {})
  await page.waitForFunction(async () => (await (await caches.open('waypoint-data-v1')).keys()).length > 0, null, { timeout: 10000 }).catch(() => {})
}
// A fresh offline load offers the map's rooms.
async function offlineShowsMap(ctx, st, page) {
  await ctx.setOffline(true)
  st.offline = true
  await page.reload()
  await page.waitForSelector('.brand', { timeout: 15000 }).catch(() => {})
  await page.waitForTimeout(1500)
  const from = page.locator('#from')
  if (await from.count()) await from.click().catch(() => {})
  return (await page.locator(OFFLINE_MAP_OPTION).count()) > 0 || (await page.innerText('body')).includes('Room 112')
}

// (a) positive: warm, then a fresh offline load shows the saved map
{
  const { ctx, st } = await session(browser)
  const page = await ctx.newPage()
  await warm(page)
  check('a: map saved while online', (await cachedUrls(page)).includes('/api/buildings/b1/map'))
  check('a: fresh offline load shows the saved map', await offlineShowsMap(ctx, st, page))
  await ctx.close()
}

// (b) 401 while online, then offline
{
  const { ctx, st } = await session(browser)
  const page = await ctx.newPage()
  await warm(page)
  check('b: map saved before the 401', (await cachedUrls(page)).includes('/api/buildings/b1/map'))
  // The map endpoint now answers 401: it must not be cached or shown offline.
  st.map401 = true
  await page.reload()
  await page.waitForTimeout(1500)
  const left = await cachedUrls(page)
  check('b: a 401 removes the saved map', !left.includes('/api/buildings/b1/map'), left.join(','))
  check('b: a 401 removes the saved photos', !left.some((p) => p.startsWith('/bfiles/b1/')), left.join(','))
  check('b: next offline load does not show the map', !(await offlineShowsMap(ctx, st, page)))
  await ctx.close()
}

await browser.close()
const failed = results.filter((r) => !r.ok)
console.log(`\n${results.length - failed.length} passed, ${failed.length} failed of ${results.length}`)
process.exit(failed.length ? 1 : 0)
