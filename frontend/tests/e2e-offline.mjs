// Offline map and photos. Run: node tests/e2e-offline.mjs  (needs `playwright` resolvable and a production build served
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

// A context with the mock API. No sign-in: every request is open.
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
    if (p === '/buildings/b1/map') { st.mapHits += 1; return json(MAP) }
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

// ---------- main flow: online once, then offline ----------
{
  const { ctx, st, errors } = await session(browser)
  const page = await ctx.newPage()
  page.on('pageerror', (e) => errors.push('PAGEERROR ' + e.message))
  page.on('dialog', (d) => d.dismiss())

  // One load with network: the shell, the map and the route (with its photos) get saved.
  await page.goto(`${BASE}/#/navigate?b=b1`)
  await page.waitForSelector('#from', { timeout: 15000 }).catch(() => {})
  check('online: the service worker takes control of the page', await waitController(page))
  await page.reload()
  await page.waitForSelector('#from', { timeout: 15000 }).catch(() => {})
  // A route is planned when the page opens on it (the address is read on mount), so reload onto the trip.
  await page.evaluate(() => { window.location.hash = '#/navigate?b=b1&from=r112&to=bh' })
  await page.reload()
  await page.waitForSelector('.scene', { timeout: 15000 }).catch(() => {})
  await page.waitForFunction(async () => (await caches.open('waypoint-data-v1').then((c) => c.keys())).length >= 3, null, { timeout: 15000 }).catch(() => {})
  const beforeOffline = await cachedUrls(page)
  check('online: the building map is saved on this device', beforeOffline.includes('/api/buildings/b1/map'), beforeOffline.join(','))
  check('online: the route photos are saved on this device', beforeOffline.includes('/bfiles/b1/frames/f1.png') && beforeOffline.includes('/bfiles/b1/frames/f2.png'), beforeOffline.join(','))
  const stored = await page.evaluate(async () => {
    const c = await caches.open('waypoint-data-v1')
    const r = await c.match('/api/buildings/b1/map')
    return r ? { auth: r.headers.get('authorization'), body: await r.text() } : null
  })
  check('storage: the saved map has no Authorization header', stored && stored.auth === null)

  // Offline, then reload: the shell and the map come from the saved copy.
  await ctx.setOffline(true)
  st.offline = true
  await page.reload()
  await page.waitForSelector('.brand', { timeout: 15000 }).catch(() => {})
  check('offline: the app shell loads', (await page.locator('.brand').count()) === 1)
  await page.waitForSelector('#from', { timeout: 15000 }).catch(() => {})
  await page.click('#from')
  const roomOffered = await page.locator('#from-list .place-option:has-text("Room 112")').count()
  check('offline: the building map loads from the saved copy', roomOffered === 1, `options=${roomOffered}`)
  check('offline: the offline banner is shown', (await page.getByRole('status').filter({ hasText: 'You are offline' }).count()) === 1)
  const photo = await page.evaluate(async (u) => (await fetch(u)).status, F1)
  check('offline: a route photo loads from the saved copy', photo === 200, `status=${photo}`)
  check('offline: no page errors', realErrors(errors).length === 0, realErrors(errors).join('|').slice(0, 200))

  // Back online: the banner goes away.
  await ctx.setOffline(false)
  st.offline = false
  await page.evaluate(() => window.dispatchEvent(new Event('online')))
  await page.waitForFunction(() => !document.body.innerText.includes('You are offline'), null, { timeout: 5000 }).catch(() => {})
  check('online again: the offline banner is removed', !(await page.innerText('body')).includes('You are offline'))

  // No sign-in: there is no sign-out control, and the saved map survives a reload while offline.
  check('no sign-out control exists', (await page.getByRole('button', { name: /sign out/i }).count()) === 0)
  await page.waitForTimeout(500)
  check('the saved map is still there', (await cachedUrls(page)).includes('/api/buildings/b1/map'))
  await ctx.setOffline(true)
  st.offline = true
  await page.reload()
  await page.waitForSelector('#from', { timeout: 15000 }).catch(() => {})
  await page.click('#from')
  check('offline after reload: the saved map still loads', (await page.locator('#from-list .place-option:has-text("Room 112")').count()) === 1)
  await ctx.close()
}

await browser.close()
const failed = results.filter((r) => !r.ok)
console.log(`\n${results.length - failed.length} passed, ${failed.length} failed of ${results.length}`)
process.exit(failed.length ? 1 : 0)
