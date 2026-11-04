// Session on the map-edit and walk-management writes. Run: node tests/e2e-write-auth.mjs (needs `playwright` resolvable
// and a vite dev server at BASE, default http://localhost:5199). The API is mocked with page.route.
// Covers: without a session, each write (floor, rebuild, delete walk, save map, link videos, add photo) opens the
// sign-in sheet with its reason and sends nothing; with a session each write carries Authorization: Bearer <session>
// and GETs still do not; a manual photo refused with 422 shows the moderation reason, 503 shows "try again later";
// a write refused with 401 opens "Your sign-in ended" and clears the session.
import { chromium } from 'playwright'

const BASE = process.env.BASE || 'http://localhost:5199'
const SESSION = 'sess-123'
const TOKEN_KEY = 'session_token'
const ACCOUNT_KEY = 'session_account'
const ENDED = 'Your sign-in ended'
const REJECT_TEXT = 'Blocked: photo shows a face without consent.'
const UNAVAILABLE = 'The photo check is unavailable right now. Try again later.'

const MAP = {
  building_id: 'b1',
  name: 'Hall B',
  nodes: [{ id: 'n1', label: 'Lobby', floor: 1, type: 'hallway', verified: false, texts: [], sign_texts: [] }],
  edges: [],
  suggested_merges: [],
  videos: [{ id: 'v1', filename: 'walk.mp4', duration: 12, keyframes: 4, status: 'done', start_floor: null, floor_offset: 0, has_sensors: false }],
}

const results = []
const check = (name, ok, info = '') => {
  results.push({ name, ok: !!ok, info })
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok ? '' : '  -> ' + info}`)
}
const realErrors = (errs) => errs.filter((e) => !/Failed to load resource/.test(e))
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

const browser = await chromium.launch()

// One browser context per scenario. opts: { stored, status: { patch, delete, rebuild, ... } }.
async function session(opts = {}) {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 } })
  const page = await ctx.newPage()
  const errors = []
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
  page.on('pageerror', (e) => errors.push('PAGEERROR ' + e.message))
  page.on('dialog', (d) => d.accept())
  const st = {
    requests: [], // { method, path, auth, body }
    status: {}, // per-route status override: patch | delete | photo
    photo: { status: 201, body: null },
    ...opts,
  }
  await ctx.route('**/api/**', async (route) => {
    const req = route.request()
    const url = new URL(req.url())
    const p = url.pathname.replace(/^\/api/, '')
    const m = req.method()
    const auth = req.headers()['authorization'] || null
    let body = null
    try {
      body = req.postDataJSON()
    } catch {
      body = null
    }
    st.requests.push({ method: m, path: p, auth, body })
    const json = (b, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(b) })

    if (p === '/health') return json({ gemma: { models: [] } })
    if (p === '/buildings' && m === 'GET') return json([{ id: 'b1', name: 'Hall B' }])
    if (p === '/buildings/b1/map' && m === 'GET') return json(MAP)
    if (p === '/buildings/b1/map' && m === 'PUT') return json({ map: MAP })
    if (p === '/buildings/b1/queue') return json([])
    if (p === '/buildings/b1/reports') return json([])
    if (p === '/buildings/b1/videos/v1' && m === 'PATCH') return st.status.patch ? json({ detail: 'sign in' }, st.status.patch) : json({ ok: true })
    if (p === '/buildings/b1/videos/v1' && m === 'DELETE') return json({ removed_places: 0, deleted_files: 1 })
    if (p === '/buildings/b1/rebuild') return json({ job_id: 'j1' }, 202)
    if (p === '/buildings/b1/consolidate') return json({ job_id: 'j1' }, 202)
    if (p === '/buildings/b1/merge' || p.startsWith('/buildings/b1/merges/')) return json(MAP)
    if (p === '/buildings/b1/frames/manual') {
      if (st.photo.status !== 201) return json(st.photo.body, st.photo.status)
      return json({ node: { id: 'n9' } }, 201)
    }
    if (p.startsWith('/jobs/')) return json({ id: 'j1', status: 'done', stages: [], log: [] })
    return route.fulfill({ status: 404, body: '{}' })
  })
  if (opts.stored) {
    await page.addInitScript(([k, v, a]) => {
      if (!sessionStorage.getItem('seeded')) {
        sessionStorage.setItem('seeded', '1')
        localStorage.setItem(k, v)
        localStorage.setItem(a, JSON.stringify({ id: 'acc1', email: 'ada@example.com', is_owner: true }))
      }
    }, [TOKEN_KEY, opts.stored, ACCOUNT_KEY])
  }
  return { ctx, page, errors, st }
}

const dialog = (page) => page.getByRole('dialog', { name: 'Sign in' })
const storedToken = (page) => page.evaluate((k) => localStorage.getItem(k), TOKEN_KEY)
const writes = (st) => st.requests.filter((r) => r.method !== 'GET' && !r.path.startsWith('/auth/'))
const closeSheet = async (page) => {
  await page.keyboard.press('Escape')
  await dialog(page).waitFor({ state: 'detached', timeout: 3000 }).catch(() => {})
}
// Owner tools live on the owner page now (Videos & map). The Map tab is view-only for everyone.
const gotoBuild = async (page) => {
  await page.goto(`${BASE}/#/owner?s=videos&b=b1`)
  await page.locator('.floor-input input').first().waitFor({ timeout: 5000 })
}
const gotoMap = async (page) => {
  await page.goto(`${BASE}/#/owner?s=videos&b=b1`)
  await page.locator('.map-actions').waitFor({ timeout: 5000 })
}
const editLabel = async (page, text) => {
  await page.locator('.node-card').first().click()
  await page.getByLabel('Label', { exact: true }).fill(text)
}
const photoFile = { name: 'lobby.jpg', mimeType: 'image/jpeg', buffer: Buffer.from('x') }

// 1. No session: the owner page is not allowed, so no owner tool (floor, rebuild, delete) is reachable and nothing is sent.
{
  const { ctx, page, errors, st } = await session()
  await page.goto(`${BASE}/#/owner?s=videos&b=b1`)
  await page.getByText('This page is not allowed').waitFor({ timeout: 5000 })
  check('owner gate: signed out, the owner page says not allowed', (await page.getByText('This page is not allowed').count()) === 1)
  check('owner gate: signed out, no floor, rebuild or delete controls', (await page.locator('.floor-input input').count()) === 0
    && (await page.getByRole('button', { name: 'Rebuild map' }).count()) === 0 && (await page.locator('button[title="Remove this video and its data"]').count()) === 0)
  await sleep(200)
  check('owner gate: no PATCH, rebuild or DELETE sent without a session', writes(st).length === 0, JSON.stringify(writes(st)))
  check('owner gate: no page errors', realErrors(errors).length === 0, realErrors(errors).join('|').slice(0, 200))
  await ctx.close()
}

// 2. No session, Map tab: view-only. No save, link videos or add photo, no edit card, and nothing is sent.
{
  const { ctx, page, errors, st } = await session()
  await page.goto(`${BASE}/#/review?b=b1`)
  await page.locator('.map-actions').waitFor({ timeout: 5000 })
  check('map view: no Save, Link videos or Add photo for a visitor', (await page.getByRole('button', { name: /Save changes|Saved|Link videos/ }).count()) === 0 && (await page.locator('.file-input').count()) === 0)
  await page.locator('.node-card').first().click()
  await sleep(200)
  check('map view: picking a place shows no edit fields', (await page.locator('.node-editor input').count()) === 0)
  await sleep(200)
  check('map view: no PUT, consolidate or photo POST without a session', writes(st).length === 0, JSON.stringify(writes(st)))
  check('map view: no page errors', realErrors(errors).length === 0, realErrors(errors).join('|').slice(0, 200))
  await ctx.close()
}

// 3. Signed in as the owner: every write carries the bearer header; GETs still do not (except the building list, see below).
{
  const { ctx, page, errors, st } = await session({ stored: SESSION })
  await gotoBuild(page)
  await page.locator('.floor-input input').first().fill('3')
  await page.locator('.floor-input input').first().blur()
  await sleep(400)
  const patch = st.requests.filter((r) => r.method === 'PATCH')
  check('signed in: floor PATCH carries Authorization: Bearer <session>', patch.length === 1 && patch[0].auth === `Bearer ${SESSION}`, JSON.stringify(patch))
  check('signed in: floor PATCH sends start_floor', patch[0]?.body?.start_floor === 3, JSON.stringify(patch[0]?.body))

  await page.getByRole('button', { name: 'Rebuild map' }).click()
  await sleep(400)
  const rb = st.requests.filter((r) => r.path === '/buildings/b1/rebuild')
  check('signed in: rebuild POST carries the bearer header', rb.length === 1 && rb[0].auth === `Bearer ${SESSION}`, JSON.stringify(rb))

  await page.locator('button[title="Remove this video and its data"]').click()
  await sleep(400)
  const del = st.requests.filter((r) => r.method === 'DELETE')
  check('signed in: delete walk carries the bearer header', del.length === 1 && del[0].auth === `Bearer ${SESSION}`, JSON.stringify(del))
  check('signed in: no sign-in sheet on writes', (await dialog(page).count()) === 0)

  await gotoMap(page)
  await editLabel(page, 'Renamed')
  await page.getByRole('button', { name: 'Save changes' }).click()
  await sleep(400)
  const put = st.requests.filter((r) => r.method === 'PUT')
  check('signed in: save map PUT carries the bearer header', put.length === 1 && put[0].auth === `Bearer ${SESSION}`, JSON.stringify(put))

  await page.getByRole('button', { name: 'Link videos' }).click()
  await sleep(600)
  const cons = st.requests.filter((r) => r.path === '/buildings/b1/consolidate')
  check('signed in: consolidate POST carries the bearer header', cons.length === 1 && cons[0].auth === `Bearer ${SESSION}`, JSON.stringify(cons))

  await page.locator('.file-input').setInputFiles(photoFile)
  await sleep(500)
  const ph = st.requests.filter((r) => r.path === '/buildings/b1/frames/manual')
  check('signed in: manual photo POST carries the bearer header', ph.length === 1 && ph[0].auth === `Bearer ${SESSION}`, JSON.stringify(ph))

  // The building list carries the bearer (the owner sees every building); every other GET carries none.
  const gets = st.requests.filter((r) => r.method === 'GET' && !r.path.startsWith('/auth/') && r.path !== '/buildings')
  check('signed in: GET requests still carry no Authorization header', gets.length > 0 && gets.every((r) => !r.auth), JSON.stringify(gets.slice(0, 3)))
  check('signed in: the building list carries the bearer (owner sees all)', st.requests.some((r) => r.path === '/buildings' && r.method === 'GET' && r.auth === `Bearer ${SESSION}`))
  check('signed in: no page errors', realErrors(errors).length === 0, realErrors(errors).join('|').slice(0, 200))
  await ctx.close()
}

// 4. Manual photo refused: 422 shows the moderation reason, 503 shows the try-again message, no raw error dump.
{
  const { ctx, page, errors, st } = await session({ stored: SESSION })
  st.photo = { status: 422, body: { detail: 'Blocked: photo shows a face without consent.' } }
  await gotoMap(page)
  await page.locator('.file-input').setInputFiles(photoFile)
  await page.getByRole('alert').filter({ hasText: REJECT_TEXT }).waitFor({ timeout: 3000 })
  check('photo 422: the moderation reason is shown', (await page.getByText(REJECT_TEXT).count()) === 1)
  check('photo 422: no raw status text is shown', (await page.getByText(/status code/).count()) === 0)
  check('photo 422: the session is kept', (await storedToken(page)) === SESSION)
  check('photo 422: no sign-in sheet opens', (await dialog(page).count()) === 0)

  st.photo = { status: 503, body: { detail: 'moderation down' } }
  await page.locator('.file-input').setInputFiles(photoFile)
  await page.getByText(UNAVAILABLE).waitFor({ timeout: 3000 })
  check('photo 503: the try-again message is shown', (await page.getByText(UNAVAILABLE).count()) === 1)
  check('photo 503: the server text is not dumped', (await page.getByText('moderation down').count()) === 0)
  check('photo 503: the earlier 422 note is cleared', (await page.getByText(REJECT_TEXT).count()) === 0)
  check('photo 503: no page errors', realErrors(errors).length === 0, realErrors(errors).join('|').slice(0, 200))
  await ctx.close()
}

// 5. Expired session: a write refused with 401 clears the session and opens "Your sign-in ended".
{
  const { ctx, page } = await session({ stored: SESSION, status: { patch: 401 } })
  await gotoBuild(page)
  await page.locator('.floor-input input').first().fill('3')
  await page.locator('.floor-input input').first().blur()
  await dialog(page).waitFor({ timeout: 3000 })
  check('expired write: the sheet says the sign-in ended', (await page.getByText(ENDED).count()) === 1)
  check('expired write: the stored session is cleared', (await storedToken(page)) === null)
  await ctx.close()
}

await browser.close()
const failed = results.filter((r) => !r.ok)
console.log(`\n${results.length - failed.length} passed, ${failed.length} failed of ${results.length}`)
process.exit(failed.length ? 1 : 0)
