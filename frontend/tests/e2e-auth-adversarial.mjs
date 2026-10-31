// Adversarial checks for invite access. Run: node tests/e2e-auth-adversarial.mjs  (needs `playwright` resolvable and a vite dev server at BASE, default http://localhost:5199)
// The API is a stateful mock. It reads the same headers the backend will (Authorization Bearer, X-Access-Tokens) and
// answers 401 / 403 / 429 the way the contract says, so the real header logic in api.js is what is under test.
import { chromium } from 'playwright'

const BASE = process.env.BASE || 'http://localhost:5199'
const ORIGIN = new URL(BASE).origin
const OWNER = 'owner-key-0123456789abcdef'
const VIEWER = 'viewer-tok-0123456789abcdef'
const EDITOR = 'editor-tok-0123456789abcdef'
const STALE = 'stale-tok-0123456789abcdef'
const ALL_SECRETS = [OWNER, VIEWER, EDITOR, STALE]
const MAP = { building_id: 'b1', name: 'B1', nodes: [], edges: [], suggested_merges: [], videos: [] }
const DESKTOP = { width: 1280, height: 800 }

const results = []
const check = (name, ok, info = '') => {
  results.push({ name, ok: !!ok, info })
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok ? '' : '  -> ' + info}`)
}
const bearer = (h) => (/^Bearer (.+)$/.exec(h || '') || [])[1] || ''
const noSecrets = (text) => ALL_SECRETS.every((t) => !String(text).includes(t))
const realErrors = (errs) => errs.filter((e) => !/Failed to load resource/.test(e))

function makeServer(overrides = {}) {
  return {
    invites: [
      { id: 'i-viewer', role: 'viewer', label: 'Old phone', created_at: '2026-10-01T10:00:00Z', revoked: false, token: VIEWER },
      { id: 'i-editor', role: 'editor', label: 'Pat', created_at: '2026-10-02T10:00:00Z', revoked: false, token: EDITOR },
    ],
    rate: 0, // the next N access checks answer 429
    ownerRate: 0, // the next N owner-key checks (GET /buildings with a bearer) answer 429
    noBuildings: false, // a valid owner key then gets 200 [] (a server with no buildings yet)
    mapStatus: 0, // when set, GET /buildings/b1/map answers with this status
    seq: 0,
    urls: [], // every API URL requested, checked for secrets
    auths: [], // Authorization values seen on building calls
    xTokens: [], // X-Access-Tokens values seen on GET /buildings
    ...overrides,
  }
}

const roleOf = (st, tok) => {
  if (tok && tok === OWNER) return 'owner'
  const inv = tok ? st.invites.find((i) => i.token === tok && !i.revoked) : null
  return inv ? inv.role : null
}

async function handle(st, route) {
  const req = route.request()
  const p = new URL(req.url()).pathname.replace(/^\/api/, '')
  const m = req.method()
  const h = req.headers()
  const auth = bearer(h.authorization)
  const json = (b, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(b) })
  st.urls.push(req.url())
  if (auth) st.auths.push(auth)
  if (p === '/health') return json({ gemma: { ok: true, selected: 'gemma3', models: ['gemma3'] } })

  if (p === '/buildings' && m === 'GET') {
    st.xTokens.push(h['x-access-tokens'] || '')
    // Contract: an Authorization header must be a valid owner key. Anything else is refused with 401 (never 200 []),
    // and a valid owner key gets 200 with the buildings, possibly [] on a new server.
    if (auth) {
      if (st.ownerRate > 0) {
        st.ownerRate -= 1
        return json({ detail: 'Too many attempts' }, 429)
      }
      return auth === OWNER ? json(st.noBuildings ? [] : [{ id: 'b1', name: 'B1', role: 'owner' }]) : json({ detail: 'Owner key not valid' }, 401)
    }
    const roles = (h['x-access-tokens'] || '').split(',').map((t) => roleOf(st, t)).filter(Boolean)
    const best = ['owner', 'editor', 'viewer'].find((r) => roles.includes(r))
    return json(best ? [{ id: 'b1', name: 'B1', role: best }] : [])
  }
  if (p === '/buildings' && m === 'POST') return auth === OWNER ? json({ building_id: 'b2' }) : json({ detail: 'Owner key needed' }, auth ? 403 : 401)

  const bm = /^\/buildings\/b1(\/.*)?$/.exec(p)
  if (!bm) return json({ detail: 'nope' }, 404)
  const sub = bm[1] || ''
  const role = roleOf(st, auth)

  if (sub === '/access' && m === 'GET') {
    if (st.rate > 0) {
      st.rate -= 1
      return json({ detail: 'Too many attempts' }, 429)
    }
    return role ? json({ role }) : json({ detail: 'Invite not valid' }, 401)
  }
  if (!role) return json({ detail: 'Invite needed' }, 401)
  if (sub === '/map' && m === 'GET') {
    if (st.mapStatus) return json({ detail: st.mapStatus === 403 ? 'Forbidden' : 'Invite no longer valid' }, st.mapStatus)
    return json(MAP)
  }
  if (sub === '/queue') return role === 'owner' || role === 'editor' ? json([]) : json({ detail: 'Forbidden' }, 403)
  if (sub === '/invites' && m === 'GET') {
    if (role !== 'owner') return json({ detail: 'Owners only' }, 403)
    return json(st.invites.map(({ token, ...rest }) => rest))
  }
  if (sub === '/invites' && m === 'POST') {
    if (role !== 'owner') return json({ detail: 'Owners only' }, 403)
    const body = JSON.parse(req.postData() || '{}')
    st.seq += 1
    const token = `new-${st.seq}-tok-0123456789abcdef`
    const inv = { id: `i-new-${st.seq}`, role: body.role, label: body.label || '', created_at: new Date().toISOString(), revoked: false, token }
    st.invites.push(inv)
    return json({ id: inv.id, role: inv.role, label: inv.label, created_at: inv.created_at, token, join_path: `#/join?b=b1&t=${token}` })
  }
  const dm = /^\/invites\/([^/]+)$/.exec(sub)
  if (dm && m === 'DELETE') {
    if (role !== 'owner') return json({ detail: 'Owners only' }, 403)
    const inv = st.invites.find((i) => i.id === dm[1])
    if (!inv) return json({ detail: 'Invite not found' }, 404)
    inv.revoked = true
    return json({ ok: true })
  }
  return json({ detail: 'nope' }, 404)
}


const browser = await chromium.launch()
async function session(overrides = {}, init = null) {
  const st = makeServer(overrides)
  st.posts = 0
  const ctx = await browser.newContext({ viewport: DESKTOP })
  await ctx.grantPermissions(['clipboard-read', 'clipboard-write'], { origin: ORIGIN })
  if (init) await ctx.addInitScript(init)
  const page = await ctx.newPage()
  const errors = []
  page.on('console', (msg) => ['error', 'warning', 'log'].includes(msg.type()) && errors.push(msg.text()))
  page.on('pageerror', (e) => errors.push('PAGEERROR ' + e.message))
  page.on('dialog', (d) => d.dismiss())
  await page.route('**/api/**', (route) => {
    if (route.request().method() === 'POST' && /\/invites$/.test(route.request().url())) st.posts += 1
    return handle(st, route)
  })
  return { ctx, page, errors, st }
}
const tabNames = (page) => page.$$eval('.tabbar button span', (els) => els.map((e) => e.innerText.trim()))
const stored = (page) => page.evaluate(() => { try { return JSON.parse(localStorage.getItem('waypoint.tokens')) } catch { return 'unreadable' } })
const JOIN = (t, b = 'b1', extra = '') => `${BASE}/#/join?b=${b}&t=${t}${extra}`
const settle = (page) => page.waitForTimeout(700)

// 1. wrong owner key must be rejected (backend answers 200 [] for it)
{
  const { ctx, page, errors } = await session()
  await page.goto(BASE + '/#/navigate?b=b1'); await settle(page)
  await page.getByRole('button', { name: /I own this server/ }).click()
  await page.getByLabel('Owner key').fill('totally-wrong-key-123456')
  await page.getByRole('button', { name: 'Save owner key' }).click(); await settle(page)
  const k = await page.evaluate(() => localStorage.getItem('waypoint.ownerKey'))
  check('owner key: a wrong key is NOT saved', !k, `stored=${k}`)
  check('owner key: a wrong key shows a role=alert error', await page.locator('.access-screen [role=alert]').count() > 0)
  check('owner key: still on the access screen', await page.locator('.access-screen').count() > 0)
  check('owner key: no secret in console', !errors.join('|').includes('totally-wrong'))
  await ctx.close()
}
// a stored (stale) invite must not make a wrong owner key look valid: the key check sends only the key
{
  const { ctx, page, st } = await session({}, `localStorage.setItem('waypoint.tokens', JSON.stringify({ b1: '${STALE}' }))`)
  await page.goto(BASE + '/#/navigate?b=b1'); await settle(page)
  await page.getByRole('button', { name: /I own this server/ }).click()
  await page.getByLabel('Owner key').fill('totally-wrong-key-123456')
  await page.getByRole('button', { name: 'Save owner key' }).click(); await settle(page)
  const k = await page.evaluate(() => localStorage.getItem('waypoint.ownerKey'))
  check('owner key + stored invite: a wrong key is NOT saved', !k, `stored=${k}`)
  check('owner key + stored invite: the refusal is shown', /isn't valid/.test(await page.locator('.access-screen [role=alert]').first().innerText().catch(() => '')))
  check('owner key + stored invite: the key check sent no X-Access-Tokens', st.xTokens.at(-1) === '', JSON.stringify(st.xTokens))
  await ctx.close()
}
// owner key rate limited (429): the wait message, nothing saved, and the right key works on the next try
{
  const { ctx, page } = await session({ ownerRate: 1 })
  await page.goto(BASE + '/#/navigate?b=b1'); await settle(page)
  await page.getByRole('button', { name: /I own this server/ }).click()
  await page.getByLabel('Owner key').fill(OWNER)
  await page.getByRole('button', { name: 'Save owner key' }).click(); await settle(page)
  const t = await page.locator('.access-screen [role=alert]').first().innerText().catch(() => '')
  check('owner key 429: wait message shown', /wait/i.test(t), t)
  check('owner key 429: key not saved', !(await page.evaluate(() => localStorage.getItem('waypoint.ownerKey'))))
  check('owner key 429: key not echoed', !t.includes(OWNER))
  await page.getByRole('button', { name: 'Save owner key' }).click(); await settle(page)
  check('owner key 429: the right key is accepted on retry', (await tabNames(page)).length === 4, JSON.stringify(await tabNames(page)))
  await ctx.close()
}
// the right key is accepted
{
  const { ctx, page } = await session()
  await page.goto(BASE + '/#/navigate?b=b1'); await settle(page)
  await page.getByRole('button', { name: /I own this server/ }).click()
  await page.getByLabel('Owner key').fill(OWNER)
  await page.getByRole('button', { name: 'Save owner key' }).click(); await settle(page)
  check('owner key: right key accepted, 4 tabs', (await tabNames(page)).length === 4, JSON.stringify(await tabNames(page)))
  await ctx.close()
}
// 2. history and console
{
  const { ctx, page, errors } = await session()
  await page.goto(BASE + '/#/navigate?b=b1'); await settle(page)
  const before = await page.evaluate(() => history.length)
  await page.goto(JOIN(VIEWER)); await settle(page)
  const entries = await page.evaluate(() => history.length)
  check('history: join link adds at most one entry vs plain navigation', entries - before <= 1, `${before}->${entries}`)
  await page.goBack().catch(() => {}); await settle(page)
  check('history: going back never shows the token', !page.url().includes(VIEWER), page.url())
  check('history: no join entry reachable by going back', !page.url().includes('join'), page.url())
  check('console: no token logged', ALL_SECRETS.every((t) => !errors.join('|').includes(t)))
  await ctx.close()
}
// 3. extra params, special-character ids, role change by re-join, forbidden deep link
{
  const { ctx, page } = await session()
  await page.goto(JOIN(VIEWER, 'b1', '&utm=x&b=other&t=zzz')); await settle(page)
  check('extra params: link still joins (first b/t wins) and strips URL', !page.url().includes(VIEWER) && !page.url().includes('utm'), page.url())
  check('extra params: viewer sees only Directions', JSON.stringify(await tabNames(page)) === '["Directions"]', JSON.stringify(await tabNames(page)))
  await page.goto(BASE + '/#/review?b=b1'); await settle(page)
  check('deep link: viewer to #/review lands on navigate', page.url().includes('#/navigate'), page.url())
  await page.goto(BASE + '/#/build?b=b1'); await settle(page)
  check('deep link: viewer to #/build lands on navigate', page.url().includes('#/navigate'), page.url())
  await page.goto(JOIN(EDITOR)); await settle(page)
  check('re-join: editor token upgrades to 4 tabs', (await tabNames(page)).length === 4, JSON.stringify(await tabNames(page)))
  check('re-join: editor has no Share panel', (await page.locator('.share-panel').count()) === 0)
  await ctx.close()
}
for (const id of ['a%20b', 'a%26t%3Dx', '..%2F..%2Fetc', '%3Cscript%3E', '%E0%A4%A', 'x'.repeat(300), '']) {
  const { ctx, page, errors } = await session()
  await page.goto(JOIN(VIEWER, id)); await settle(page)
  const s = await stored(page)
  const bad = Object.keys(s || {}).some((k) => /[\s/&=]/.test(k) || k.length > 128)
  check(`odd building id "${id.slice(0, 20)}": no unsafe key stored, token out of URL, no page error`,
    !bad && !page.url().includes(VIEWER) && !errors.some((e) => /PAGEERROR/.test(e)), `${page.url()} ${JSON.stringify(s)} ${errors.slice(-1)}`)
  await ctx.close()
}
// 4. storage failures
{
  const throwing = () => {
    const boom = () => { throw new DOMException('denied', 'SecurityError') }
    Storage.prototype.getItem = boom; Storage.prototype.setItem = boom; Storage.prototype.removeItem = boom
  }
  const { ctx, page, errors } = await session({}, throwing)
  await page.goto(JOIN(VIEWER)); await settle(page)
  check('storage throws: join still works for this page load', JSON.stringify(await tabNames(page)) === '["Directions"]', JSON.stringify(await tabNames(page)) + page.url())
  check('storage throws: no page error', !errors.some((e) => /PAGEERROR/.test(e)), errors.join('|').slice(0, 200))
  await ctx.close()
}
{
  const full = () => {
    const orig = Storage.prototype.setItem
    Storage.prototype.setItem = function (k, v) { if (/waypoint/.test(k)) throw new DOMException('quota', 'QuotaExceededError'); return orig.call(this, k, v) }
  }
  const { ctx, page, errors } = await session({}, full)
  await page.goto(JOIN(VIEWER)); await settle(page)
  check('storage full: join still works', JSON.stringify(await tabNames(page)) === '["Directions"]', JSON.stringify(await tabNames(page)))
  check('storage full: no page error', !errors.some((e) => /PAGEERROR/.test(e)))
  await ctx.close()
}
for (const junk of ['not json{', '[1,2]', '"str"', 'null', '{"b1":123,"b 2":"x","b3":"has space"}']) {
  const { ctx, page, errors } = await session({}, `localStorage.setItem('waypoint.tokens', ${JSON.stringify(junk)})`)
  await page.goto(BASE + '/#/navigate?b=b1'); await settle(page)
  check(`malformed tokens ${junk.slice(0, 14)}: access screen shown, no crash`, (await page.locator('.access-screen').count()) > 0 && !errors.some((e) => /PAGEERROR/.test(e)), errors.join('|').slice(0, 150))
  await ctx.close()
}
// 5. owner: XSS label, long label, double click, tap targets, keyboard
{
  const evil = '<img src=x onerror="window.__xss=1">'
  const long = 'L'.repeat(3000)
  const { ctx, page, errors, st } = await session({})
  st.invites.push({ id: 'i-xss', role: 'viewer', label: evil, created_at: '2026-10-03T10:00:00Z', revoked: false, token: 'x1x1x1x1x1x1x1x1x1x1' })
  st.invites.push({ id: 'i-long', role: 'viewer', label: long, created_at: '2026-10-03T10:00:00Z', revoked: false, token: 'y1y1y1y1y1y1y1y1y1y1' })
  await page.addInitScript(`localStorage.setItem('waypoint.ownerKey', '${OWNER}')`)
  await page.goto(BASE + '/#/build?b=b1'); await settle(page)
  check('share: owner sees Share panel on Build', (await page.locator('.share-panel').count()) === 1)
  check('xss: label rendered as text, no script ran', !(await page.evaluate(() => window.__xss)) && (await page.locator('.invite-list img').count()) === 0)
  check('xss: label text visible literally', (await page.locator('.invite-list').innerText()).includes('<img'))
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
  check('long label: no horizontal page overflow', overflow <= 1, `overflow=${overflow}`)
  await page.setViewportSize({ width: 375, height: 800 }); await settle(page)
  const overflowM = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
  check('long label: no horizontal overflow at 375px', overflowM <= 1, `overflow=${overflowM}`)
  await page.setViewportSize(DESKTOP)
  await page.getByRole('button', { name: 'Create link' }).dblclick(); await settle(page)
  check('double click Create link makes exactly one invite', st.posts === 1, `posts=${st.posts}`)
  const small = await page.$$eval('.share-panel button, .share-panel input[type=text]', (els) =>
    els.filter((e) => { const r = e.getBoundingClientRect(); return r.width && r.height < 44 }).map((e) => (e.innerText || e.getAttribute('aria-label') || e.type).slice(0, 20)))
  check('tap targets: Share panel controls are at least 44px tall', small.length === 0, small.join(','))
  // tab away: link shown once
  await page.getByRole('button', { name: 'Directions' }).click(); await page.getByRole('button', { name: 'Build' }).click(); await settle(page)
  check('share: made link gone after switching tabs', (await page.locator('.share-made').count()) === 0)
  // keyboard revoke
  const btn = page.getByRole('button', { name: 'Revoke Old phone' })
  await btn.focus(); await page.keyboard.press('Enter')
  check('keyboard: confirm focus lands on Keep it', await page.evaluate(() => document.activeElement?.innerText) === 'Keep it')
  await page.keyboard.press('Escape'); await settle(page)
  check('keyboard: after cancelling revoke, focus is not lost to body', await page.evaluate(() => document.activeElement?.tagName) !== 'BODY')
  check('console: no owner key in console', !errors.join('|').includes(OWNER))
  await ctx.close()
}
// 6. 401 / 403 / 429
{
  const { ctx, page } = await session({ mapStatus: 403 })
  await page.goto(JOIN(EDITOR)); await settle(page)
  await page.getByRole('button', { name: 'Map' }).click(); await settle(page)
  const s = await stored(page)
  check('403 on action: not signed out, token kept', s?.b1 === EDITOR, JSON.stringify(s))
  check('403 on action: access screen not shown', (await page.locator('.access-screen').count()) === 0)
  await ctx.close()
}
{
  const { ctx, page } = await session({ rate: 5 })
  await page.goto(BASE + '/#/navigate?b=b1'); await settle(page)
  await page.getByLabel('Paste an invite link').fill(`#/join?b=b1&t=${VIEWER}`)
  await page.getByRole('button', { name: 'Open building' }).click(); await settle(page)
  const t = await page.locator('.access-screen [role=alert]').first().innerText().catch(() => '')
  check('429: wait message shown', /wait/i.test(t), t)
  check('429: token not saved', !(await stored(page))?.b1)
  check('429: pasted token not echoed', !t.includes(VIEWER))
  await ctx.close()
}
// 7. stale photo URLs. A signed photo URL goes stale after a revoke or when its day is up, and the server answers 404.
//    A failed photo must show its placeholder and refetch the view's data ONCE for fresh URLs, never in a loop.
//    The photo server is a stub: a URL is served only while its signature is valid. Responses carry the current
//    signature (st.issue). Modes: 'once' (old URLs dead; the first 404 makes the server issue fresh ones), 'always'
//    (old URLs dead and the server keeps issuing them), 'hold' (like 'once', but the old URLs wait until st.open()
//    is called, so the test can make an unsaved edit before the photo fails).
const PNG = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=', 'base64')
const photoPath = (name, sig) => `/bfiles/b1/frames/${name}.jpg?exp=9999999999&sig=${sig}`
const PHOTO_NODES = [{ id: 'n1', label: 'Room 112', frame: 'f-a' }, { id: 'n2', label: 'Room 201', frame: 'f-b' }]
const mapBody = (sig) => ({
  building_id: 'b1',
  name: 'B1',
  nodes: PHOTO_NODES.map((n, k) => ({ id: n.id, label: n.label, type: 'room', floor: 1, x: k * 10, y: 0, verified: true, texts: [], sign_texts: [], description: '', frame_id: n.frame, frame_url: photoPath(n.frame, sig) })),
  edges: [{ id: 'e1', from_node: 'n1', to_node: 'n2', heading_out: 90, heading_in: 90, vertical: 'none', length_m: 10, frames_fwd: [], frames_rev: null, source: 'walk', verified: true }],
  suggested_merges: [],
  videos: [{ id: 'v1', filename: 'walk.mp4', status: 'done', duration: 12, keyframes: 2, start_floor: 1 }],
})
const stepsBody = (sig) => ({
  steps: [
    { step_index: 0, node_id: 'n1', label: 'Room 112', frame_url: photoPath('f-a', sig), action: 'start', vertical: 'none', instruction_text: 'Start in Room 112', floor: 1, turn_degrees: 0 },
    { step_index: 1, node_id: 'n2', label: 'Room 201', frame_url: photoPath('f-b', sig), action: 'destination_reached', vertical: 'none', instruction_text: 'You are at Room 201', floor: 1, turn_degrees: 0 },
  ],
  node_path: ['n1', 'n2'],
})
const debugBody = (sig) => ({
  source: 'imu', t: [0, 4, 8], heading: [0, 40, 90], events: [],
  keyframes: [
    { frame_id: 'f-a', t: 0, heading: 0, kind: 'walk', score: 80, soft: false, analysis: null, url: photoPath('f-a', sig) },
    { frame_id: 'f-b', t: 8, heading: 90, kind: 'turn', score: 75, soft: false, analysis: null, url: photoPath('f-b', sig) },
  ],
})

async function photoSession(mode) {
  const st = {
    valid: new Set(['new']),
    issue: 'old',
    autoFresh: mode !== 'always',
    seen404: false,
    calls: { route: 0, map: 0, debug: 0 },
    after: { route: 0, map: 0, debug: 0 }, // calls made after the first photo failed
    open: () => {},
  }
  // 'hold': stale photo requests wait here until the test calls st.open()
  const gate = mode === 'hold' ? new Promise((r) => { st.open = r }) : Promise.resolve()
  const ctx = await browser.newContext({ viewport: DESKTOP })
  await ctx.addInitScript(`localStorage.setItem('waypoint.ownerKey', '${OWNER}')`)
  const page = await ctx.newPage()
  const errors = []
  page.on('pageerror', (e) => errors.push('PAGEERROR ' + e.message))
  page.on('dialog', (d) => d.dismiss())
  const count = (kind) => {
    st.calls[kind] += 1
    if (st.seen404) st.after[kind] += 1
  }
  await page.route('**/bfiles/**', async (route) => {
    const sig = new URL(route.request().url()).searchParams.get('sig')
    const ok = st.valid.has(sig)
    if (!ok) await gate
    if (!ok) {
      st.seen404 = true
      if (st.autoFresh) st.issue = 'new'
    }
    // no-store: every <img> asks the stub again, so a stale URL is really re-requested.
    const headers = { 'cache-control': 'no-store' }
    return route.fulfill(ok ? { status: 200, contentType: 'image/png', body: PNG, headers } : { status: 404, contentType: 'text/plain', body: 'gone', headers })
  })
  await page.route('**/api/**', (route) => {
    const req = route.request()
    const p = new URL(req.url()).pathname.replace(/^\/api/, '')
    const json = (b, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(b) })
    if (p === '/health') return json({ gemma: { ok: true, selected: 'gemma3', models: ['gemma3'] } })
    if (p === '/buildings' && req.method() === 'GET') return json([{ id: 'b1', name: 'B1', role: 'owner' }])
    if (p === '/buildings/b1/map') { count('map'); return json(mapBody(st.issue)) }
    if (p === '/buildings/b1/route' && req.method() === 'POST') { count('route'); return json(stepsBody(st.issue)) }
    if (p === '/buildings/b1/videos/v1/debug') { count('debug'); return json(debugBody(st.issue)) }
    if (p === '/buildings/b1/queue') return json([])
    return json({ detail: 'nope' }, 404)
  })
  return { ctx, page, errors, st }
}

// Directions: the scene photo fails, the trip is refetched once and the fresh photo shows.
{
  const { ctx, page, errors, st } = await photoSession('once')
  await page.goto(`${BASE}/#/navigate?b=b1&from=n1&to=n2`)
  await page.waitForFunction(() => document.querySelector('.scene img[src*="sig=new"]'), null, { timeout: 6000 }).catch(() => {})
  await settle(page)
  check('photo 404 (directions): fresh photo shown after the refetch', (await page.locator('.scene img[src*="sig=new"]').count()) === 1)
  check('photo 404 (directions): the shown photo loaded (not broken)', await page.locator('.scene img').first().evaluate((i) => i.complete && i.naturalWidth > 0).catch(() => false))
  check('photo 404 (directions): route refetched exactly once', st.calls.route === 2 && st.after.route === 1, JSON.stringify(st.calls))
  check('photo 404 (directions): no page errors', !errors.some((e) => /PAGEERROR/.test(e)), errors.join('|').slice(0, 200))
  await ctx.close()
}
// Directions: stale again after the refetch: placeholder kept, one refetch only, no loop.
{
  const { ctx, page, errors, st } = await photoSession('always')
  await page.goto(`${BASE}/#/navigate?b=b1&from=n1&to=n2`)
  await page.waitForSelector('.scene .no-photo', { timeout: 6000 }).catch(() => {})
  await page.waitForTimeout(1200)
  check('photo always stale (directions): placeholder shown, no broken image', (await page.locator('.scene .no-photo').count()) === 1 && (await page.locator('.scene img').count()) === 0)
  check('photo always stale (directions): exactly one refetch, no loop', st.calls.route === 2 && st.after.route === 1, JSON.stringify(st.calls))
  check('photo always stale (directions): no page errors', !errors.some((e) => /PAGEERROR/.test(e)), errors.join('|').slice(0, 200))
  await ctx.close()
}
// Map: an unsaved label edit survives the photo refetch, and the fresh photos replace the stale ones.
{
  const { ctx, page, errors, st } = await photoSession('hold')
  await page.goto(`${BASE}/#/review?b=b1`)
  await page.locator('.node-grid .node-card').first().waitFor({ timeout: 6000 }).catch(() => {})
  await page.locator('.node-card', { hasText: 'Room 112' }).click()
  await page.locator('.node-editor').getByLabel('Label', { exact: true }).fill('Room 112 edited')
  check('map photo 404: the edit is unsaved before the photos fail', (await page.getByRole('button', { name: 'Save changes' }).count()) === 1)
  st.open()
  await page.waitForFunction(() => document.querySelector('.node-editor img.node-photo[src*="sig=new"]'), null, { timeout: 6000 }).catch(() => {})
  await page.waitForFunction(() => document.querySelectorAll('.node-grid .node-card img[src*="sig=new"]').length === 2, null, { timeout: 6000 }).catch(() => {})
  await settle(page)
  check('map photo 404: the fresh photo shows for the selected place', (await page.locator('.node-editor img.node-photo[src*="sig=new"]').count()) === 1)
  check('map photo 404: the fresh photos show in the grid too', (await page.locator('.node-grid .node-card img[src*="sig=new"]').count()) === 2)
  check('map photo 404: the map was refetched exactly once', st.after.map === 1, JSON.stringify(st.after))
  await page.locator('.node-card', { hasText: 'Room 201' }).click()
  await page.locator('.node-card', { hasText: 'Room 112' }).click()
  check('map photo 404: unsaved edit kept after the refetch', (await page.locator('.node-editor').getByLabel('Label', { exact: true }).inputValue()) === 'Room 112 edited')
  check('map photo 404: still unsaved (Save changes)', (await page.getByRole('button', { name: 'Save changes' }).count()) === 1)
  check('map photo 404: no page errors', !errors.some((e) => /PAGEERROR/.test(e)), errors.join('|').slice(0, 200))
  await ctx.close()
}
// Map: stale from the start, placeholders in every card, one refetch, no loop.
{
  const { ctx, page, errors, st } = await photoSession('always')
  await page.goto(`${BASE}/#/review?b=b1`)
  await page.waitForFunction(() => document.querySelectorAll('.node-grid .node-card .nophoto').length === 2, null, { timeout: 6000 }).catch(() => {})
  await page.waitForTimeout(1000)
  check('map photos always stale: placeholders shown, no broken image', (await page.locator('.node-grid .node-card .nophoto').count()) === 2 && (await page.locator('.node-grid .node-card img').count()) === 0)
  check('map photos always stale: exactly one map refetch, no loop', st.after.map === 1, JSON.stringify(st.after))
  check('map photos always stale: no page errors', !errors.some((e) => /PAGEERROR/.test(e)), errors.join('|').slice(0, 200))
  await ctx.close()
}
// Build: the walk's "What the pipeline saw" keyframes fail, one debug refetch, fresh thumbnails.
{
  const { ctx, page, errors, st } = await photoSession('once')
  await page.goto(`${BASE}/#/build?b=b1`)
  await page.getByRole('button', { name: 'Inspect' }).first().click()
  await page.waitForFunction(() => document.querySelectorAll('.strip .thumb img[src*="sig=new"]').length === 2, null, { timeout: 6000 }).catch(() => {})
  await settle(page)
  check('photo 404 (keyframes): fresh thumbnails shown', (await page.locator('.strip .thumb img[src*="sig=new"]').count()) === 2)
  check('photo 404 (keyframes): thumbnails loaded (not broken)', await page.$$eval('.strip .thumb img', (els) => els.length === 2 && els.every((i) => i.complete && i.naturalWidth > 0)))
  check('photo 404 (keyframes): analysis refetched exactly once', st.after.debug === 1, JSON.stringify(st.after))
  check('photo 404 (keyframes): no page errors', !errors.some((e) => /PAGEERROR/.test(e)), errors.join('|').slice(0, 200))
  await ctx.close()
}
// Build: stale keyframes from the start: placeholders, one refetch, no loop.
{
  const { ctx, page, errors, st } = await photoSession('always')
  await page.goto(`${BASE}/#/build?b=b1`)
  await page.getByRole('button', { name: 'Inspect' }).first().click()
  await page.waitForFunction(() => document.querySelectorAll('.strip .thumb .nophoto').length === 2, null, { timeout: 6000 }).catch(() => {})
  await page.waitForTimeout(1000)
  check('keyframes always stale: placeholders shown, no broken thumbnail', (await page.locator('.strip .thumb .nophoto').count()) === 2 && (await page.locator('.strip .thumb img').count()) === 0)
  check('keyframes always stale: exactly one analysis refetch, no loop', st.after.debug === 1, JSON.stringify(st.after))
  check('keyframes always stale: no page errors', !errors.some((e) => /PAGEERROR/.test(e)), errors.join('|').slice(0, 200))
  await ctx.close()
}
// 7b. static wiring: every server photo goes through ServerPhoto, which falls back and asks for fresh URLs once.
{
  const fs = await import('node:fs')
  const read = (f) => fs.readFileSync(new URL(`../src/components/${f}`, import.meta.url), 'utf8')
  const sp = read('ServerPhoto.jsx')
  const nav = read('Navigator.jsx')
  const rev = read('MapReview.jsx')
  const vd = read('VideoDebug.jsx')
  check('photo 404: ServerPhoto falls back to the placeholder on error', /onError/.test(sp) && /placeholder/.test(sp))
  check('photo 404: Navigator scene photo uses ServerPhoto with a refetch hook', /<ServerPhoto/.test(nav) && /onStale=\{onPhotoStale\}/.test(nav))
  check('photo 404: MapReview photos use ServerPhoto with a refetch hook', /<ServerPhoto/.test(rev) && /onStale=\{onPhotoStale\}/.test(rev) && !/<img /.test(rev))
  check('photo 404: VideoDebug photos use ServerPhoto with a refetch hook', /<ServerPhoto/.test(vd) && /onStale=\{onPhotoStale\}/.test(vd) && !/<img /.test(vd))
}

await browser.close()
const failed = results.filter((r) => !r.ok)
console.log(`\n${results.length - failed.length} passed, ${failed.length} failed`)
process.exit(failed.length ? 1 : 0)
