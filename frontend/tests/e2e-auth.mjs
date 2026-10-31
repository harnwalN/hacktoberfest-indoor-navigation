// Invite access behaviour. Run: node tests/e2e-auth.mjs  (needs `playwright` resolvable and a vite dev server at BASE, default http://localhost:5199)
// The API is a stateful mock. It reads the same headers the backend will (Authorization Bearer, X-Access-Tokens) and
// answers 401 / 403 / 429 the way the contract says, so the real header logic in api.js is what is under test.
import { chromium } from 'playwright'

const BASE = process.env.BASE || 'http://localhost:5199'
const ORIGIN = new URL(BASE).origin
const OWNER = 'owner-key-0123456789abcdef'
const VIEWER = 'viewer-tok-0123456789abcdef'
const EDITOR = 'editor-tok-0123456789abcdef'
const STALE = 'stale-tok-0123456789abcdef'
const STALE_OWNER = 'stale-owner-0123456789abcdef' // an owner key the server no longer accepts
const ALL_SECRETS = [OWNER, VIEWER, EDITOR, STALE, STALE_OWNER]
const KEY_REMOVED = 'Your saved owner key no longer works, so it was removed from this device.'
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
    noBuildings: false, // a valid owner key then gets 200 [] and building calls answer 404 (a new server)
    mapStatus: 0, // when set, GET /buildings/b1/map answers with this status
    seq: 0,
    urls: [], // every API URL requested, checked for secrets
    auths: [], // Authorization values seen on any call
    bAuths: [], // Authorization values seen on building calls (/buildings/b1...), not the list
    xTokens: [], // X-Access-Tokens values seen on GET /buildings
    lists: [], // one entry per GET /buildings: { owner: whether an Authorization header was sent }
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
  if (auth && /^\/buildings\/b1/.test(p)) st.bAuths.push(auth) // building calls only (not the list)
  if (p === '/health') return json({ gemma: { ok: true, selected: 'gemma3', models: ['gemma3'] } })

  if (p === '/buildings' && m === 'GET') {
    st.xTokens.push(h['x-access-tokens'] || '')
    st.lists.push({ owner: !!auth, xt: h['x-access-tokens'] || '' })
    // Contract: an Authorization header must be a valid owner key. Anything else is refused with 401 (never 200 []).
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
  if (!bm || st.noBuildings) return json({ detail: 'Building not found' }, 404)
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
async function session(overrides = {}) {
  const st = makeServer(overrides)
  const ctx = await browser.newContext({ viewport: DESKTOP })
  await ctx.grantPermissions(['clipboard-read', 'clipboard-write'], { origin: ORIGIN })
  const page = await ctx.newPage()
  const errors = []
  const dialogs = []
  page.on('console', (msg) => ['error', 'warning'].includes(msg.type()) && errors.push(msg.text()))
  page.on('pageerror', (e) => errors.push('PAGEERROR ' + e.message))
  page.on('dialog', (d) => {
    dialogs.push(d.message())
    d.dismiss()
  })
  await page.route('**/api/**', (route) => handle(st, route))
  return { ctx, page, errors, dialogs, st }
}

const stored = (page) => page.evaluate(() => { try { return JSON.parse(localStorage.getItem('waypoint.tokens')) } catch { return 'unreadable' } })
const ownerStored = (page) => page.evaluate(() => localStorage.getItem('waypoint.ownerKey'))
const tabNames = (page) => page.$$eval('.tabbar button span', (els) => els.map((e) => e.innerText.trim()))
const heading = (page) => page.getByRole('heading', { level: 2 })
const alertText = (page) => page.locator('.access-screen [role=alert]').first().innerText().catch(() => '')
const focusedTag = (page) => page.evaluate(() => document.activeElement?.tagName || '')
const hashFromUrl = (page) => new URL(page.url()).hash
const JOIN = (t, b = 'b1') => `${BASE}/#/join?b=${b}&t=${t}`

// ---------- join link: stores the token and takes it out of the address bar ----------
{
  const { ctx, page, errors, st } = await session()
  await page.goto(JOIN(VIEWER))
  await page.waitForSelector('#from', { timeout: 8000 }).catch(() => {})
  check('join: lands on Directions for the building', hashFromUrl(page) === '#/navigate?b=b1', page.url())
  check('join: token saved for building b1', (await stored(page))?.b1 === VIEWER, JSON.stringify(await stored(page)))
  check('join: token removed from the address bar', !page.url().includes(VIEWER) && !page.url().includes('join'), page.url())
  check('join: no request URL carries the token', st.urls.every((u) => !u.includes(VIEWER)))
  check('join: building calls send the token as a Bearer header', st.auths.includes(VIEWER))
  check('join: token list sent in X-Access-Tokens', st.xTokens.some((x) => x === VIEWER), JSON.stringify(st.xTokens))
  check('join: no token in console output', noSecrets(errors.join('|')), errors.join('|').slice(0, 200))
  check('join: no page errors', !errors.some((e) => /PAGEERROR/.test(e)), errors.join('|').slice(0, 200))
  await ctx.close()
}

// ---------- roles decide the tabs ----------
{
  const { ctx, page } = await session()
  await page.goto(JOIN(VIEWER))
  await page.waitForSelector('#from', { timeout: 8000 }).catch(() => {})
  check('viewer: tab bar holds only Directions', JSON.stringify(await tabNames(page)) === '["Directions"]', JSON.stringify(await tabNames(page)))
  await page.goto(`${BASE}/#/review?b=b1`)
  await page.waitForFunction(() => location.hash.startsWith('#/navigate'), null, { timeout: 4000 }).catch(() => {})
  check('viewer: #/review falls back to Directions', hashFromUrl(page) === '#/navigate?b=b1', page.url())
  await page.goto(`${BASE}/#/build?b=b1`)
  await page.waitForFunction(() => location.hash.startsWith('#/navigate'), null, { timeout: 4000 }).catch(() => {})
  check('viewer: #/build falls back to Directions', hashFromUrl(page) === '#/navigate?b=b1', page.url())
  await page.waitForSelector('#from', { timeout: 4000 }).catch(() => {})
  check('viewer: no Share panel, no New building, no map', (await page.locator('.share-panel, .new-building, .map-panel, .node-grid').count()) === 0)
  await ctx.close()
}
{
  const { ctx, page } = await session()
  await page.goto(JOIN(EDITOR))
  await page.waitForSelector('#from', { timeout: 8000 }).catch(() => {})
  check('editor: Directions, Map, Record and Build', JSON.stringify(await tabNames(page)) === '["Directions","Map","Record","Build"]', JSON.stringify(await tabNames(page)))
  await page.locator('.tabbar button', { hasText: 'Build' }).click()
  await page.getByRole('heading', { name: 'Add a walkthrough' }).waitFor({ timeout: 5000 }).catch(() => {})
  check('editor: Build shows the upload form', (await page.getByRole('heading', { name: 'Add a walkthrough' }).count()) === 1)
  check('editor: no Share panel and no New building', (await page.locator('.share-panel, .new-building').count()) === 0)
  await ctx.close()
}

// ---------- owner: no invite, owner key, share panel ----------
{
  const { ctx, page, errors, dialogs, st } = await session()
  await page.goto(`${BASE}/#/navigate?b=b1`)
  await heading(page).waitFor({ timeout: 5000 })
  check('no invite: heading says the building is invite-only', /invite-only/.test(await heading(page).innerText()), await heading(page).innerText())
  check('no invite: plain sentence shown', (await page.textContent('.access-screen')).includes('You need an invite link to open this building'))
  check('no invite: tab bar hidden on the access screen', (await page.locator('.tabbar').count()) === 0)
  await page.waitForTimeout(150)
  check('no invite: the heading takes focus', (await focusedTag(page)) === 'H2', await focusedTag(page))
  await page.keyboard.press('Tab')
  check('no invite: keyboard Tab reaches the paste field', await page.evaluate(() => document.activeElement?.getAttribute('placeholder') === 'Paste the whole link'))
  const toggle = page.getByRole('button', { name: 'I own this server' })
  check('owner key: hidden behind a toggle (collapsed)', (await toggle.getAttribute('aria-expanded')) === 'false')
  await toggle.click()
  check('owner key: toggle expands', (await toggle.getAttribute('aria-expanded')) === 'true')
  await page.getByLabel('Owner key').fill('not-the-key-0000000000')
  await page.getByRole('button', { name: 'Save owner key' }).click()
  await page.locator('.access-screen [role=alert]').waitFor({ timeout: 5000 }).catch(() => {})
  check('owner key: a wrong key is refused with a message', /isn't valid/.test(await alertText(page)), await alertText(page))
  check('owner key: a wrong key is not saved', (await ownerStored(page)) === null)
  await page.getByLabel('Owner key').fill(OWNER)
  await page.getByRole('button', { name: 'Save owner key' }).click()
  await page.locator('.tabbar button').nth(3).waitFor({ timeout: 5000 }).catch(() => {})
  check('owner key: tabs unlock with four tabs', (await tabNames(page)).length === 4, JSON.stringify(await tabNames(page)))
  check('owner key: saved on this device', (await ownerStored(page)) === OWNER)
  check('owner key: never in the URL', !page.url().includes(OWNER))

  await page.locator('.tabbar button', { hasText: 'Build' }).click()
  await page.locator('.share-panel').waitFor({ timeout: 5000 }).catch(() => {})
  check('owner: Share panel on Build', (await page.locator('.share-panel').count()) === 1)
  check('owner: New building card shown', (await page.locator('.new-building').count()) === 1)
  await page.locator('.invite-list li').nth(1).waitFor({ timeout: 5000 }).catch(() => {})
  check('owner: existing invites listed with Revoke buttons', (await page.getByRole('button', { name: /^Revoke / }).count()) === 2)

  // create a link
  await page.getByRole('radio', { name: /Editor/ }).check()
  await page.getByLabel('Label (optional)').fill("Alex's phone")
  await page.getByRole('button', { name: 'Create link' }).click()
  await page.locator('.share-made').waitFor({ timeout: 5000 }).catch(() => {})
  const link = await page.locator('.share-made input[readonly]').inputValue().catch(() => '')
  const newToken = (/#\/join\?b=b1&t=([^&]+)$/.exec(link) || [])[1] || ''
  check('create: shows the full link (origin + path + join path)', link.startsWith(`${ORIGIN}/`) && link.includes('#/join?b=b1&t=') && !!newToken, link)
  check('create: "Shown once, copy it now." note', /Shown once, copy it now\./.test(await page.textContent('.share-made').catch(() => '')))
  check('create: new link listed as Editor with its label', /Alex's phone/.test(await page.locator('.invite-list').innerText()) && /Editor/.test(await page.locator('.invite-list').innerText()))
  check('create: the token is not in the address bar', !page.url().includes(newToken))
  check('create: the link is only sent to the server once, by the create call', st.urls.filter((u) => u.includes('/invites')).length >= 2)

  await page.getByRole('button', { name: 'Copy link' }).click()
  await page.getByText('Copied to the clipboard.').waitFor({ timeout: 3000 }).catch(() => {})
  check('copy: reports success', /Copied/.test(await page.locator('.share-made [role=status]').innerText().catch(() => '')))
  const clip = await page.evaluate(() => navigator.clipboard.readText()).catch(() => '')
  check('copy: clipboard holds the full link', clip === link, clip)

  const probe = (t) => page.evaluate(async (tok) => (await fetch('/api/buildings/b1/access', { headers: { Authorization: `Bearer ${tok}` } })).status, t)
  check('create: the new link opens the building (access 200)', (await probe(newToken)) === 200)

  await page.locator('.tabbar button', { hasText: 'Map' }).click()
  await page.locator('.tabbar button', { hasText: 'Build' }).click()
  await page.locator('.share-panel').waitFor({ timeout: 5000 }).catch(() => {})
  check('create: the link is not shown again after leaving the tab', (await page.locator('.share-made').count()) === 0)

  // revoke: inline confirm, no native dialogs
  const before = dialogs.length
  await page.getByRole('button', { name: "Revoke Alex's phone" }).click()
  check('revoke: asks inline, in the page', await page.getByText('Revoke this link? It stops working straight away.').isVisible())
  await page.getByRole('button', { name: 'Keep it' }).click()
  check('revoke: "Keep it" leaves the link working', (await page.getByRole('button', { name: "Revoke Alex's phone" }).count()) === 1 && st.invites.find((i) => i.label === "Alex's phone")?.revoked === false)
  await page.getByRole('button', { name: "Revoke Alex's phone" }).click()
  await page.getByRole('button', { name: 'Yes, revoke' }).click()
  await page.locator('.invite-row.is-revoked').first().waitFor({ timeout: 5000 }).catch(() => {})
  check('revoke: row shows Revoked', /Revoked/.test(await page.locator('.invite-row.is-revoked').first().innerText().catch(() => '')))
  check('revoke: server marks the link revoked', st.invites.find((i) => i.label === "Alex's phone")?.revoked === true)
  check('revoke: the revoked token is refused (401)', (await probe(newToken)) === 401)
  check('revoke: no window.confirm or alert used', dialogs.length === before, JSON.stringify(dialogs))
  check('owner: no page errors', !errors.some((e) => /PAGEERROR/.test(e)), errors.join('|').slice(0, 200))
  check('owner: no secret in console output', noSecrets(errors.join('|')), errors.join('|').slice(0, 200))
  await ctx.close()
}

// ---------- a valid owner key on a server with no buildings yet: accepted, and the create-a-building view ----------
{
  const { ctx, page, errors, st } = await session({ noBuildings: true })
  // a saved building from before the server was reset: must not be opened
  await page.goto(`${BASE}/#/navigate?b=b1`)
  await page.evaluate(() => localStorage.setItem('building', 'b1'))
  await page.reload()
  await heading(page).waitFor({ timeout: 5000 }).catch(() => {})
  check('empty server: no owner key yet, so the access screen is shown', (await page.locator('.access-screen').count()) === 1)
  await page.getByRole('button', { name: 'I own this server' }).click()
  await page.getByLabel('Owner key').fill(OWNER)
  await page.getByRole('button', { name: 'Save owner key' }).click()
  await page.getByRole('heading', { name: 'New building' }).waitFor({ timeout: 5000 }).catch(() => {})
  await page.waitForTimeout(150)
  check('empty server: a valid owner key is accepted (no refusal shown)', (await page.locator('.access-screen').count()) === 0 && (await page.locator('[role=alert]').count()) === 0)
  check('empty server: the owner key is saved on this device', (await ownerStored(page)) === OWNER)
  check('empty server: the create-a-building view is shown', (await page.getByRole('heading', { name: 'New building' }).count()) === 1 && (await page.getByLabel('New building name').count()) === 1)
  check('empty server: the building picker says there are no buildings yet', /No buildings yet/.test(await page.locator('select[aria-label="Building"]').innerText()))
  check('empty server: no Directions planner for a building that does not exist', (await page.locator('#from').count()) === 0 && (await page.getByText('Could not load the map').count()) === 0)
  // the owner shell: every tab is listed, the ones that need a building are disabled, Build is the open tab
  check('empty server: owner sees all four tabs', JSON.stringify(await tabNames(page)) === '["Directions","Map","Record","Build"]', JSON.stringify(await tabNames(page)))
  const disabled = await page.$$eval('.tabbar button', (els) => els.map((e) => e.disabled))
  check('empty server: Directions, Map and Record are disabled', JSON.stringify(disabled) === '[true,true,true,false]', JSON.stringify(disabled))
  check('empty server: Build is the current tab (aria-current)', (await page.locator('.tabbar button[aria-current=page]').innerText()).includes('Build'))
  check('empty server: the hash moved to #/build, not #/navigate', hashFromUrl(page).startsWith('#/build'), page.url())
  check('empty server: the unlock hint is shown', (await page.getByText('Create your first building to unlock Directions, Map and Record.').count()) === 1)
  check('empty server: no Share panel and no uploader without a building', (await page.locator('.share-panel').count()) === 0 && (await page.getByRole('heading', { name: 'Add a walkthrough' }).count()) === 0)
  await page.locator('.tabbar button', { hasText: 'Map' }).click({ force: true }).catch(() => {})
  await page.waitForTimeout(100)
  check('empty server: a disabled tab cannot be opened', (await page.getByRole('heading', { name: 'New building' }).count()) === 1 && (await page.locator('.map-panel, .node-grid').count()) === 0)
  check('empty server: the owner key check sent no invite list', st.xTokens.at(-1) === '', JSON.stringify(st.xTokens))
  check('empty server: no page errors', !errors.some((e) => /PAGEERROR/.test(e)), errors.join('|').slice(0, 200))
  check('empty server: no secret in console output', noSecrets(errors.join('|')))
  await ctx.close()
}

// ---------- a stored owner key the server refuses is removed, and a valid invite still opens the building ----------
{
  const { ctx, page, errors, st } = await session()
  await page.goto(JOIN(VIEWER))
  await page.waitForSelector('#from', { timeout: 8000 }).catch(() => {})
  await page.evaluate((k) => localStorage.setItem('waypoint.ownerKey', k), STALE_OWNER)
  st.lists = []
  await page.reload()
  await page.waitForSelector('#from', { timeout: 8000 }).catch(() => {})
  await page.getByRole('status').filter({ hasText: KEY_REMOVED }).waitFor({ timeout: 5000 }).catch(() => {})
  check('stale owner + invite: the invite role is used (Directions only)', JSON.stringify(await tabNames(page)) === '["Directions"]', JSON.stringify(await tabNames(page)))
  check('stale owner + invite: no access screen', (await page.locator('.access-screen').count()) === 0)
  check('stale owner + invite: the notice says the key was removed', (await page.locator('.notice', { hasText: KEY_REMOVED }).count()) === 1, await page.locator('.notice').allInnerTexts().then(String).catch(() => ''))
  check('stale owner + invite: the stored owner key is cleared', (await ownerStored(page)) === null, String(await ownerStored(page)))
  check('stale owner + invite: the invite is kept', (await stored(page))?.b1 === VIEWER)
  check('stale owner + invite: exactly two list calls (owner key, then invites only)', st.lists.length === 2, JSON.stringify(st.lists))
  check('stale owner + invite: the retry sends no Authorization, only the invite list', st.lists[1]?.owner === false && st.lists[1]?.xt === VIEWER, JSON.stringify(st.lists[1]))
  check('stale owner + invite: building calls use the invite, never the stale key', st.bAuths.includes(VIEWER) && !st.bAuths.includes(STALE_OWNER), JSON.stringify(st.bAuths))
  check('stale owner + invite: the notice can be dismissed', await (async () => {
    await page.getByRole('button', { name: 'Dismiss notice' }).click()
    return (await page.locator('.notice', { hasText: KEY_REMOVED }).count()) === 0
  })())
  check('stale owner + invite: no stale key in the URL', !page.url().includes(STALE_OWNER))
  check('stale owner + invite: no secret in console output', noSecrets(errors.join('|')), errors.join('|').slice(0, 200))
  check('stale owner + invite: no page errors', !errors.some((e) => /PAGEERROR/.test(e)), errors.join('|').slice(0, 200))
  await ctx.close()
}
{
  const { ctx, page, errors, st } = await session()
  await page.goto(`${BASE}/#/navigate?b=b1`)
  await page.evaluate((k) => localStorage.setItem('waypoint.ownerKey', k), STALE_OWNER)
  st.lists = []
  await page.reload()
  await heading(page).waitFor({ timeout: 5000 }).catch(() => {})
  await page.waitForTimeout(800) // long enough for any retry loop to show itself
  check('stale owner, no invite: the access screen is shown', (await page.locator('.access-screen').count()) === 1)
  check('stale owner, no invite: explained as invite-only (not as a refused owner key)', /invite-only/.test(await heading(page).innerText().catch(() => '')) && !/owner key isn't working/.test(await page.textContent('.access-screen')))
  check('stale owner, no invite: the notice says the key was removed', (await page.locator('.notice', { hasText: KEY_REMOVED }).count()) === 1)
  check('stale owner, no invite: the stored owner key is cleared', (await ownerStored(page)) === null)
  check('stale owner, no invite: no loop, exactly two list calls', st.lists.length === 2, JSON.stringify(st.lists))
  check('stale owner, no invite: the second list call sends no Authorization', st.lists[1]?.owner === false, JSON.stringify(st.lists))
  check('stale owner, no invite: no tab bar on the access screen', (await page.locator('.tabbar').count()) === 0)
  check('stale owner, no invite: no secret in console output', noSecrets(errors.join('|')), errors.join('|').slice(0, 200))
  check('stale owner, no invite: no page errors', !errors.some((e) => /PAGEERROR/.test(e)), errors.join('|').slice(0, 200))
  await ctx.close()
}

// ---------- a refused invite (401 on a building call) shows the access screen ----------
{
  const { ctx, page, errors, st } = await session()
  await page.goto(JOIN(VIEWER))
  await page.waitForSelector('#from', { timeout: 8000 }).catch(() => {})
  st.mapStatus = 401
  await page.reload()
  await heading(page).waitFor({ timeout: 5000 }).catch(() => {})
  check('401: access screen shown when a building call is refused', /isn't working/.test(await heading(page).innerText().catch(() => '')), await heading(page).innerText().catch(() => ''))
  await page.waitForTimeout(150)
  check('401: focus moves to the access heading', (await focusedTag(page)) === 'H2')
  check('401: asks for a new link and offers paste', (await page.getByLabel('Paste an invite link').count()) === 1)
  check('401: no secret in console output', noSecrets(errors.join('|')))
  await ctx.close()
}

// ---------- a stored invite the server no longer accepts ----------
{
  const { ctx, page } = await session()
  await page.goto(`${BASE}/#/navigate?b=b1`)
  await page.evaluate((t) => localStorage.setItem('waypoint.tokens', JSON.stringify({ b1: t })), STALE)
  await page.reload()
  await heading(page).waitFor({ timeout: 5000 }).catch(() => {})
  check('stale invite: explained as not working', /isn't working/.test(await heading(page).innerText().catch(() => '')))
  check('stale invite: no tab bar', (await page.locator('.tabbar').count()) === 0)
  check('stale invite: paste field available', (await page.getByLabel('Paste an invite link').count()) === 1)
  check('stale invite: sign out offered on the screen', (await page.getByRole('button', { name: 'Sign out on this device' }).count()) === 1)
  await ctx.close()
}

// ---------- pasting an invite link ----------
{
  const { ctx, page, errors, st } = await session()
  await page.goto(`${BASE}/#/navigate?b=b1`)
  await page.getByLabel('Paste an invite link').waitFor({ timeout: 5000 })
  await page.getByLabel('Paste an invite link').fill('hello there')
  await page.getByRole('button', { name: 'Open building' }).click()
  check('paste: text that is not a link gets a clear message', /doesn't look like an invite link/.test(await alertText(page)), await alertText(page))

  await page.getByLabel('Paste an invite link').fill(`${ORIGIN}/#/join?b=b1&t=not-a-real-code-0000`)
  await page.getByRole('button', { name: 'Open building' }).click()
  await page.waitForFunction(() => document.querySelector('.access-screen [role=alert]')?.innerText.includes("isn't valid"), null, { timeout: 4000 }).catch(() => {})
  check('paste: a link the server refuses says so', /isn't valid/.test(await alertText(page)), await alertText(page))
  check('paste: a refused link is not saved', (await stored(page)) === null)

  st.rate = 1
  await page.getByLabel('Paste an invite link').fill(JOIN(VIEWER))
  await page.getByRole('button', { name: 'Open building' }).click()
  await page.waitForFunction(() => document.querySelector('.access-screen [role=alert]')?.innerText.includes('Too many tries'), null, { timeout: 4000 }).catch(() => {})
  check('429: says "Too many tries, wait a minute"', /Too many tries, wait a minute/.test(await alertText(page)), await alertText(page))
  check('429: nothing saved', (await stored(page)) === null)

  await page.getByLabel('Paste an invite link').fill(JOIN(VIEWER))
  await page.getByRole('button', { name: 'Open building' }).click()
  await page.waitForSelector('#from', { timeout: 6000 }).catch(() => {})
  check('paste: a valid viewer link opens Directions', hashFromUrl(page) === '#/navigate?b=b1' && JSON.stringify(await tabNames(page)) === '["Directions"]', `${page.url()} ${JSON.stringify(await tabNames(page))}`)
  check('paste: token saved for b1', (await stored(page))?.b1 === VIEWER)
  check('paste: token not in the address bar', !page.url().includes(VIEWER))
  check('paste: no secret in console output', noSecrets(errors.join('|')), errors.join('|').slice(0, 200))
  await ctx.close()
}
{
  const { ctx, page } = await session()
  await page.goto(`${BASE}/#/navigate?b=b1`)
  await page.getByLabel('Paste an invite link').waitFor({ timeout: 5000 })
  await page.getByLabel('Paste an invite link').fill(JOIN(EDITOR))
  await page.getByRole('button', { name: 'Open building' }).click()
  await page.waitForSelector('.tabbar button:nth-child(4)', { timeout: 6000 }).catch(() => {})
  check('paste: an editor link gives the edit tabs', JSON.stringify(await tabNames(page)) === '["Directions","Map","Record","Build"]', JSON.stringify(await tabNames(page)))
  await ctx.close()
}

// ---------- a join link for a building already held replaces the old token ----------
{
  const { ctx, page, st } = await session()
  await page.goto(JOIN(VIEWER))
  await page.waitForSelector('#from', { timeout: 8000 }).catch(() => {})
  await page.goto(JOIN(EDITOR))
  await page.waitForFunction(() => document.querySelectorAll('.tabbar button').length === 4, null, { timeout: 5000 }).catch(() => {})
  check('re-join: token for b1 replaced', (await stored(page))?.b1 === EDITOR, JSON.stringify(await stored(page)))
  check('re-join: only one token sent after replacement', st.xTokens.at(-1) === EDITOR, st.xTokens.at(-1))
  check('re-join: now four tabs', JSON.stringify(await tabNames(page)) === '["Directions","Map","Record","Build"]')
  check('re-join: address bar clean', !page.url().includes(EDITOR) && !page.url().includes('join'))
  await ctx.close()
}

// ---------- a join link missing its code ----------
{
  const { ctx, page } = await session()
  await page.goto(`${BASE}/#/join?b=b1`)
  await heading(page).waitFor({ timeout: 5000 }).catch(() => {})
  check('incomplete join link: explained, not silently ignored', /incomplete/.test(await heading(page).innerText().catch(() => '')))
  check('incomplete join link: address bar cleared of the join path', !page.url().includes('join'), page.url())
  await ctx.close()
}

// ---------- a 403 on an action says so and keeps the user signed in ----------
{
  const { ctx, page, errors, st } = await session()
  await page.goto(JOIN(EDITOR))
  await page.waitForSelector('#from', { timeout: 8000 }).catch(() => {})
  st.mapStatus = 403
  await page.goto(`${BASE}/#/review?b=b1`)
  await page.locator('[role=alert]').first().waitFor({ timeout: 5000 }).catch(() => {})
  check('403: says the invite does not allow that', /Your invite doesn't allow that/.test(await page.locator('[role=alert]').first().innerText().catch(() => '')))
  check('403: no access screen', (await page.locator('.access-screen').count()) === 0)
  check('403: still signed in, tabs kept', (await page.locator('.tabbar button').count()) === 4 && (await stored(page))?.b1 === EDITOR)
  check('403: no secret in console output', noSecrets(errors.join('|')))
  await ctx.close()
}

// ---------- sign out on this device ----------
{
  const { ctx, page, errors } = await session()
  await page.goto(JOIN(VIEWER))
  await page.waitForSelector('#from', { timeout: 8000 }).catch(() => {})
  await page.evaluate((k) => localStorage.setItem('waypoint.ownerKey', k), OWNER)
  await page.reload()
  await page.locator('.tabbar button').nth(3).waitFor({ timeout: 5000 }).catch(() => {})
  check('sign out: owner key in use gives four tabs', (await tabNames(page)).length === 4)
  await page.getByRole('button', { name: 'Sign out', exact: true }).click()
  await page.getByRole('button', { name: 'Sign out on this device' }).click()
  await heading(page).waitFor({ timeout: 5000 }).catch(() => {})
  check('sign out: access screen shown', (await page.locator('.access-screen').count()) === 1)
  check('sign out: invite tokens cleared', (await stored(page)) === null || Object.keys((await stored(page)) || {}).length === 0, JSON.stringify(await stored(page)))
  check('sign out: owner key cleared', (await ownerStored(page)) === null)
  check('sign out: says what was removed', /Signed out on this device/.test(await page.locator('.notice').first().innerText().catch(() => '')))
  await page.waitForTimeout(150)
  check('sign out: focus on the access heading', (await focusedTag(page)) === 'H2')
  check('sign out: no secret in console output', noSecrets(errors.join('|')))
  await ctx.close()
}
{
  const { ctx, page } = await session()
  await page.goto(JOIN(VIEWER))
  await page.waitForSelector('#from', { timeout: 8000 }).catch(() => {})
  await page.getByRole('button', { name: 'Sign out', exact: true }).click()
  await page.getByRole('button', { name: 'Forget this building' }).click()
  await heading(page).waitFor({ timeout: 5000 }).catch(() => {})
  check('forget building: its invite is removed', Object.keys((await stored(page)) || {}).length === 0, JSON.stringify(await stored(page)))
  check('forget building: access screen shown again', (await page.locator('.access-screen').count()) === 1)
  await ctx.close()
}

await browser.close()
const failed = results.filter((r) => !r.ok)
console.log(`\n${results.length - failed.length} passed, ${failed.length} failed of ${results.length}`)
process.exit(failed.length ? 1 : 0)
