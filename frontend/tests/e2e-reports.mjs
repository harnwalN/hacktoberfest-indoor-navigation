// "Something's wrong" reports and the entrance poster. Run: node tests/e2e-reports.mjs  (needs `playwright` resolvable and
// a vite dev server at BASE, default http://localhost:5199). The API is mocked with page.route: the mock answers the
// reports, access and invite calls the way the backend contract says, so the frontend's own request and render code is
// what runs. Covers: the report panel (focus, body, every message, double submit), the editors' list (filter, show on
// map, resolve with and without a note, reopen, the open count), what viewers never see, and the poster (QR, text, print).
import { chromium } from 'playwright'

const BASE = process.env.BASE || 'http://localhost:5199'
const ORIGIN = new URL(BASE).origin
const DESKTOP = { width: 1280, height: 800 }
const MOBILE = { width: 390, height: 844 }
const PANEL = 'Report a problem with this step'
const SENT_OK = "Thanks, the map's editors will check this"
const TOO_MANY = "You've sent a lot of reports, try again later"
const FULL = 'Reports are full for this building'

const N = (id, label, type, floor, extra = {}) => ({
  id, label, type, floor, x: 0, y: 0, verified: true, texts: [], sign_texts: [], description: '', ...extra,
})
const nodes = [N('r112', 'Room 112', 'room', 1), N('bh', 'Room 210', 'room', 1), N('hp1', 'Hallway point', 'hallway', 1, { verified: false })]
const edges = [
  { id: 'e1', from_node: 'r112', to_node: 'bh', heading_out: 90, heading_in: 90, vertical: 'none', length_m: 12, frames_fwd: [], source: 'walk', verified: true },
]
const MAP = { building_id: 'b1', name: 'Hall B', nodes, edges, suggested_merges: [], videos: [{ id: 'v1', filename: 'a.mp4' }] }
// Two steps, both on floor 1: the start (raw step 0) and the destination (raw step 1).
const STEPS = [
  { step_index: 0, node_id: 'r112', label: 'Room 112', frame_url: null, action: 'start', vertical: 'none', floor: 1, instruction_text: 'Start at Room 112', turn_degrees: 0 },
  { step_index: 1, node_id: 'bh', label: 'Room 210', frame_url: null, action: 'destination_reached', vertical: 'none', floor: 1, instruction_text: 'Room 210 is on the left', turn_degrees: 0 },
]
const NODE_LABEL = Object.fromEntries(nodes.map((n) => [n.id, n.label]))
const LINK_TOKEN = 'poster-tok-0123456789abcdef'

const results = []
const check = (name, ok, info = '') => {
  results.push({ name, ok: !!ok, info })
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok ? '' : '  -> ' + info}`)
}
const realErrors = (errs) => errs.filter((e) => !/Failed to load resource/.test(e))
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

// The stored reports. The list returns them newest first; node_label is computed from the map, as the server does.
const seedReports = () => [
  { id: 'rp-a', kind: 'wrong_direction', note: 'Turned the wrong way at the lift', node_id: 'bh', step_index: 1, from_node: 'r112', to_node: 'bh',
    created_at: '2026-10-30T09:00:00Z', reporter: 'i-viewer', status: 'open', resolution_note: null, updated_at: '2026-10-30T09:00:00Z',
    history: [{ status: 'open', at: '2026-10-30T09:00:00Z', by: 'i-viewer', resolution_note: null }] },
  { id: 'rp-b', kind: 'place_missing', note: '', node_id: 'gone', step_index: null, from_node: null, to_node: null,
    created_at: '2026-10-29T09:00:00Z', reporter: 'owner', status: 'open', resolution_note: null, updated_at: '2026-10-29T09:00:00Z',
    history: [{ status: 'open', at: '2026-10-29T09:00:00Z', by: 'owner', resolution_note: null }] },
  { id: 'rp-c', kind: 'blocked', note: 'Door is chained', node_id: 'r112', step_index: 0, from_node: 'r112', to_node: 'bh',
    created_at: '2026-10-28T09:00:00Z', reporter: 'i-abcdef', status: 'resolved', resolution_note: 'Door cleared', updated_at: '2026-10-28T12:00:00Z',
    history: [
      { status: 'open', at: '2026-10-28T09:00:00Z', by: 'i-abcdef', resolution_note: null },
      { status: 'resolved', at: '2026-10-28T12:00:00Z', by: 'owner', resolution_note: 'Door cleared' },
    ] },
]

// One mock session. role is what GET /buildings says the signed-in user is.
async function session(vp, role, opts = {}) {
  const ctx = await browser.newContext({ viewport: vp })
  const page = await ctx.newPage()
  const errors = []
  const st = {
    role,
    reportMode: 'ok', // ok | 429 | 507 | 422 | slow
    reportBodies: [],
    reports: seedReports(),
    gets: [], // the query of each GET /reports
    patches: [], // { id, body }
    accessCalls: 0,
    urls: [],
    invites: [],
    created: 0,
    ...opts,
  }
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
  page.on('pageerror', (e) => errors.push('PAGEERROR ' + e.message))
  page.on('dialog', (d) => d.dismiss())
  page.on('request', (req) => {
    if (!req.url().startsWith(ORIGIN) && !req.url().startsWith('data:') && !req.url().startsWith('blob:')) st.external = (st.external || []).concat(req.url())
  })
  await page.route('**/api/**', async (route) => {
    const req = route.request()
    const url = new URL(req.url())
    const p = url.pathname.replace(/^\/api/, '')
    const m = req.method()
    st.urls.push(req.url())
    const json = (b, status = 200, headers = {}) => route.fulfill({ status, contentType: 'application/json', headers, body: JSON.stringify(b) })

    if (p === '/buildings' && m === 'GET') return json([{ id: 'b1', name: 'Hall B', role }])
    if (p === '/buildings/b1/map') return json(MAP)
    if (p === '/buildings/b1/queue') return json([])
    if (p === '/buildings/b1/route' && m === 'POST') {
      if (st.reportMode === 'slow') await sleep(700)
      return json({ steps: STEPS, node_path: ['r112', 'bh'], step_free: true })
    }
    if (p === '/buildings/b1/access' && m === 'GET') {
      st.accessCalls += 1
      if (role === 'viewer') return json({ role })
      return json({ role, open_reports: st.reports.filter((r) => r.status === 'open').length })
    }
    if (p === '/buildings/b1/reports' && m === 'POST') {
      const body = JSON.parse(req.postData() || '{}')
      st.reportBodies.push(body)
      if (st.reportMode === '429') return json({ detail: 'Too many reports from you for this building' }, 429, { 'Retry-After': '1200' })
      if (st.reportMode === '507') return json({ detail: 'This building already holds its maximum' }, 507)
      if (st.reportMode === '422') return json({ detail: 'note: String should have at most 500 characters' }, 422)
      if (st.reportMode === 'slow') await sleep(300)
      st.seq = (st.seq || 0) + 1
      const id = `rp-new-${st.seq}`
      st.reports.unshift({ id, kind: body.kind, note: body.note || '', node_id: body.node_id || null, step_index: body.step_index ?? null,
        from_node: body.from_node || null, to_node: body.to_node || null, created_at: new Date().toISOString(), reporter: 'i-viewer',
        status: 'open', resolution_note: null, updated_at: new Date().toISOString(), history: [] })
      return json({ id, status: 'open' })
    }
    if (p === '/buildings/b1/reports' && m === 'GET') {
      const status = url.searchParams.get('status') || 'open'
      const limit = Number(url.searchParams.get('limit') || 100)
      st.gets.push(url.search)
      const list = st.reports
        .filter((r) => status === 'all' || r.status === status)
        .sort((a, b) => (a.created_at < b.created_at ? 1 : -1))
        .slice(0, limit)
        .map((r) => ({ ...r, node_label: r.node_id && NODE_LABEL[r.node_id] ? NODE_LABEL[r.node_id] : null }))
      return json(list)
    }
    const pm = /^\/buildings\/b1\/reports\/([^/]+)$/.exec(p)
    if (pm && m === 'PATCH') {
      const body = JSON.parse(req.postData() || '{}')
      st.patches.push({ id: pm[1], body })
      const r = st.reports.find((x) => x.id === pm[1])
      if (!r) return json({ detail: 'Unknown report' }, 404)
      const at = new Date().toISOString()
      r.status = body.status
      r.resolution_note = body.status === 'resolved' ? body.resolution_note ?? null : null
      r.updated_at = at
      r.history = [...r.history, { status: body.status, at, by: 'owner', resolution_note: r.resolution_note }]
      return json({ ...r, node_label: r.node_id && NODE_LABEL[r.node_id] ? NODE_LABEL[r.node_id] : null })
    }
    if (p === '/buildings/b1/invites' && m === 'GET') return json(st.invites.map(({ token, ...rest }) => rest))
    if (p === '/buildings/b1/invites' && m === 'POST') {
      const body = JSON.parse(req.postData() || '{}')
      st.created += 1
      const inv = { id: `i-new-${st.created}`, role: body.role, label: body.label || '', created_at: new Date().toISOString(), revoked: false }
      st.invites.push(inv)
      return json({ ...inv, token: LINK_TOKEN, join_path: `#/join?b=b1&t=${LINK_TOKEN}` })
    }
    return json({ detail: 'nope' }, 404)
  })
  return { ctx, page, errors, st }
}

const browser = await chromium.launch()
const tripLinkHash = '#/navigate?b=b1&from=r112&to=bh'
const reportButton = (page) => page.getByRole('button', { name: /Something.s wrong\?/ })
const panel = (page) => page.locator(`form[aria-label="${PANEL}"]`)
const sendButton = (page) => panel(page).getByRole('button', { name: 'Send' })
const focusInPanel = (page) => page.evaluate((label) => !!document.activeElement?.closest(`form[aria-label="${label}"]`), PANEL)
const focusIsToggle = (page) => page.evaluate(() => document.activeElement?.classList.contains('report-toggle') === true)
const thanks = (page) => page.locator('.report-thanks').innerText().catch(() => '')
const alertIn = (scope) => scope.locator('[role=alert]').first().innerText().catch(() => '')
const startTrip = async (page) => {
  await page.waitForSelector('.trip-view', { timeout: 6000 }).catch(() => {})
}
const sentBodies = (st) => st.reportBodies
const openBadge = (page) => page.locator('.count-badge').first().innerText().catch(() => '')

// ---------- directions: the report panel ----------
for (const [vpName, vp] of [['desktop', DESKTOP], ['mobile', MOBILE]]) {
  const T = (n) => `[${vpName}] ${n}`
  {
    const { ctx, page, errors, st } = await session(vp, 'viewer')
    await page.goto(`${BASE}/${tripLinkHash}`)
    await startTrip(page)
    const btn = reportButton(page)
    check(T('directions: "Something\'s wrong?" is a quiet button on the step'), (await btn.count()) === 1)
    const h = await btn.evaluate((el) => el.getBoundingClientRect().height)
    check(T('directions: the report button is at least 44 px tall'), h >= 44, String(h))
    check(T('directions: panel closed at first'), (await panel(page).count()) === 0)
    await btn.click()
    check(T('panel: opens inline with the choices'), (await panel(page).count()) === 1)
    check(T('panel: the toggle reports aria-expanded'), (await btn.getAttribute('aria-expanded')) === 'true')
    check(T('panel: focus moves into the panel when it opens'), await focusInPanel(page), await page.evaluate(() => document.activeElement?.outerHTML.slice(0, 120)))
    const labels = await panel(page).getByRole('radio').count()
    check(T('panel: six plain-language choices'), labels === 6, String(labels))
    const choiceText = await panel(page).locator('.report-choice').allInnerTexts()
    check(T('panel: choices are in plain words'), ['Directions are wrong', 'A place is missing', 'A sign has changed', "Photo doesn't match", 'The way is blocked', 'Something else'].every((t) => choiceText.some((c) => c.trim() === t)), choiceText.join('|'))
    check(T('panel: the note has a character counter'), (await panel(page).locator('small', { hasText: '0 of 500 characters' }).count()) === 1)
    const cancel = panel(page).getByRole('button', { name: 'Cancel' })
    check(T('panel: Cancel and Send are present'), (await cancel.count()) === 1 && (await sendButton(page).count()) === 1)
    const rh = await panel(page).getByRole('radio').first().evaluate((el) => el.closest('label').getBoundingClientRect().height)
    check(T('panel: each choice is at least 44 px tall'), rh >= 44, String(rh))

    // Escape closes the panel, returns focus to the button, and leaves the trip on screen.
    await page.keyboard.press('Escape')
    check(T('panel: Escape closes the panel'), (await panel(page).count()) === 0)
    check(T('panel: Escape returns focus to the button'), await focusIsToggle(page))
    check(T('panel: Escape does not end the trip'), (await page.locator('.trip-view').count()) === 1)
    check(T('panel: no request sent by closing'), sentBodies(st).length === 0)

    // Send with nothing chosen: a message, no request.
    await btn.click()
    await sendButton(page).click()
    check(T('send without a choice: asks for one, sends nothing'), (await alertIn(panel(page))).includes('Choose what is wrong first.') && sentBodies(st).length === 0)

    // Full send: the body carries the step's place, the trip ends, and the raw step index.
    await panel(page).getByRole('radio', { name: 'The way is blocked' }).check()
    await panel(page).locator('textarea').fill('Gate is locked')
    check(T('panel: the counter follows the note'), (await panel(page).locator('small', { hasText: '14 of 500 characters' }).count()) === 1)
    await sendButton(page).click()
    await page.waitForFunction(() => !document.querySelector('form[aria-label="Report a problem with this step"]'), null, { timeout: 4000 }).catch(() => {})
    const sent = sentBodies(st).at(-1)
    // The trip opens on step 0, so the place is the start (r112) and the raw index is 0.
    check(T('send: body has kind, note, node_id, from_node, to_node, step_index'),
      JSON.stringify(sent) === JSON.stringify({ kind: 'blocked', note: 'Gate is locked', node_id: 'r112', from_node: 'r112', to_node: 'bh', step_index: 0 }), JSON.stringify(sent))
    check(T('send: success message shown'), (await thanks(page)) === SENT_OK, await thanks(page))
    check(T('send: panel closes on success'), (await panel(page).count()) === 0)
    check(T('send: focus returns to the button on success'), await focusIsToggle(page))
    check(T('send: no page errors'), realErrors(errors).length === 0, realErrors(errors).join('|').slice(0, 200))
    await ctx.close()
  }
}

// ---------- directions: body rules, messages, double submit (desktop) ----------
{
  const { ctx, page, errors, st } = await session(DESKTOP, 'viewer')
  await page.goto(`${BASE}/${tripLinkHash}`)
  await startTrip(page)
  // First step: step_index 0 is sent (it is a real step, not "missing"), and an empty note is left out.
  await reportButton(page).click()
  await panel(page).getByRole('radio', { name: 'Directions are wrong' }).check()
  await sendButton(page).click()
  await page.waitForFunction(() => !document.querySelector('form[aria-label="Report a problem with this step"]'), null, { timeout: 4000 }).catch(() => {})
  const first = sentBodies(st).at(-1)
  check('body: step 0 sends step_index 0', first?.step_index === 0, JSON.stringify(first))
  check('body: an empty note is not sent', first && !('note' in first), JSON.stringify(first))
  check('body: start place is node_id, trip ends are from_node and to_node', first?.node_id === 'r112' && first?.from_node === 'r112' && first?.to_node === 'bh', JSON.stringify(first))

  // Server refusals, each with its own message. The panel stays open so the reporter can see it.
  const refusal = async (mode, expect, name) => {
    st.reportMode = mode
    await reportButton(page).click()
    await panel(page).getByRole('radio', { name: 'Something else' }).check()
    await sendButton(page).click()
    await page.locator('form[aria-label="Report a problem with this step"] [role=alert]').first().waitFor({ timeout: 4000 }).catch(() => {})
    const msg = await alertIn(panel(page))
    check(`refused ${name}: message is "${expect}"`, msg.includes(expect), msg)
    check(`refused ${name}: panel stays open`, (await panel(page).count()) === 1)
    check(`refused ${name}: no success message`, (await thanks(page)) === '')
    await page.keyboard.press('Escape')
  }
  await refusal('429', TOO_MANY, '429')
  await refusal('507', FULL, '507')
  await refusal('422', 'note: String should have at most 500 characters', '422')
  st.reportMode = 'ok'

  // Double submit: a slow server, two clicks in the same tick. Only one request leaves.
  st.reportMode = 'slow'
  await reportButton(page).click()
  await panel(page).getByRole('radio', { name: 'Something else' }).check()
  const before = sentBodies(st).length
  await page.evaluate(() => {
    const b = [...document.querySelectorAll('form[aria-label="Report a problem with this step"] button')].find((x) => x.textContent.trim() === 'Send')
    b.click()
    b.click()
  })
  await sleep(80)
  const sending = panel(page).getByRole('button', { name: 'Sending…' })
  check('double submit: the Send button says Sending and is marked busy', (await sending.count()) === 1 && (await sending.getAttribute('aria-disabled')) === 'true')
  await page.waitForFunction(() => !document.querySelector('form[aria-label="Report a problem with this step"]'), null, { timeout: 4000 }).catch(() => {})
  await sleep(400)
  check('double submit: one request for two clicks', sentBodies(st).length - before === 1, String(sentBodies(st).length - before))
  st.reportMode = 'ok'
  check('directions: no page errors across reports', realErrors(errors).length === 0, realErrors(errors).join('|').slice(0, 200))
  await ctx.close()
}

// ---------- directions: a report on the second step, and the Escape key does not step the route ----------
{
  const { ctx, page, st } = await session(DESKTOP, 'viewer')
  await page.goto(`${BASE}/${tripLinkHash}`)
  await startTrip(page)
  await page.getByRole('button', { name: 'Next' }).click()
  await reportButton(page).click()
  await page.keyboard.press('ArrowLeft')
  check('panel: arrow keys inside the panel do not change the step', /^2 of 2/.test(await page.locator('.step-count').innerText()), await page.locator('.step-count').innerText())
  await page.keyboard.press('Escape')
  await reportButton(page).click()
  await panel(page).getByRole('radio', { name: 'Directions are wrong' }).check()
  await sendButton(page).click()
  await page.waitForFunction(() => !document.querySelector('form[aria-label="Report a problem with this step"]'), null, { timeout: 4000 }).catch(() => {})
  check('second step: step_index 1 and place bh', JSON.stringify(st.reportBodies.at(-1)) === JSON.stringify({ kind: 'wrong_direction', node_id: 'bh', from_node: 'r112', to_node: 'bh', step_index: 1 }), JSON.stringify(st.reportBodies.at(-1)))
  await ctx.close()
}

// ---------- viewers: the report button yes, the list and the badge never ----------
{
  const { ctx, page, st } = await session(DESKTOP, 'viewer')
  await page.goto(`${BASE}/${tripLinkHash}`)
  await startTrip(page)
  check('viewer: can send a report from Directions', (await reportButton(page).count()) === 1)
  check('viewer: no Map tab', (await page.getByRole('button', { name: /^Map/ }).count()) === 0)
  await page.goto(`${BASE}/#/review?b=b1`)
  await sleep(600)
  check('viewer: no Reports section on the review route', (await page.getByRole('heading', { name: 'Reports', exact: true }).count()) === 0)
  check('viewer: no list request and no access count request', st.gets.length === 0 && st.accessCalls === 0, JSON.stringify({ gets: st.gets, access: st.accessCalls }))
  check('viewer: no open-count badge anywhere', (await page.locator('.count-badge, .tab-badge').count()) === 0)
  await ctx.close()
}

// ---------- editors: the list, its filter, show on map, resolve, reopen, and the badge ----------
for (const [vpName, vp] of [['desktop', DESKTOP], ['mobile', MOBILE]]) {
  const T = (n) => `[${vpName}] ${n}`
  const { ctx, page, errors, st } = await session(vp, 'editor')
  await page.goto(`${BASE}/#/review?b=b1`)
  await page.getByRole('heading', { name: 'Reports', exact: true }).waitFor({ timeout: 8000 }).catch(() => {})
  const list = () => page.getByRole('list', { name: 'Open reports' })
  check(T('list: the Reports section is on the Map tab'), (await page.getByRole('heading', { name: /^Reports/ }).count()) === 1)
  check(T('list: starts on Open, newest first, with the open items only'), (await list().locator('.report-item').count()) === 2 && st.gets.at(-1)?.includes('status=open'), st.gets.at(-1))
  check(T('list: the request asks for the newest 100'), st.gets.at(-1)?.includes('limit=100'), st.gets.at(-1))
  check(T('list: the first item is the newest report'), (await list().locator('.report-item').first().innerText()).includes('Directions are wrong'))
  check(T('list: the note is shown'), (await list().getByText('Turned the wrong way at the lift').count()) === 1)
  check(T('list: the kind is in plain words'), (await list().getByText('A place is missing').count()) === 1)
  check(T('list: a place no longer on the map is named so'), (await list().getByText('Place no longer on the map').count()) === 1)
  check(T('list: the reporter shows only as Owner or Invite xxxx'), (await list().getByText('Invite i-vi', { exact: false }).count()) >= 1 && (await list().getByText('Owner', { exact: false }).count()) >= 1)
  check(T('list: the date is shown'), (await list().locator('.report-meta time').count()) === 2)

  // The open count: badge in the section header and on the Map tab, from /access.
  check(T('badge: section header shows 2 open'), (await openBadge(page)) === '2 open', await openBadge(page))
  const tabBadge = await page.locator('.tab-badge').first().innerText().catch(() => '')
  check(T('badge: Map tab shows 2'), tabBadge === '2', tabBadge)
  check(T('badge: Map tab name carries the count for screen readers'), (await page.getByRole('button', { name: /^Map.*2 open reports/ }).count()) === 1)

  // Filter.
  await page.getByRole('button', { name: 'Resolved', exact: true }).click()
  await sleep(250)
  check(T('filter: Resolved asks for status=resolved'), st.gets.at(-1)?.includes('status=resolved'), st.gets.at(-1))
  check(T('filter: Resolved shows the fixed report with its fix note'), (await page.locator('.report-resolution').innerText()) === 'Fix: Door cleared', await page.locator('.report-resolution').innerText().catch(() => ''))
  check(T('filter: a resolved report offers Reopen and not Mark fixed'), (await page.getByRole('button', { name: 'Reopen' }).count()) === 1 && (await page.getByRole('button', { name: 'Mark fixed' }).count()) === 0)
  await page.getByRole('button', { name: 'All', exact: true }).click()
  await sleep(250)
  check(T('filter: All asks for status=all and lists all three'), st.gets.at(-1)?.includes('status=all') && (await page.getByRole('list', { name: 'All reports' }).locator('.report-item').count()) === 3, st.gets.at(-1))
  check(T('filter: the history is available'), (await page.locator('details.report-history').count()) === 3)
  await page.getByRole('button', { name: 'Open', exact: true }).click()
  await sleep(250)
  check(T('filter: back to Open'), (await list().locator('.report-item').count()) === 2)

  // Show on map: selects the node in the existing review UI, if it is on the map.
  const showButtons = await list().getByRole('button', { name: 'Show on map' }).count()
  check(T('show on map: offered only for a place that exists (one of two)'), showButtons === 1, String(showButtons))
  await list().getByRole('button', { name: 'Show on map' }).click()
  await sleep(200)
  check(T('show on map: the node editor opens on that place'), (await page.locator('.node-editor h3').innerText().catch(() => '')).includes('Room 210'))

  // Mark fixed with a resolution note.
  await list().locator('.report-item').first().getByRole('button', { name: 'Mark fixed' }).click()
  await sleep(100)
  check(T('mark fixed: an inline field opens'), (await page.getByRole('group', { name: 'Mark as fixed' }).count()) === 1)
  check(T('mark fixed: focus moves to the resolution note'), await page.evaluate(() => document.activeElement?.tagName === 'TEXTAREA'))
  await page.getByRole('group', { name: 'Mark as fixed' }).locator('textarea').fill('Sign replaced on Tuesday')
  await page.getByRole('button', { name: 'Confirm fixed' }).click()
  await sleep(400)
  const patch = st.patches.at(-1)
  check(T('resolve: PATCH sends status resolved with the note'),
    patch?.id === 'rp-a' && JSON.stringify(patch.body) === JSON.stringify({ status: 'resolved', resolution_note: 'Sign replaced on Tuesday' }), JSON.stringify(patch))
  check(T('resolve: the report leaves the open list'), (await list().locator('.report-item').count()) === 1)
  check(T('resolve: the header badge updates to 1 open'), (await openBadge(page)) === '1 open', await openBadge(page))
  check(T('resolve: the Map tab badge updates to 1'), (await page.locator('.tab-badge').first().innerText().catch(() => '')) === '1')
  check(T('resolve: the status line says it is done'), (await page.getByRole('status').filter({ hasText: 'Marked as fixed.' }).count()) === 1)

  // Mark fixed without a note: no resolution_note key at all. Cancel first, to check focus goes back.
  await list().locator('.report-item').first().getByRole('button', { name: 'Mark fixed' }).click()
  await page.getByRole('group', { name: 'Mark as fixed' }).getByRole('button', { name: 'Cancel' }).click()
  await sleep(100)
  check(T('cancel: no request sent'), st.patches.length === 1)
  check(T('cancel: focus returns to the Mark fixed button'), await page.evaluate(() => document.activeElement?.id === 'report-fix-rp-b' || document.activeElement?.textContent?.includes('Mark fixed')))
  await list().locator('.report-item').first().getByRole('button', { name: 'Mark fixed' }).click()
  await page.getByRole('button', { name: 'Confirm fixed' }).click()
  await sleep(400)
  const bare = st.patches.at(-1)
  check(T('resolve without a note: body has no resolution_note'), bare?.id === 'rp-b' && JSON.stringify(bare.body) === JSON.stringify({ status: 'resolved' }), JSON.stringify(bare))
  check(T('resolve without a note: badge now 0 open'), (await openBadge(page)) === '0 open', await openBadge(page))
  check(T('resolve without a note: an empty open list says so'), (await page.locator('.reports-empty').innerText().catch(() => '')).includes('No open reports'))

  // Reopen a resolved report.
  await page.getByRole('button', { name: 'Resolved', exact: true }).click()
  await sleep(250)
  await page.getByRole('button', { name: 'Reopen' }).first().click()
  await sleep(400)
  const reopen = st.patches.at(-1)
  check(T('reopen: PATCH sends status open and no note'), JSON.stringify(reopen?.body) === JSON.stringify({ status: 'open' }), JSON.stringify(reopen))
  check(T('reopen: the badge counts the reopened report again'), (await openBadge(page)) === '1 open', await openBadge(page))

  check(T('editors: no page errors'), realErrors(errors).length === 0, realErrors(errors).join('|').slice(0, 200))
  await ctx.close()
}

// ---------- editors: a failed resolve shows the server's message, and the list is still usable ----------
{
  const { ctx, page, st } = await session(DESKTOP, 'editor')
  await page.goto(`${BASE}/#/review?b=b1`)
  await page.getByRole('heading', { name: /^Reports/ }).waitFor({ timeout: 8000 }).catch(() => {})
  await page.getByRole('list', { name: 'Open reports' }).locator('.report-item').first().getByRole('button', { name: 'Mark fixed' }).click()
  // The report is gone on the server after the list loaded: the resolve is refused and the reason is shown.
  st.reports = st.reports.filter((r) => r.id !== 'rp-a')
  await page.getByRole('button', { name: 'Confirm fixed' }).click()
  await sleep(400)
  const msg = await page.locator('.reports [role=alert]').first().innerText().catch(() => '')
  check('editors: a failed resolve shows the server message', msg.includes('Unknown report'), msg)
  check('editors: the field stays open so nothing typed is lost', (await page.getByRole('group', { name: 'Mark as fixed' }).count()) === 1)
  await ctx.close()
}

// ---------- owner: the poster ----------
{
  const { ctx, page, errors, st } = await session(DESKTOP, 'owner')
  const external = []
  await page.goto(`${BASE}/#/build?b=b1`)
  await page.getByRole('heading', { name: /Share this building/ }).waitFor({ timeout: 8000 }).catch(() => {})
  check('poster: no "Make a poster" before a viewer link exists', (await page.getByRole('button', { name: 'Make a poster' }).count()) === 0)
  await page.getByLabel('Label (optional)').fill('Entrance poster')
  check('poster: the label field suggests "Entrance poster" for a viewer link', (await page.locator('.share-form').innerText()).includes('Entrance poster'))
  await page.getByRole('button', { name: 'Create link' }).click()
  await page.getByRole('region', { name: 'New invite link' }).waitFor({ timeout: 5000 }).catch(() => {})
  const made = await page.getByRole('region', { name: 'New invite link' }).getByLabel('Invite link').inputValue()
  check('poster: the link is the full join link', made.endsWith(`#/join?b=b1&t=${LINK_TOKEN}`), made)
  const offer = page.getByRole('button', { name: 'Make a poster' })
  check('poster: "Make a poster" offered after a viewer link', (await offer.count()) === 1)
  const offerText = await page.getByRole('region', { name: 'New invite link' }).innerText()
  check('poster: says anyone who scans gets viewer access', /scans? .*viewer access/i.test(offerText), offerText.slice(0, 300))
  check('poster: says revoking the invite disables the poster', /revok\w* .*poster|poster .*revok/i.test(offerText), offerText.slice(0, 300))

  // Open the poster. Network requests must stay on this site (no external QR service).
  const externalBefore = (st.external || []).length
  await offer.click()
  const dialog = page.getByRole('dialog', { name: 'Entrance poster for Hall B' })
  await dialog.waitFor({ timeout: 5000 }).catch(() => {})
  check('poster: opens as a dialog', (await dialog.count()) === 1)
  check('poster: focus moves into the poster', await page.evaluate(() => !!document.activeElement?.closest('.poster-root')))
  await page.waitForSelector('.poster-qr', { timeout: 5000 }).catch(() => {})
  check('poster: a QR code is drawn', (await page.locator('svg.poster-qr path').count()) === 1)
  const modules = Number(await page.locator('svg.poster-qr').getAttribute('data-modules'))
  check('poster: the code has a real module grid (version 1 or more)', modules >= 21 && modules % 4 === 1, String(modules))
  const d = await page.locator('svg.poster-qr path').getAttribute('d')
  check('poster: the code path has dark modules', (d || '').split('M').length > 10, String((d || '').length))
  check('poster: the building name is the heading', (await page.locator('h1.poster-name').innerText()) === 'Hall B')
  check('poster: "Scan for directions inside Hall B"', (await page.locator('.poster-cta').innerText()) === 'Scan for directions inside Hall B')
  const printed = await page.locator('.poster-link').innerText()
  check('poster: the link written below matches the link copied', printed === made, printed)
  check('poster: the app underneath is inert while the poster is open', await page.evaluate(() => document.getElementById('root').inert === true))
  check('poster: no request leaves this site while it renders', (st.external || []).length === externalBefore, JSON.stringify((st.external || []).slice(externalBefore)))

  // Decode the code, if this browser can. Otherwise the module grid is checked above and the decode is reported as skipped.
  const decoded = await page.evaluate(async () => {
    if (!('BarcodeDetector' in window)) return null
    const svg = document.querySelector('svg.poster-qr')
    const clone = svg.cloneNode(true)
    clone.setAttribute('width', '600')
    clone.setAttribute('height', '600')
    const img = new Image()
    img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(new XMLSerializer().serializeToString(clone))}`
    await img.decode()
    const c = document.createElement('canvas')
    c.width = 600
    c.height = 600
    c.getContext('2d').drawImage(img, 0, 0, 600, 600)
    const found = await new window.BarcodeDetector({ formats: ['qr_code'] }).detect(c)
    return found[0]?.rawValue ?? ''
  })
  if (decoded === null) console.log('INFO  poster: BarcodeDetector not available in this browser, decode skipped')
  else check('poster: the code decodes to the join link', decoded === made, String(decoded).slice(0, 120))

  // Print: only the poster prints. The app is hidden and the poster is laid out on its own.
  await page.emulateMedia({ media: 'print' })
  await sleep(150)
  const printState = await page.evaluate(() => ({
    root: getComputedStyle(document.getElementById('root')).display,
    poster: getComputedStyle(document.querySelector('.poster-root')).display,
    bar: getComputedStyle(document.querySelector('.poster-bar')).display,
    sheet: getComputedStyle(document.querySelector('.poster-sheet')).display,
    position: getComputedStyle(document.querySelector('.poster-root')).position,
  }))
  check('print: the app is hidden', printState.root === 'none', JSON.stringify(printState))
  check('print: the poster is shown', printState.poster !== 'none' && printState.sheet !== 'none', JSON.stringify(printState))
  check('print: the controls are hidden', printState.bar === 'none', JSON.stringify(printState))
  check('print: the poster sits in normal flow, not as an overlay', printState.position === 'static', printState.position)
  await page.emulateMedia({ media: 'screen' })

  // Close: Escape closes, the app is live again, and focus goes back to "Make a poster".
  await page.keyboard.press('Escape')
  await sleep(150)
  check('poster: Escape closes the poster', (await dialog.count()) === 0)
  check('poster: the app is live again after closing', await page.evaluate(() => document.getElementById('root').inert === false))
  check('poster: focus returns to "Make a poster"', await page.evaluate(() => document.activeElement?.textContent?.includes('Make a poster')))
  await offer.click()
  await dialog.waitFor({ timeout: 4000 }).catch(() => {})
  await page.getByRole('button', { name: 'Close' }).click()
  await sleep(150)
  check('poster: Close button closes it', (await dialog.count()) === 0)

  // Revoking the invite switches the poster off: the link stops working, and the made region goes with "Done".
  await page.getByRole('button', { name: 'Done' }).click()
  check('poster: "Done" removes the offer', (await page.getByRole('button', { name: 'Make a poster' }).count()) === 0)
  check('poster: no page errors', realErrors(errors).length === 0, realErrors(errors).join('|').slice(0, 200))
  await ctx.close()
}

await browser.close()
const failed = results.filter((r) => !r.ok)
console.log(`\n${results.length - failed.length} passed, ${failed.length} failed of ${results.length}`)
process.exit(failed.length ? 1 : 0)
