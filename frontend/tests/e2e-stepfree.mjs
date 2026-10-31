// Avoid-stairs behaviour. Run: node tests/e2e-stepfree.mjs  (needs `playwright` resolvable and a vite dev server at BASE,
// default http://localhost:5199). The API is mocked with page.route: the mock answers POST /route the way the backend
// contract says (avoid_stairs, step_free, the two 400 messages), and the frontend's own request/response code is what runs.
import { chromium } from 'playwright'

const BASE = process.env.BASE || 'http://localhost:5199'
const STEP_FREE_MSG = 'No step-free route between these places. The only way uses stairs.'
const NO_PATH_MSG = 'No valid path found between selected locations. Please check graph connections in Map Editor.'
const DESKTOP = { width: 1280, height: 800 }
const MOBILE = { width: 390, height: 844 }

// Map: Room 112 to Room 210 (bh) climbs the stairs (its only way, cost 30). Room 201 is reached by the elevator only, so
// it is step-free either way. Room 210 has no step-free way to anything else. Isolated Room (iso) has no connections.
const N = (id, label, type, floor, extra = {}) => ({ id, label, type, floor, x: 0, y: 0, verified: true, texts: [], sign_texts: [], description: '', ...extra })
const nodes = [
  N('r112', 'Room 112', 'room', 1), N('hp1', 'Hallway point', 'hallway', 1, { verified: false }),
  N('st1', 'Stairs A', 'stairs', 1), N('st2', 'Stairs A', 'stairs', 2),
  N('el1', 'Elevator B', 'elevator', 1), N('el2', 'Elevator B', 'elevator', 2),
  N('r201', 'Room 201', 'room', 2), N('bh', 'Room 210', 'room', 2), N('iso', 'Isolated room', 'room', 1),
]
const E = (a, b, len, vertical = 'none') => ({ id: `${a}-${b}`, from_node: a, to_node: b, heading_out: 90, heading_in: 90, vertical, length_m: len, frames_fwd: [], source: 'walk', verified: true })
const edges = [
  E('r112', 'hp1', 12), E('hp1', 'st1', 10), E('st1', 'st2', 3, 'stairs_up'), E('st2', 'bh', 5),
  E('r112', 'el1', 25), E('el1', 'el2', 3, 'elevator'), E('el2', 'r201', 7),
]
const MAP = { building_id: 'b1', name: 'B1', nodes, edges, suggested_merges: [], videos: [{ id: 'v1', filename: 'a.mp4' }] }

const isStairs = (e) => e.vertical === 'stairs_up' || e.vertical === 'stairs_down'
// Dijkstra over the edges; avoid drops every stairs edge. Returns the node ids or null.
function shortest(from, to, avoid) {
  const dist = { [from]: 0 }
  const prev = {}
  const done = new Set()
  for (;;) {
    let cur = null
    for (const k of Object.keys(dist)) if (!done.has(k) && (cur === null || dist[k] < dist[cur])) cur = k
    if (cur === null) return null
    if (cur === to) break
    done.add(cur)
    for (const e of edges) {
      if (avoid && isStairs(e)) continue
      const other = e.from_node === cur ? e.to_node : e.to_node === cur ? e.from_node : null
      if (!other) continue
      const nd = dist[cur] + e.length_m
      if (dist[other] === undefined || nd < dist[other]) { dist[other] = nd; prev[other] = cur }
    }
  }
  const path = [to]
  while (path[0] !== from) path.unshift(prev[path[0]])
  return path
}
const edgeBetween = (a, b) => edges.find((e) => (e.from_node === a && e.to_node === b) || (e.from_node === b && e.to_node === a))
const byId = Object.fromEntries(nodes.map((n) => [n.id, n]))

// The contract answer for POST /buildings/b1/route.
function routeAnswer(from, to, avoid) {
  if (!byId[from] || !byId[to]) return { status: 404, body: { detail: `Unknown node ${from}` } }
  const plain = shortest(from, to, false)
  if (!plain) return { status: 400, body: { detail: NO_PATH_MSG } }
  const path = avoid ? shortest(from, to, true) : plain
  if (!path) return { status: 400, body: { detail: STEP_FREE_MSG } }
  const stepFree = path.slice(1).every((n, k) => !isStairs(edgeBetween(path[k], n)))
  const steps = path.map((id, k) => {
    const node = byId[id]
    const next = path[k + 1] ? byId[path[k + 1]] : null
    const e = next ? edgeBetween(id, next.id) : null
    let action = 'go_straight'
    let vertical = 'none'
    if (k === 0) action = 'start'
    else if (!next) action = 'destination_reached'
    if (next && e && isStairs(e)) { action = next.floor > node.floor ? 'stairs_up' : 'stairs_down'; vertical = action }
    if (next && e && e.vertical === 'elevator') { action = 'elevator'; vertical = 'elevator' }
    return {
      step_index: k, node_id: id, label: node.label, frame_url: null, action, vertical, floor: node.floor,
      instruction_text: `Walk to ${node.label}`, turn_degrees: 0,
    }
  })
  return { status: 200, body: { steps, node_path: path, step_free: stepFree } }
}

const results = []
const check = (name, ok, info = '') => {
  results.push({ name, ok: !!ok, info })
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok ? '' : '  -> ' + info}`)
}
const realErrors = (errs) => errs.filter((e) => !/Failed to load resource/.test(e))

async function session(vp) {
  const ctx = await browser.newContext({ viewport: vp })
  const page = await ctx.newPage()
  const errors = []
  const st = { routes: [], lists: 0 }
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
  page.on('pageerror', (e) => errors.push('PAGEERROR ' + e.message))
  page.on('dialog', (d) => d.dismiss())
  await page.route('**/api/**', async (route) => {
    const req = route.request()
    const p = new URL(req.url()).pathname.replace(/^\/api/, '')
    const json = (b, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(b) })
    if (p === '/buildings' && req.method() === 'GET') { st.lists += 1; return json([{ id: 'b1', name: 'B1', role: 'owner' }]) }
    if (p === '/buildings/b1/map') return json(MAP)
    if (p === '/buildings/b1/route' && req.method() === 'POST') {
      const body = JSON.parse(req.postData() || '{}')
      st.routes.push({ start: body.start_node_id, target: body.target_node_id, avoid: body.avoid_stairs })
      const a = routeAnswer(body.start_node_id, body.target_node_id, body.avoid_stairs === true)
      return json(a.body, a.status)
    }
    return json({ detail: 'nope' }, 404)
  })
  return { ctx, page, errors, st }
}

const browser = await chromium.launch()
const storedAvoid = (page) => page.evaluate(() => localStorage.getItem('waypoint.avoidStairs'))
const avoidBox = (page) => page.getByLabel('Avoid stairs', { exact: true })
const pickPlace = async (page, id, label) => {
  await page.click(`#${id}`)
  await page.click(`#${id}-list .place-option:has-text("${label}")`)
}
const startTrip = async (page) => page.getByRole('button', { name: /Start directions/ }).click()
const alertText = (page) => page.locator('[role=alert]').first().innerText().catch(() => '')
const stairsButton = (page) => page.getByRole('button', { name: 'Show route with stairs' })
const badgeStairs = (page) => page.locator('.route-badge.is-stairs')
const badgeFree = (page) => page.locator('.route-badge.is-step-free')
const stepCount = (page) => page.locator('.step-count').innerText().catch(() => '')
// The route link lives in the hash: #/navigate?b=..&from=..&to=..[&avoid=1]
const hashParam = (page, key) => new URLSearchParams(new URL(page.url()).hash.split('?')[1] || '').get(key)

for (const [vpName, vp] of [['mobile', MOBILE], ['desktop', DESKTOP]]) {
  const T = (n) => `[${vpName}] ${n}`

  // ---------- the toggle in the planner: label, helper text, tap target, default off ----------
  {
    const { ctx, page, errors, st } = await session(vp)
    await page.goto(`${BASE}/#/navigate?b=b1`)
    await page.waitForSelector('#from', { timeout: 8000 }).catch(() => {})
    const box = avoidBox(page)
    check(T('planner: "Avoid stairs" checkbox has an accessible label'), (await box.count()) === 1)
    check(T('planner: avoid stairs is off by default'), !(await box.isChecked()))
    check(T('planner: helper text "Use elevators and ramps only" shown'), (await page.getByText('Use elevators and ramps only').count()) === 1)
    const h = await page.locator('label.check', { hasText: 'Avoid stairs' }).evaluate((el) => el.getBoundingClientRect().height)
    check(T('planner: toggle tap target is at least 44 px tall'), h >= 44, String(h))
    await pickPlace(page, 'from', 'Room 112')
    await pickPlace(page, 'to', 'Room 210')
    await startTrip(page)
    await page.waitForSelector('.trip-view', { timeout: 6000 }).catch(() => {})
    const sent = st.routes.at(-1)
    check(T('toggle off: route request sends avoid_stairs false explicitly'), sent?.avoid === false, JSON.stringify(sent))
    check(T('stairs route: "This route uses stairs" badge shown'), (await badgeStairs(page).innerText().catch(() => '')) === 'This route uses stairs')
    check(T('stairs route: no "No stairs" badge'), (await badgeFree(page).count()) === 0)
    check(T('no page errors on planning a stairs route'), realErrors(errors).length === 0, realErrors(errors).join('|').slice(0, 200))
    await ctx.close()
  }

  // ---------- persistence across reload, and the link parameter ----------
  {
    const { ctx, page, st } = await session(vp)
    await page.goto(`${BASE}/#/navigate?b=b1`)
    await page.waitForSelector('#from', { timeout: 8000 }).catch(() => {})
    await page.locator('label.check', { hasText: 'Avoid stairs' }).click()
    check(T('persist: checking the box is stored as "1"'), (await storedAvoid(page)) === '1', String(await storedAvoid(page)))
    await page.reload()
    await page.waitForSelector('#from', { timeout: 8000 }).catch(() => {})
    check(T('persist: still checked after reload'), await avoidBox(page).isChecked())
    await pickPlace(page, 'from', 'Room 112')
    await pickPlace(page, 'to', 'Room 201')
    await startTrip(page)
    await page.waitForSelector('.trip-view', { timeout: 6000 }).catch(() => {})
    check(T('persist: the route request sends avoid_stairs true'), st.routes.at(-1)?.avoid === true, JSON.stringify(st.routes.at(-1)))
    check(T('link: starting a trip with avoid on writes avoid=1 into the route link'), hashParam(page, 'avoid') === '1', page.url())
    check(T('link: the route link keeps the building and both places'), hashParam(page, 'b') === 'b1' && hashParam(page, 'from') === 'r112' && hashParam(page, 'to') === 'r201', page.url())
    await page.getByRole('button', { name: 'End directions' }).click()
    await page.waitForSelector('#from', { timeout: 4000 }).catch(() => {})
    check(T('link: ending the trip removes the trip from the link'), hashParam(page, 'from') === null && hashParam(page, 'avoid') === null, page.url())
    await avoidBox(page).uncheck()
    check(T('persist: unchecking is stored as "0"'), (await storedAvoid(page)) === '0', String(await storedAvoid(page)))
    await ctx.close()
  }

  // ---------- a shared link reads the setting from the link on load ----------
  {
    const { ctx, page, st } = await session(vp)
    await page.goto(`${BASE}/#/navigate?b=b1&from=r112&to=r201&avoid=1`)
    await page.waitForSelector('.trip-view', { timeout: 6000 }).catch(() => {})
    check(T('link avoid=1: route requested with avoid_stairs true'), st.routes[0]?.avoid === true, JSON.stringify(st.routes))
    check(T('link avoid=1: the toggle on the trip is checked'), await avoidBox(page).isChecked())
    check(T('link avoid=1: badge "No stairs on this route"'), (await badgeFree(page).innerText().catch(() => '')) === 'No stairs on this route')
    check(T('link avoid=1: no "uses stairs" badge'), (await badgeStairs(page).count()) === 0)
    await ctx.close()
  }
  {
    // a stored "on" does not change a link that says nothing about stairs: the link carries its own setting
    const { ctx, page, st } = await session(vp)
    await page.goto(`${BASE}/#/navigate?b=b1`)
    await page.evaluate(() => localStorage.setItem('waypoint.avoidStairs', '1'))
    await page.goto(`${BASE}/#/navigate?b=b1&from=r112&to=bh`)
    await page.reload()
    await page.waitForSelector('.trip-view', { timeout: 6000 }).catch(() => {})
    check(T('link without avoid: route requested with avoid_stairs false'), st.routes.at(-1)?.avoid === false, JSON.stringify(st.routes))
    check(T('link without avoid: "This route uses stairs" badge'), (await badgeStairs(page).innerText().catch(() => '')) === 'This route uses stairs')
    await ctx.close()
  }

  // ---------- both badges: step-free shows "No stairs"; stairs shows "uses stairs" only when the toggle is off ----------
  {
    const { ctx, page } = await session(vp)
    await page.goto(`${BASE}/#/navigate?b=b1&from=r112&to=r201&avoid=1`)
    await page.waitForSelector('.trip-view', { timeout: 6000 }).catch(() => {})
    check(T('badges: step-free route shows exactly one badge'), (await page.locator('.route-badge').count()) === 1)
    check(T('badges: the step-free route says "No stairs on this route"'), (await badgeFree(page).innerText().catch(() => '')) === 'No stairs on this route')
    await ctx.close()
  }

  // ---------- 400 from the server: its message, and the one-click retry with stairs ----------
  {
    const { ctx, page, errors, st } = await session(vp)
    await page.goto(`${BASE}/#/navigate?b=b1&from=bh&to=r112&avoid=1`)
    await page.locator('[role=alert]').first().waitFor({ timeout: 6000 }).catch(() => {})
    check(T('400 step-free: the server message is shown in the error style'), (await alertText(page)).includes(STEP_FREE_MSG), await alertText(page))
    check(T('400 step-free: the error uses the notice error style'), (await page.locator('.notice.error[role=alert]').count()) === 1)
    check(T('400 step-free: "Show route with stairs" offered'), (await stairsButton(page).count()) === 1)
    check(T('400 step-free: no trip is started'), (await page.locator('.trip-view').count()) === 0)
    await stairsButton(page).click()
    await page.waitForSelector('.trip-view', { timeout: 6000 }).catch(() => {})
    check(T('retry: asks again with avoid_stairs false'), st.routes.at(-1)?.avoid === false && st.routes.at(-1)?.start === 'bh', JSON.stringify(st.routes.at(-1)))
    check(T('retry: the trip with stairs opens'), (await page.locator('.trip-view').count()) === 1)
    check(T('retry: toggle now off'), !(await avoidBox(page).isChecked()))
    check(T('retry: badge "This route uses stairs"'), (await badgeStairs(page).innerText().catch(() => '')) === 'This route uses stairs')
    check(T('retry: stored choice is off'), (await storedAvoid(page)) === '0', String(await storedAvoid(page)))
    check(T('400 step-free + retry: no page errors'), realErrors(errors).length === 0, realErrors(errors).join('|').slice(0, 200))
    await ctx.close()
  }
  {
    // no path at all: the normal message, and no stairs offer
    const { ctx, page } = await session(vp)
    await page.goto(`${BASE}/#/navigate?b=b1&from=r112&to=iso&avoid=1`)
    await page.locator('[role=alert]').first().waitFor({ timeout: 6000 }).catch(() => {})
    check(T('no path: the normal no-path message is shown'), (await alertText(page)).includes(NO_PATH_MSG), await alertText(page))
    check(T('no path: no "Show route with stairs" offer'), (await stairsButton(page).count()) === 0)
    await ctx.close()
  }

  // ---------- changing the toggle during a trip ----------
  {
    const { ctx, page, errors, st } = await session(vp)
    await page.goto(`${BASE}/#/navigate?b=b1&from=r112&to=bh`)
    await page.waitForSelector('.trip-view', { timeout: 6000 }).catch(() => {})
    await page.getByRole('button', { name: 'Next' }).click()
    await page.getByRole('button', { name: 'Next' }).click()
    check(T('mid-trip: two steps forward'), /^3 of/.test(await stepCount(page)), await stepCount(page))
    const here = st.routes.at(-1)
    await avoidBox(page).check()
    await page.waitForFunction(() => document.querySelector('.route-badge.is-step-free'), null, { timeout: 5000 }).catch(() => {})
    const sent = st.routes.at(-1)
    check(T('mid-trip: turning the toggle on re-routes with avoid_stairs true'), sent?.avoid === true, JSON.stringify(sent))
    check(T('mid-trip: re-route starts from the current step place'), sent?.start === 'st2' && here?.start === 'r112', JSON.stringify(sent))
    check(T('mid-trip: the trip is back at its first step, on the new route'), /^1 of 2/.test(await stepCount(page)), await stepCount(page))
    check(T('mid-trip: badge is "No stairs on this route"'), (await badgeFree(page).innerText().catch(() => '')) === 'No stairs on this route')
    check(T('mid-trip: no page errors'), realErrors(errors).length === 0, realErrors(errors).join('|').slice(0, 200))
    await ctx.close()
  }
  {
    // a mid-trip change that the server refuses: the trip stays, the message shows in the trip, the retry works
    const { ctx, page, st } = await session(vp)
    await page.goto(`${BASE}/#/navigate?b=b1&from=bh&to=r112`)
    await page.waitForSelector('.trip-view', { timeout: 6000 }).catch(() => {})
    await avoidBox(page).check()
    await page.locator('[role=alert]').first().waitFor({ timeout: 6000 }).catch(() => {})
    check(T('mid-trip refused: the step-free message shows in the trip'), (await alertText(page)).includes(STEP_FREE_MSG), await alertText(page))
    check(T('mid-trip refused: the trip stays on screen'), (await page.locator('.trip-view').count()) === 1 && /^1 of/.test(await stepCount(page)))
    check(T('mid-trip refused: "Show route with stairs" offered'), (await stairsButton(page).count()) === 1)
    await stairsButton(page).click()
    await page.waitForFunction(() => !document.querySelector('[role=alert]'), null, { timeout: 5000 }).catch(() => {})
    check(T('mid-trip refused: retry turns the toggle off and keeps the trip'), !(await avoidBox(page).isChecked()) && (await page.locator('.trip-view').count()) === 1)
    check(T('mid-trip refused: badge now "This route uses stairs"'), (await badgeStairs(page).innerText().catch(() => '')) === 'This route uses stairs')
    check(T('mid-trip refused: retry request sent with avoid_stairs false'), st.routes.at(-1)?.avoid === false)
    await ctx.close()
  }
}

// ---------- keyboard: Tab reaches the toggle, Space changes it ----------
{
  const { ctx, page, errors } = await session(DESKTOP)
  await page.goto(`${BASE}/#/navigate?b=b1`)
  await page.waitForSelector('#from', { timeout: 8000 }).catch(() => {})
  await page.getByLabel('Show unverified places too').focus()
  await page.keyboard.press('Tab')
  check('keyboard: Tab from "Show unverified places" lands on Avoid stairs', await page.evaluate(() => document.activeElement?.id === 'avoid-stairs'), await page.evaluate(() => document.activeElement?.outerHTML.slice(0, 120)))
  await page.keyboard.press('Space')
  check('keyboard: Space turns Avoid stairs on', await avoidBox(page).isChecked())
  check('keyboard: the change is stored', (await storedAvoid(page)) === '1', String(await storedAvoid(page)))
  check('keyboard: helper text is linked to the checkbox', (await avoidBox(page).getAttribute('aria-describedby')) === 'avoid-stairs-help')
  check('keyboard: no page errors', realErrors(errors).length === 0, realErrors(errors).join('|').slice(0, 200))
  await ctx.close()
}

await browser.close()
const failed = results.filter((r) => !r.ok)
console.log(`\n${results.length - failed.length} passed, ${failed.length} failed of ${results.length}`)
process.exit(failed.length ? 1 : 0)
