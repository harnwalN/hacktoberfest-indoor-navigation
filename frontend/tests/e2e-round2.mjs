// Round-2 behaviour tests. Run: node tests/e2e-round2.mjs (playwright resolvable, vite dev server at BASE, default :5199). API fully mocked.
import { chromium } from 'playwright'

const BASE = process.env.BASE || 'http://localhost:5199'
const N = (id, label, type, floor, x, y, extra = {}) => ({ id, label, type, floor, x, y, verified: true, texts: [], sign_texts: [], description: '', ...extra })
const nodes = [
  N('r112', 'Room 112', 'room', 1, 0, 0), N('hp1', 'Hallway point', 'hallway', 1, 20, 0, { verified: false }),
  N('hp2', 'Hallway point 2', 'hallway', 1, 30, 0, { verified: false }), N('st1', 'Stairs A', 'stairs', 1, 90, 0),
  N('st2', 'Stairs A', 'stairs', 2, 90, 0), N('r201', 'Room 201', 'room', 2, 100, 0), N('ne', 'No Exit', 'exit', 1, 60, 0),
]
const E = (a, b, len, vertical = 'none') => ({ id: `${a}-${b}`, from_node: a, to_node: b, heading_out: 90, heading_in: 90, vertical, length_m: len, frames_fwd: [], source: 'walk', verified: true })
const edges = [E('r112', 'hp1', 12), E('hp1', 'hp2', 11), E('hp2', 'st1', 23), E('st1', 'st2', null, 'stairs_up'), E('st2', 'r201', 7)]
const FULL = { building_id: 'b1', name: 'B1', nodes, edges, suggested_merges: [], videos: [] }
const S = (i, node_id, label, action, floor, extra = {}) => ({ step_index: i, node_id, label, frame_url: null, action, vertical: 'none', instruction_text: `Do ${label}`, floor, turn_degrees: 0, ...extra })
const STEPS = [
  S(0, 'r112', 'Room 112', 'start', 1), S(1, 'hp1', 'Hallway point', 'go_straight', 1), S(2, 'hp2', 'Hallway point 2', 'go_straight', 1),
  S(3, 'st1', 'Stairs A', 'stairs_up', 1, { vertical: 'stairs_up' }), S(4, 'st2', 'Stairs A', 'go_straight', 2), S(5, 'r201', 'Room 201', 'destination_reached', 2),
]
// long route: 14 distinct rooms (each non-hallway so none merge)
const longNodes = Array.from({ length: 14 }, (_, k) => N(`l${k}`, `Lab ${k}`, 'room', 1, k * 10, 0))
const longEdges = longNodes.slice(1).map((n, k) => E(`l${k}`, n.id, 8))
const LONG = { ...FULL, nodes: longNodes, edges: longEdges }
const LONG_STEPS = longNodes.map((n, k) => S(k, n.id, n.label, k === 0 ? 'start' : k === 13 ? 'destination_reached' : 'turn_left', 1, { instruction_text: `Walk past ${n.label} and keep going along the corridor for a good while`, turn_degrees: -90 }))
// many places for the picker scroll test
const manyNodes = Array.from({ length: 60 }, (_, k) => N(`m${k}`, `Place ${k}`, 'room', 1 + (k % 3), k, 0))
const MANY = { ...FULL, nodes: manyNodes, edges: [] }

const results = []
const check = (name, ok, info = '') => { results.push({ name, ok: !!ok, info }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok ? '' : '  -> ' + info}`) }

async function mockApi(page, o = {}) {
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url()); const p = url.pathname.replace(/^\/api/, ''); const m = route.request().method()
    const json = (b, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(b) })
    if (p === '/buildings') return o.buildingsFail ? json({ detail: 'Server exploded' }, 500) : json([{ id: 'b1', name: 'B1' }])
    if (p === '/buildings/b1/map' && m === 'GET') return json(o.map || FULL)
    if (p === '/buildings/b1/route') return json(o.route || { steps: STEPS, node_path: STEPS.map((s) => s.node_id) })
    return json({ detail: 'nope' }, 404)
  })
}
const browser = await chromium.launch()
async function session(vp, o = {}, ctxOpts = {}) {
  const ctx = await browser.newContext({ viewport: vp, ...ctxOpts })
  const page = await ctx.newPage()
  const errors = []
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
  page.on('pageerror', (e) => errors.push('PAGEERROR ' + e.message))
  await mockApi(page, o)
  return { ctx, page, errors }
}
const MOBILE = { width: 390, height: 844 }
const DESKTOP = { width: 1280, height: 800 }

for (const [vpName, vp] of [['mobile', MOBILE], ['desktop', DESKTOP]]) {
  const T = (n) => `[${vpName}] ${n}`
  // arrow keys clamp to merged steps (6 raw -> 4 merged)
  {
    const { ctx, page } = await session(vp)
    await page.goto(`${BASE}/#/navigate?b=b1&from=r112&to=r201`)
    await page.waitForSelector('.trip-view')
    const cnt = async () => (await page.textContent('.step-count')).replace(/\s+/g, ' ')
    check(T('merged total is 4'), /1 of 4/.test(await cnt()), await cnt())
    await page.keyboard.press('ArrowLeft')
    check(T('ArrowLeft at first step stays 1 of 4'), /1 of 4/.test(await cnt()), await cnt())
    for (let k = 0; k < 10; k++) await page.keyboard.press('ArrowRight')
    check(T('10x ArrowRight clamps at 4 of 4 (not raw 6)'), /4 of 4/.test(await cnt()), await cnt())
    await page.keyboard.press('ArrowLeft')
    check(T('ArrowLeft from clamped end goes to 3 of 4 immediately'), /3 of 4/.test(await cnt()), await cnt())
    await ctx.close()
  }
  // node with no floor does not crash
  {
    const nf = { ...FULL, nodes: FULL.nodes.map((n) => { const c = { ...n }; if (n.id === 'hp1' || n.id === 'r201') delete c.floor; return c }) }
    nf.nodes.push({ id: 'nofloor', label: 'Mystery', type: 'room', verified: true })
    const { ctx, page, errors } = await session(vp, { map: nf })
    await page.goto(`${BASE}/#/navigate?b=b1`)
    await page.waitForSelector('#from')
    await page.click('#from')
    const txt = await page.textContent('#from-list')
    check(T('no-floor node: picker renders and lists Mystery'), /Mystery/.test(txt), txt.slice(0, 120))
    await page.goto(`${BASE}/#/navigate?b=b1&from=r112&to=r201`); await page.reload()
    await page.waitForSelector('.trip-view', { timeout: 5000 }).catch(() => {})
    check(T('no-floor node: route renders'), !!(await page.$('.trip-view')) && !(await page.$('text=Directions hit a problem')), errors.join('|'))
    check(T('no-floor node: no console/page errors'), errors.filter((e) => !/Failed to load resource/.test(e)).length === 0, errors.join('|'))
    await ctx.close()
  }
  // ErrorBoundary: Navigator (route step null => throws during render in useMemo)
  {
    const { ctx, page, errors } = await session(vp, { route: { steps: [null, null], node_path: [] } })
    await page.goto(`${BASE}/#/navigate?b=b1&from=r112&to=r201`)
    await page.waitForSelector('text=Directions hit a problem', { timeout: 5000 }).catch(() => {})
    check(T('Navigator boundary fallback shown'), !!(await page.$('text=Directions hit a problem')), (await page.textContent('body')).slice(0, 200))
    check(T('Navigator fallback has Try again and role=alert'), !!(await page.$('[role=alert] >> text=Try again')))
    check(T('app shell (tab bar) survives Navigator crash'), await page.isVisible('.tabbar'))
    // tab away and back works
    await page.click('.tabbar >> text=Map'); await page.waitForSelector('.map-panel, .node-grid', { timeout: 5000 }).catch(() => {})
    check(T('other tab still works after Navigator crash'), !!(await page.$('.map-panel, .node-grid')))
    await page.click('.tabbar >> text=Directions'); await page.waitForSelector('text=Directions hit a problem, .planner', { timeout: 3000 }).catch(() => {})
    await ctx.close()
  }
  {
    // Try again resets boundary; with a malformed label object rendering throws again => fallback persists, no blank page
    const bad = { ...FULL, nodes: FULL.nodes.map((n, k) => (k === 0 ? { ...n, label: { oops: 1 } } : n)) }
    const { ctx, page } = await session(vp, { map: bad })
    await page.goto(`${BASE}/#/navigate?b=b1`)
    await page.click("#from").catch(() => {})
    await page.waitForSelector('text=Directions hit a problem', { timeout: 5000 }).catch(() => {})
    const shown = !!(await page.$('text=Directions hit a problem'))
    check(T('Navigator boundary catches malformed label object'), shown, (await page.textContent('body')).slice(0, 200))
    if (shown) {
      await page.click('[role=alert] >> text=Try again'); await page.waitForTimeout(300)
      check(T('Navigator Try again click leaves page non-blank (fallback again)'), (await page.textContent('body')).length > 20 && await page.isVisible('.tabbar'))
    }
    await ctx.close()
  }
  // ErrorBoundary: MapReview
  {
    const bad = { ...FULL, nodes: FULL.nodes.map((n, k) => (k === 0 ? { ...n, label: { oops: 1 } } : n)) }
    const { ctx, page } = await session(vp, { map: bad })
    await page.goto(`${BASE}/#/review?b=b1`)
    await page.waitForSelector('text=The map hit a problem', { timeout: 5000 }).catch(() => {})
    const shown = !!(await page.$('text=The map hit a problem'))
    check(T('MapReview boundary fallback shown'), shown, (await page.textContent('body')).slice(0, 200))
    check(T('MapReview fallback has Try again'), !!(await page.$('[role=alert] >> text=Try again')))
    check(T('app shell survives MapReview crash'), await page.isVisible('.tabbar'))
    await ctx.close()
  }
  {
    // Boundary recovery: first map malformed, then valid after Try again (re-mount of child)
    let calls = 0
    const { ctx, page } = await session(vp)
    await page.route('**/api/buildings/b1/map', (route) => {
      calls++
      const body = calls === 1 ? { ...FULL, nodes: FULL.nodes.map((n, k) => (k === 0 ? { ...n, label: { oops: 1 } } : n)) } : FULL
      route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) })
    })
    await page.goto(`${BASE}/#/review?b=b1`)
    await page.waitForSelector('text=The map hit a problem', { timeout: 5000 }).catch(() => {})
    await page.click('[role=alert] >> text=Try again').catch(() => {})
    await page.waitForSelector('.map-panel', { timeout: 5000 }).catch(() => {})
    check(T('MapReview Try again recovers when data becomes valid'), !!(await page.$('.map-panel')), `calls=${calls}`)
    await ctx.close()
  }
  // error banner keyboard dismiss
  for (const key of ['Enter', 'Space']) {
    const { ctx, page } = await session(vp, { buildingsFail: true })
    await page.goto(`${BASE}/#/navigate`)
    await page.waitForSelector('.banner.error')
    check(T(`banner has a real <button> dismiss control (${key})`), !!(await page.$('button.banner.error, .banner.error button')))
    let reached = false
    for (let k = 0; k < 8 && !reached; k++) { await page.keyboard.press('Tab'); reached = await page.evaluate(() => !!document.activeElement?.closest('.banner')) }
    check(T(`banner reachable by Tab (${key})`), reached)
    await page.keyboard.press(key === 'Space' ? ' ' : 'Enter')
    check(T(`banner dismissed with ${key}`), !(await page.$('.banner.error')))
    await ctx.close()
  }
  // picker list scrolls
  {
    const { ctx, page } = await session(vp, { map: MANY })
    await page.goto(`${BASE}/#/navigate?b=b1`)
    await page.waitForSelector('#from')
    await page.click('#from')
    await page.waitForSelector('#from-list .place-option')
    const m = await page.$eval('.place-picker-list, #from-list', (e) => ({ sh: e.scrollHeight, ch: e.clientHeight, ov: getComputedStyle(e).overflowY }))
    check(T('picker list is a bounded scroll container'), m.sh > m.ch && /auto|scroll/.test(m.ov), JSON.stringify(m))
    const before = await page.$eval('.place-picker-list', (e) => e.scrollTop)
    await page.hover('#from-list .place-option >> nth=0')
    await page.mouse.wheel(0, 600); await page.waitForTimeout(200)
    const after = await page.$eval('.place-picker-list', (e) => e.scrollTop)
    check(T('wheel scrolls picker list'), after > before, `${before}->${after}`)
    // keyboard to the last option scrolls it into view
    await page.fill('#from', '')
    await page.press('#from', 'ArrowDown')
    for (let k = 0; k < 59; k++) await page.keyboard.press('ArrowDown')
    const vis = await page.evaluate(() => { const a = document.activeElement; const l = document.querySelector('.place-picker-list'); if (!a || !l) return false; const r = a.getBoundingClientRect(), lr = l.getBoundingClientRect(); return r.top >= lr.top - 1 && r.bottom <= lr.bottom + 1 && /Place/.test(a.innerText) })
    check(T('keyboard to last option keeps it visible in the list'), vis)
    await ctx.close()
  }
}

// Long route: Next/Back visible and clickable, not covered by tab bar, 390x844
{
  const { ctx, page, errors } = await session(MOBILE, { map: LONG, route: { steps: LONG_STEPS, node_path: LONG_STEPS.map((s) => s.node_id) } })
  await page.goto(`${BASE}/#/navigate?b=b1&from=l0&to=l13`)
  await page.waitForSelector('.trip-view')
  const total = Number((/of (\d+)/.exec(await page.textContent('.step-count')) || [])[1])
  check('long route has >= 10 steps', total >= 10, String(total))
  const probe = async (sel) => page.evaluate((s) => {
    const el = document.querySelector(s); if (!el) return { missing: true }
    const r = el.getBoundingClientRect(); const cx = r.left + r.width / 2, cy = r.top + r.height / 2
    const hit = document.elementFromPoint(cx, cy); const tab = document.querySelector('.tabbar').getBoundingClientRect()
    return { top: r.top, bottom: r.bottom, h: r.height, w: r.width, inView: r.top >= 0 && r.bottom <= innerHeight, hitOk: !!hit && (hit === el || el.contains(hit)), aboveTab: r.bottom <= tab.top + 0.5, tabTop: tab.top, hidden: el.disabled }
  }, sel)
  for (const scrollTo of ['top', 'middle', 'bottom']) {
    await page.evaluate((w) => { const h = document.documentElement.scrollHeight; window.scrollTo(0, w === 'top' ? 0 : w === 'middle' ? h / 2 : h) }, scrollTo)
    await page.waitForTimeout(150)
    const nx = await probe('.step-nav .next')
    check(`long route [scroll ${scrollTo}] Next in viewport, hit-testable, above tab bar`, nx.inView && nx.hitOk && nx.aboveTab, JSON.stringify(nx))
    check(`long route [scroll ${scrollTo}] Next tap target >= 40px high`, nx.h >= 40, JSON.stringify(nx))
  }
  // advance with real clicks to the end, probing Back each time
  let backOk = true, info = ''
  for (let k = 1; k < total; k++) {
    await page.click('.step-nav .next', { timeout: 3000 })
    if (k === 5) { const bk = await probe('.step-nav .back, .step-nav button:first-child'); backOk = bk.inView && bk.hitOk && bk.aboveTab; info = JSON.stringify(bk) }
  }
  check('long route: Next clicked through all steps with real pointer clicks', new RegExp(`${total} of ${total}`).test(await page.textContent('.step-count')))
  check('long route: Back visible/clickable/above tab bar mid-route', backOk, info)
  await page.click('.step-nav button:first-child'); 
  check('long route: Back click goes back', new RegExp(`${total - 1} of ${total}`).test(await page.textContent('.step-count')))
  const overflowX = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)
  check('long route: no horizontal overflow at 390', !overflowX)
  check('long route: no console errors', errors.filter((e) => !/Failed to load resource/.test(e)).length === 0, errors.join('|'))
  await ctx.close()
}

// Console errors across all screens, light + dark, incl 3D toggle
for (const scheme of ['light', 'dark']) {
  for (const [vpName, vp] of [['mobile', MOBILE], ['desktop', DESKTOP]]) {
    const { ctx, page, errors } = await session(vp, {}, { colorScheme: scheme, permissions: [] })
    const T = (n) => `[${scheme}/${vpName}] ${n}`
    const bg = async () => page.evaluate(() => getComputedStyle(document.body).backgroundColor)
    await page.goto(`${BASE}/#/build?b=b1`); await page.waitForSelector('.tabbar'); await page.waitForTimeout(300)
    const bgBuild = await bg()
    for (const tab of ['Map', 'Directions', 'Record', 'Build']) {
      await page.click(`.tabbar >> text=${tab}`); await page.waitForTimeout(400)
    }
    await page.goto(`${BASE}/#/navigate?b=b1&from=r112&to=r201`); await page.reload(); await page.waitForSelector('.trip-view')
    await page.click('.step-nav .next'); await page.click('.step-nav .next')
    await page.goto(`${BASE}/#/review?b=b1`); await page.reload(); await page.waitForSelector('.map-panel')
    await page.click('.node-card >> nth=0').catch(() => {})
    await page.click('button:text-is("3D")'); await page.waitForTimeout(800)
    // flip scheme live while in 3D: scene must rebuild without errors
    await page.emulateMedia({ colorScheme: scheme === 'dark' ? 'light' : 'dark' }); await page.waitForTimeout(600)
    await page.emulateMedia({ colorScheme: scheme }); await page.waitForTimeout(600)
    await page.click('button:text-is("2D")'); await page.waitForTimeout(300)
    await page.click('button:text-is("3D")'); await page.waitForTimeout(500)
    check(T('3D canvas present after toggles'), !!(await page.$('.map3d canvas')))
    const filtered = errors.filter((e) => !/Failed to load resource/.test(e))
    check(T('no console errors across screens + 3D toggle + live scheme flip'), filtered.length === 0, filtered.join('|').slice(0, 400))
    check(T('body background differs by scheme sanity'), scheme === 'dark' ? bgBuild !== 'rgb(255, 255, 255)' : true, bgBuild)
    await ctx.close()
  }
}

await browser.close()
const f = results.filter((r) => !r.ok)
console.log(`\n${results.length - f.length} passed, ${f.length} failed of ${results.length}`)
process.exit(f.length ? 1 : 0)
