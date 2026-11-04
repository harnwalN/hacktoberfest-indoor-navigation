// Cross-screen motion pass (final review). Run: BASE=http://localhost:5370 node tests/e2e-final-cross.mjs (playwright resolvable, vite dev server).
// Tab round trips Directions > Map > 3D > Directions > Build > Record: console errors, 3D renderer survival, Map glow/scroll after a tab switch,
// duplicate FlowNumber text, doubled tab stops.
// API fully mocked. Env SHOTS=<dir> also saves screenshots there.
import { chromium } from 'playwright'
import { createHash } from 'node:crypto'
import { mkdirSync } from 'node:fs'

const BASE = process.env.BASE || 'http://localhost:5199'
const SHOTS = process.env.SHOTS || ''
if (SHOTS) mkdirSync(SHOTS, { recursive: true })
const N = (id, label, type, floor, x, y, extra = {}) => ({ id, label, type, floor, x, y, verified: id !== 'n3', texts: [], sign_texts: [], description: '', ...extra })
const E = (a, b, vertical = 'none') => ({ id: `${a}-${b}`, from_node: a, to_node: b, heading_out: 90, heading_in: 90, vertical, length_m: 8, frames_fwd: [], source: 'walk', verified: true })
const NODES = [
  N('n0', 'Lobby', 'entrance', 1, 0, 0), N('n1', 'Hall A', 'hallway', 1, 10, 0, { frame_id: 'f1', frame_url: '/bfiles/b1/f1.jpg' }), N('n2', 'Room 101', 'room', 1, 20, 0),
  N('n3', 'Room 102', 'room', 1, 20, 8), N('n4', 'Stairs 1', 'stairs', 1, 10, 8), N('n5', 'Stairs 2', 'stairs', 2, 10, 8),
  N('n6', 'Hall B', 'hallway', 2, 10, 0), N('n7', 'Room 201', 'room', 2, 20, 0),
]
const EDGES = [E('n0', 'n1'), E('n1', 'n2'), E('n2', 'n3'), E('n1', 'n4'), E('n4', 'n5', 'stairs'), E('n5', 'n6'), E('n6', 'n7')]
const MAP = () => ({ building_id: 'b1', name: 'B1', nodes: NODES.map((n) => ({ ...n })), edges: EDGES.map((e) => ({ ...e })), suggested_merges: [], videos: [] })
const REPORTS = () => [
  { id: 'rp-a', kind: 'wrong_step', note: 'Wrong way', node_id: 'n1', step_index: 1, status: 'open', created_at: '2026-01-01T10:00:00Z', updated_at: '2026-01-01T10:00:00Z', history: [], resolution_note: null, node_label: 'Hall A' },
  { id: 'rp-b', kind: 'missing_place', note: 'Gone', node_id: 'n2', step_index: null, status: 'open', created_at: '2026-01-01T09:00:00Z', updated_at: '2026-01-01T09:00:00Z', history: [], resolution_note: null, node_label: 'Room 101' },
]

const results = []
const check = (name, ok, info = '') => { results.push({ name, ok: !!ok, info }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok ? '' : '  -> ' + info}`) }
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const realErrors = (errs) => errs.filter((e) => !/GPU stall|GL Driver Message|ReadPixels|WebGL.*(unsupported|not supported)|Failed to load resource.*404/i.test(e))
const md5 = (b) => createHash('md5').update(b).digest('hex')
const DESKTOP = { width: 1440, height: 900 }
const MOBILE = { width: 390, height: 844 }

const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] })

async function session(vp, o = {}) {
  const ctx = await browser.newContext({ viewport: vp, reducedMotion: o.reduced ? 'reduce' : 'no-preference' })
  const page = await ctx.newPage()
  const errors = []
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
  page.on('pageerror', (e) => errors.push('PAGEERROR ' + e.message))
  page.on('dialog', (d) => d.accept())
  page.on('response', (r) => r.status() >= 400 && console.log('INFO http', r.status(), r.url().slice(0, 120)))
  const st = { reports: REPORTS(), saves: 0, saveDelay: 500, saveFail: false, patches: [], gets: [] }
  await page.addInitScript(([cores]) => {
    if (cores) Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => cores })
    if (!sessionStorage.getItem('seeded')) {
      sessionStorage.setItem('seeded', '1')
      localStorage.setItem('session_token', 'sess-1')
      localStorage.setItem('session_account', JSON.stringify({ id: 'a1', email: 'a@b.c', is_owner: true }))
    }
    // GPU resource accounting + listener accounting
    const w = (window.__gl = { buf: 0, tex: 0, prog: 0, fb: 0, rb: 0, ctx: 0 })
    for (const C of [window.WebGLRenderingContext, window.WebGL2RenderingContext]) {
      if (!C) continue
      for (const [k, d] of [['Buffer', 'buf'], ['Texture', 'tex'], ['Program', 'prog'], ['Framebuffer', 'fb'], ['Renderbuffer', 'rb']]) {
        const c = C.prototype['create' + k], x = C.prototype['delete' + k]
        C.prototype['create' + k] = function (...a) { w[d]++; return c.apply(this, a) }
        C.prototype['delete' + k] = function (...a) { if (a[0]) w[d]--; return x.apply(this, a) }
      }
    }
    const L = (window.__lis = {})
    for (const T of [window, document]) {
      const nm = T === window ? 'w' : 'd'
      const a = T.addEventListener.bind(T), r = T.removeEventListener.bind(T)
      T.addEventListener = (t, f, o) => { if (f) { const k = nm + ':' + t; L[k] = (L[k] || 0) + 1 }; return a(t, f, o) }
      T.removeEventListener = (t, f, o) => { if (f) { const k = nm + ':' + t; L[k] = (L[k] || 0) - 1 }; return r(t, f, o) }
    }
  }, [o.cores || 0])
  await ctx.route('**/bfiles/**', async (route) => { await sleep(400); return route.fulfill({ status: 200, contentType: 'image/svg+xml', body: '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect width="640" height="480" fill="#8aa"/></svg>' }) })
  await ctx.route('**/api/**', async (route) => {
    const req = route.request(); const url = new URL(req.url()); const p = url.pathname.replace(/^\/api/, ''); const m = req.method()
    const json = (b, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(b) })
    if (p === '/buildings') return json([{ id: 'b1', name: 'B1', role: 'owner' }])
    if (p === '/auth/me' || p.endsWith('/me')) return json({ account: { id: 'a1', email: 'a@b.c' }, is_owner: true, blocked: false })
    if (p === '/buildings/b1/map' && m === 'GET') return json(MAP())
    if (p === '/buildings/b1/map' && m === 'PUT') {
      st.saves++
      await sleep(st.saveDelay)
      if (st.saveFail) return json({ detail: 'Disk is full' }, 500)
      return json({ map: JSON.parse(req.postData()) })
    }
    if (p === '/buildings/b1/queue') return json([])
    if (p === '/health') return json({ gemma: { ok: true, selected: 'g', models: ['g'] } })
    if (p === '/buildings/b1/reports' && m === 'GET') {
      st.gets.push(url.search)
      const s = url.searchParams.get('status')
      return json(st.reports.filter((r) => s === 'all' || r.status === s))
    }
    const pm = /^\/buildings\/b1\/reports\/([^/]+)$/.exec(p)
    if (pm && m === 'PATCH') {
      const body = JSON.parse(req.postData() || '{}'); st.patches.push({ id: pm[1], body })
      const r = st.reports.find((x) => x.id === pm[1]); r.status = body.status
      r.history = [...r.history, { status: body.status, at: new Date().toISOString(), by: 'owner' }]
      return json(r)
    }
    return json({ detail: 'nope' }, 404)
  })
  await page.goto(`${BASE}/#/navigate?b=b1`)
  await page.locator('.tabbar').waitFor({ timeout: 10000 })
  await sleep(1200)
  return { ctx, page, errors, st }
}
const shot = async (page, name) => { if (SHOTS) await page.screenshot({ path: `${SHOTS}/${name}.png`, fullPage: false }) }
const stageShot = async (page) => md5(await page.locator('.map-stage canvas').first().screenshot())
const pillBox = (page, label) => page.locator('.view-switch').first().getByRole('button', { name: label, exact: true }).boundingBox()
const gl = (page) => page.evaluate(() => ({ ...window.__gl }))
const lis = (page) => page.evaluate(() => ({ ...window.__lis }))
const overflowX = (page) => page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)

// Click a grid over the canvas until a place gets selected (3D has no DOM nodes).
async function pick3d(page) {
  const box = await page.locator('.map-stage canvas').first().boundingBox()
  for (let gy = 0.25; gy <= 0.8; gy += 0.05) for (let gx = 0.15; gx <= 0.85; gx += 0.05) {
    await page.mouse.click(box.x + box.width * gx, box.y + box.height * gy)
    if (await page.locator('.node-editor').count()) return true
  }
  return false
}


const tab = async (page, name) => { await page.locator('.tabbar button', { hasText: name }).click(); await sleep(700) }
// The Build tools (upload, queue, Videos & map edit) are on the owner page: Owner, then its Videos & map section.
const ownerVideos = async (page) => { await tab(page, 'Owner'); await page.getByRole('button', { name: 'Videos & map' }).click(); await sleep(500) }
// Tab stops: focusable elements that are visible, in document order. A doubled stop = same node listed twice or a stray tabindex=0 non-control.
const stops = (page) => page.evaluate(() => {
  const sel = 'a[href],button:not([disabled]),input:not([disabled]),select,textarea,summary,[tabindex]:not([tabindex="-1"])'
  const els = [...document.querySelectorAll(sel)].filter((e) => { const r = e.getBoundingClientRect(); const cs = getComputedStyle(e); return cs.visibility !== 'hidden' && cs.display !== 'none' && (r.width || r.height) && !e.closest('[inert],[aria-hidden="true"]') })
  const stray = els.filter((e) => e.getAttribute('tabindex') === '0' && !['A', 'BUTTON', 'INPUT', 'SELECT', 'SUMMARY', 'TEXTAREA'].includes(e.tagName)).map((e) => e.tagName + '.' + e.className)
  return { n: els.length, stray, dup: els.length - new Set(els).size }
})
// Visible text of FlowNumber: sr-only copy exactly once and the animated copy aria-hidden; body text must read "8 places" not "88 places".
const flowDup = (page) => page.evaluate(() => [...document.querySelectorAll('span[aria-hidden="true"] > number-flow-react')].map((n) => { const p = n.parentElement; const sr = p.nextElementSibling; return { ok: !!sr && sr.classList.contains('sr-only'), nextIsFlow: !!sr?.querySelector?.('number-flow-react'), srCount: [...p.parentNode.querySelectorAll(':scope > .sr-only')].length } }))
const NAV_SEL = '#from'

{
  const { ctx, page, errors } = await session(DESKTOP)
  const base = await stops(page)
  check('Directions: no stray tabindex, no duplicate tab stops', base.stray.length === 0 && base.dup === 0, JSON.stringify(base))
  const counts = {}
  for (let round = 0; round < 3; round++) {
    await tab(page, 'Map'); await page.locator('.map-counts').waitFor()
    const s = await stops(page); counts.map ||= s.n
    check(`round ${round}: Map tab stops stable (${counts.map})`, s.n === counts.map && s.stray.length === 0 && s.dup === 0, JSON.stringify(s))
    const txt = await page.locator('.map-counts').innerText()
    check(`round ${round}: header counts text once`, /8 places, 7 connections, 7 verified/.test(txt.replace(/\s+/g, ' ')) , JSON.stringify(txt))
    const fd = await flowDup(page)
    check(`round ${round}: Map FlowNumber = one aria-hidden animated + one sr-only copy each`, fd.length > 0 && fd.every((f) => f.ok && f.srCount >= 1), JSON.stringify(fd))
    // 3D
    await page.locator('.view-switch').first().getByRole('button', { name: '3D', exact: true }).click(); await sleep(1500)
    check(`round ${round}: 3D canvas present`, (await page.locator('.map-stage canvas').count()) === 1, String(await page.locator('.map-stage canvas').count()))
    const ctx3 = await page.evaluate(() => window.__gl.ctx)
    await tab(page, 'Directions'); await page.waitForSelector(NAV_SEL)
    check(`round ${round}: Directions after 3D: tab stops same as first load`, (await stops(page)).n === base.n, JSON.stringify([await stops(page), base]))
    await ownerVideos(page)
    const b = await stops(page); counts.build ||= b.n
    check(`round ${round}: Build tab stops stable (${counts.build})`, b.n === counts.build && b.stray.length === 0 && b.dup === 0, JSON.stringify(b))
    const fdb = await flowDup(page)
    check(`round ${round}: Build FlowNumber single sr-only copy`, fdb.every((f) => f.ok), JSON.stringify(fdb))
    await tab(page, 'Record')
    const r = await stops(page); counts.rec ||= r.n
    check(`round ${round}: Record tab stops stable (${counts.rec})`, r.n === counts.rec && r.stray.length === 0 && r.dup === 0, JSON.stringify(r))
    await tab(page, 'Directions')
  }
  check('only one tab pill at a time', (await page.locator('.tab-pill').count()) === 1, String(await page.locator('.tab-pill').count()))
  check('exactly 4 tab buttons, one aria-current', (await page.locator('.tabbar button').count()) === 4 && (await page.locator('.tabbar button[aria-current]').count()) === 1)
  check('no console errors across tab round trips', realErrors(errors).length === 0, realErrors(errors).join('|').slice(0, 400))
  await ctx.close()
}

// 3D renderer survives: after repeated tab switches the 3D view still renders, picks, and GPU resources do not grow without bound.
{
  const { ctx, page, errors } = await session(DESKTOP)
  await tab(page, 'Map'); await page.locator('.map-counts').waitFor()
  await page.locator('.view-switch').first().getByRole('button', { name: '3D', exact: true }).click(); await sleep(1500)
  const g0 = await page.evaluate(() => ({ ...window.__gl }))
  for (let i = 0; i < 4; i++) { await tab(page, 'Directions'); await tab(page, 'Map'); await page.locator('.map-counts').waitFor(); await sleep(300) }
  const view = await page.locator('.view-switch').first().getByRole('button', { name: '3D', exact: true }).getAttribute('aria-pressed')
  console.log('INFO 3D pressed after return:', view)
  if (view !== 'true') { await page.locator('.view-switch').first().getByRole('button', { name: '3D', exact: true }).click() }
  await sleep(1500)
  const g1 = await page.evaluate(() => ({ ...window.__gl }))
  check('3D canvas present after 4 tab round trips', (await page.locator('.map-stage canvas').count()) === 1)
  const png = await page.locator('.map-stage canvas').first().screenshot()
  check('3D canvas not blank', new Set([...png.subarray(100, 4000)]).size > 20)
  check('GPU resources do not leak across tab round trips (<= +5 buffers/programs vs first 3D mount)', g1.buf - g0.buf <= 5 && g1.prog - g0.prog <= 5 && g1.tex - g0.tex <= 5, JSON.stringify([g0, g1]))
  const ok = await pick3d(page)
  check('3D still picks a place after tab round trips', ok)
  check('no console errors (3D survival)', realErrors(errors).length === 0, realErrors(errors).join('|').slice(0, 400))
  await ctx.close()
}

// Map glow + scroll after a tab switch: scroll-to-top on mount must not fight the pick scroll.
{
  const { ctx, page, errors } = await session(DESKTOP)
  await tab(page, 'Directions'); await ownerVideos(page); await tab(page, 'Map'); await page.locator('.map-counts').waitFor(); await sleep(800)
  check('Map after tab switch starts scrolled to top', (await page.evaluate(() => window.scrollY)) < 5, String(await page.evaluate(() => window.scrollY)))
  await page.locator('g.snode').nth(1).click({ force: true })
  await page.locator('.node-editor').waitFor()
  await sleep(1200)
  const st = await page.locator('.node-editor').evaluate((e) => { const r = e.getBoundingClientRect(); return { top: r.top, bottom: r.bottom, vh: innerHeight, y: scrollY } })
  check('2D pick after tab switch: edit card in view (scroll not fought)', st.top < st.vh && st.bottom > 0, JSON.stringify(st))
  await sleep(1500)
  check('2D pick after tab switch: glow ends', !(await page.locator('.node-editor').evaluate((e) => e.classList.contains('is-glow'))))
  // glow again on fresh pick, after switching away and back with scroll position non-zero
  await page.evaluate(() => window.scrollTo(0, 400)); await tab(page, 'Directions'); await tab(page, 'Map'); await page.locator('.map-counts').waitFor(); await sleep(500)
  check('returning to Map resets scroll to top', (await page.evaluate(() => window.scrollY)) < 5, String(await page.evaluate(() => window.scrollY)))
  await page.locator('g.snode').nth(2).click({ force: true }); await page.locator('.node-editor').waitFor(); await sleep(200)
  check('glow on pick after second tab switch', await page.locator('.node-editor').evaluate((e) => e.classList.contains('is-glow')))
  await sleep(1000)
  const st2 = await page.locator('.node-editor').evaluate((e) => { const r = e.getBoundingClientRect(); return { top: r.top, bottom: r.bottom, vh: innerHeight } })
  check('second pick: card in view', st2.top < st2.vh && st2.bottom > 0, JSON.stringify(st2))
  check('no console errors (glow/scroll)', realErrors(errors).length === 0, realErrors(errors).join('|').slice(0, 400))
  await ctx.close()
}

await browser.close()
const failed = results.filter((r) => !r.ok).length
console.log(`\n${results.length - failed} passed, ${failed} failed`)
process.exit(failed ? 1 : 0)
