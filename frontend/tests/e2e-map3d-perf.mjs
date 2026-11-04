// Map 3D render budget + floor tag placement. Run: BASE=http://localhost:5440 node tests/e2e-map3d-perf.mjs
// (playwright resolvable from frontend/, vite dev server at BASE). API fully mocked. Env SHOTS=<dir> also saves screenshots.
// Frames are counted two ways: every requestAnimationFrame callback that runs (works on any build) and, in dev builds,
// the Map3D render counter (window.__map3dRenders). Floor tag overlap uses the dev-only window.__map3dStats().
import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'

const BASE = process.env.BASE || 'http://localhost:5199'
const SHOTS = process.env.SHOTS || ''
if (SHOTS) mkdirSync(SHOTS, { recursive: true })
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const N = (id, label, type, floor, x, y, extra = {}) => ({ id, label, type, floor, x, y, verified: true, texts: [], sign_texts: [], description: '', ...extra })
const E = (a, b, vertical = 'none') => ({ id: `${a}-${b}`, from_node: a, to_node: b, heading_out: 90, heading_in: 90, vertical, length_m: 8, frames_fwd: [], source: 'walk', verified: true })
// Three floors, places spread on a grid so a floor tag can land on or away from a marker.
const NODES = [
  N('a0', 'Lobby', 'entrance', 1, 0, 0), N('a1', 'Hall A', 'hallway', 1, 10, 0), N('a2', 'Room 101', 'room', 1, 20, 0),
  N('a3', 'Room 102', 'room', 1, 20, 8), N('a4', 'Stairs 1', 'stairs', 1, 10, 8),
  N('b0', 'Hall B', 'hallway', 2, 10, 0), N('b1', 'Room 201', 'room', 2, 20, 0), N('b2', 'Room 202', 'room', 2, 20, 8),
  N('b3', 'Stairs 2', 'stairs', 2, 10, 8),
  N('c0', 'Hall C', 'hallway', 3, 10, 0), N('c1', 'Room 301', 'room', 3, 20, 0), N('c2', 'Room 302', 'room', 3, 20, 8),
  N('c3', 'Stairs 3', 'stairs', 3, 10, 8),
]
const EDGES = [
  E('a0', 'a1'), E('a1', 'a2'), E('a2', 'a3'), E('a1', 'a4'), E('a4', 'b3', 'stairs'), E('a1', 'b0'),
  E('b0', 'b1'), E('b1', 'b2'), E('b0', 'b3'), E('b3', 'c3', 'stairs'), E('b0', 'c0'), E('c0', 'c1'), E('c1', 'c2'),
]
const MAP = () => ({ building_id: 'b1', name: 'B1', nodes: NODES.map((n) => ({ ...n })), edges: EDGES.map((e) => ({ ...e })), suggested_merges: [], videos: [] })

const results = []
const check = (name, ok, info = '') => { results.push({ name, ok: !!ok }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok ? '' : '  -> ' + info}`) }
const info = (s) => console.log('INFO ' + s)
const realErrors = (errs) => errs.filter((e) => !/GPU stall|GL Driver Message|ReadPixels|WebGL.*(unsupported|not supported)|Failed to load resource.*404|status of 500/i.test(e))
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
  await page.addInitScript(() => {
    if (!sessionStorage.getItem('seeded')) {
      sessionStorage.setItem('seeded', '1')
      localStorage.setItem('session_token', 'sess-1')
      localStorage.setItem('session_account', JSON.stringify({ id: 'a1', email: 'a@b.c' }))
    }
    // Count every rAF callback that actually runs (frames the page paints or prepares), independent of Map3D internals.
    window.__raf = 0
    const raf = window.requestAnimationFrame.bind(window)
    window.requestAnimationFrame = (cb) => raf((t) => { window.__raf++; return cb(t) })
  })
  await ctx.route('**/api/**', async (route) => {
    const req = route.request(); const url = new URL(req.url()); const p = url.pathname.replace(/^\/api/, ''); const m = req.method()
    const json = (b, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(b) })
    if (p === '/buildings') return json([{ id: 'b1', name: 'B1', role: 'owner' }])
    if (p === '/auth/me' || p.endsWith('/me')) return json({ account: { id: 'a1', email: 'a@b.c' }, is_owner: false, blocked: false })
    if (p === '/buildings/b1/map' && m === 'GET') return json(MAP())
    if (p === '/buildings/b1/queue') return json([])
    if (p === '/buildings/b1/reports') return json([])
    return json({ detail: 'nope' }, 404)
  })
  await page.goto(`${BASE}/#/review?b=b1`)
  await page.locator('.map-counts').waitFor({ timeout: 10000 })
  await sleep(800)
  await page.locator('.view-switch').first().getByRole('button', { name: '3D', exact: true }).click()
  await page.locator('.map-stage canvas').first().waitFor()
  return { ctx, page, errors }
}

// Frames (rAF callbacks run) and Map3D renders (dev counter, may be absent on a prod build) over a window.
async function count(page, ms) {
  const a = await page.evaluate(() => ({ raf: window.__raf, r: window.__map3dRenders || 0 }))
  await sleep(ms)
  const b = await page.evaluate(() => ({ raf: window.__raf, r: window.__map3dRenders || 0 }))
  return { raf: b.raf - a.raf, renders: b.r - a.r }
}
const perSec = (c, ms) => ({ raf: +(c.raf * 1000 / ms).toFixed(1), renders: +(c.renders * 1000 / ms).toFixed(1) })
const stats = (page) => page.evaluate(() => (window.__map3dStats ? window.__map3dStats() : null))
const canvasBox = (page) => page.locator('.map-stage canvas').first().boundingBox()

// ---------------- Desktop: idle, orbit, hover, select, resize ----------------
{
  const { ctx, page, errors } = await session(DESKTOP)
  const intro = await count(page, 1600)
  info(`intro window 1.6s: frames=${intro.raf} renders=${intro.renders}`)
  check('intro plays (frames run during the first 1.6s)', intro.raf > 10, JSON.stringify(intro))
  await sleep(8000) // intro + auto-rotate window
  const idle = await count(page, 2000)
  info(`idle after settle 2s: ${JSON.stringify(idle)} per-sec=${JSON.stringify(perSec(idle, 2000))}`)
  check('idle after settle: zero frames (raf) and zero renders', idle.raf === 0 && idle.renders === 0, JSON.stringify(idle))

  await page.evaluate(() => window.scrollTo(0, 0))
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/map3d-desktop-default.png` })
  const b = await canvasBox(page)
  const cx = b.x + b.width / 2, cy = b.y + b.height / 2
  await page.mouse.move(cx, cy)
  const hoverEmpty = await count(page, 800)
  check('mouse over empty canvas: no frames', hoverEmpty.raf === 0, JSON.stringify(hoverEmpty))

  await page.mouse.down()
  const orbitStart = count(page, 1000)
  await page.mouse.move(cx + 200, cy + 40, { steps: 20 })
  await page.mouse.move(cx + 320, cy + 60, { steps: 20 })
  const orbit = await orbitStart
  await page.mouse.up()
  info(`orbit drag ~1s: ${JSON.stringify(orbit)} per-sec=${JSON.stringify(perSec(orbit, 1000))}`)
  check('orbit drag renders frames', orbit.raf > 5, JSON.stringify(orbit))
  // damping inertia keeps the camera gliding after release; trace it in 500 ms windows, then the settled window must be zero
  const trace = []
  for (let i = 0; i < 8; i++) trace.push((await count(page, 500)).raf)
  info(`orbit release inertia, frames per 500ms: ${trace.join(' ')}`)
  check('orbit release: inertia decays (last 500ms window below first)', trace.at(-1) < trace[0], trace.join(' '))
  const afterOrbit = await count(page, 2000)
  check('after orbit release: settles to zero frames', afterOrbit.raf === 0 && afterOrbit.renders === 0, JSON.stringify(afterOrbit))

  await page.mouse.move(cx, cy)
  const wheelWin = count(page, 900)
  await page.mouse.wheel(0, -300)
  const wheel = await wheelWin
  check('wheel zoom renders frames, then settles', wheel.raf > 0, JSON.stringify(wheel))
  check('wheel zoom: settled after 0.9s window ends (next 1.5s idle)', (await count(page, 1500)).raf === 0)

  // hover ring: a mouse sweep over the canvas renders while the ring fades; a still pointer on a place pulses
  await page.mouse.move(cx - 150, cy - 80, { steps: 10 })
  const sweep = await count(page, 300)
  info(`pointer sweep 0.3s: ${JSON.stringify(sweep)}`)

  // select a place by pick: the pulse runs for a bounded time, then holds
  const picked = await (async () => {
    for (let gy = 0.25; gy <= 0.8; gy += 0.05) for (let gx = 0.15; gx <= 0.85; gx += 0.05) {
      await page.mouse.click(b.x + b.width * gx, b.y + b.height * gy)
      if (await page.locator('.node-editor').count()) return true
    }
    return false
  })()
  check('3D node click selects a place', picked)
  const pulse = await count(page, 1500)
  info(`selected pulse window 1.5s: ${JSON.stringify(pulse)}`)
  await sleep(3500)
  const selIdle = await count(page, 2000)
  check('selected place: frames stop after the pulse window', selIdle.raf === 0 && selIdle.renders === 0, JSON.stringify(selIdle))

  // resize re-fits and renders once, then stops
  await page.setViewportSize({ width: 1100, height: 800 })
  const rz = await count(page, 600)
  check('resize renders frames', rz.raf > 0, JSON.stringify(rz))
  const rzIdle = await count(page, 1500)
  check('after resize: settles to zero frames', rzIdle.raf === 0, JSON.stringify(rzIdle))

  // floor tags vs place markers at the default angle (dev stats, multi-floor map)
  const st = await stats(page)
  if (st) {
    info(`floor tag stats at default angle: ${JSON.stringify(st)}`)
    check('floor tags sit away from place markers (no marker inside a tag box)', st.tagHits === 0, JSON.stringify(st))
    check('every floor tag is shown (none hidden behind another tag)', st.tags.length === 3 && st.tags.every((t) => t.visible), JSON.stringify(st.tags))
  } else info('no __map3dStats (prod build): tag overlap not measured')

  check('no console errors (desktop)', realErrors(errors).length === 0, realErrors(errors).join('|').slice(0, 300))
  await ctx.close()
}

// ---------------- Reduced motion: no intro, no pulse, no auto-rotate ----------------
{
  const { ctx, page, errors } = await session(DESKTOP, { reduced: true })
  await sleep(400)
  const idle = await count(page, 2000)
  info(`reduced motion idle 2s: ${JSON.stringify(idle)}`)
  check('reduced motion: idle view renders no frames after the first paint', idle.raf === 0 && idle.renders === 0, JSON.stringify(idle))
  const b = await canvasBox(page)
  await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2); await page.mouse.down()
  await page.mouse.move(b.x + b.width / 2 + 120, b.y + b.height / 2 + 30, { steps: 8 }); await page.mouse.up()
  const after = await count(page, 1200)
  check('reduced motion: orbit still renders frames', after.raf > 5, JSON.stringify(after))
  await sleep(3000)
  const redSettle = await count(page, 1200)
  check('reduced motion: settled after orbit', redSettle.raf === 0 && redSettle.renders === 0, JSON.stringify(redSettle))
  check('no console errors (reduced)', realErrors(errors).length === 0, realErrors(errors).join('|').slice(0, 300))
  await ctx.close()
}

// ---------------- Mobile: idle still ----------------
{
  const { ctx, page, errors } = await session(MOBILE)
  await sleep(8000)
  const idle = await count(page, 2000)
  check('390px: idle after settle has zero frames', idle.raf === 0 && idle.renders === 0, JSON.stringify(idle))
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/map3d-390.png` })
  check('no console errors (390)', realErrors(errors).length === 0, realErrors(errors).join('|').slice(0, 300))
  await ctx.close()
}

await browser.close()
const failed = results.filter((r) => !r.ok)
console.log(`\n${results.length - failed.length} passed, ${failed.length} failed`)
process.exit(failed.length ? 1 : 0)
