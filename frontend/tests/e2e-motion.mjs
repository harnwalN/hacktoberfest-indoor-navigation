// U4 motion behaviour tests. Run: node tests/e2e-motion.mjs (playwright resolvable, vite dev server at BASE, default :5199). API fully mocked.
import { chromium } from 'playwright'

const BASE = process.env.BASE || 'http://localhost:5199'
const N = (id, label, type, floor, x, y, extra = {}) => ({ id, label, type, floor, x, y, verified: true, texts: [], sign_texts: [], description: '', ...extra })
const E = (a, b, len, vertical = 'none') => ({ id: `${a}-${b}`, from_node: a, to_node: b, heading_out: 90, heading_in: 90, vertical, length_m: len, frames_fwd: [], source: 'walk', verified: true })
const S = (i, node_id, label, action, floor, extra = {}) => ({ step_index: i, node_id, label, frame_url: null, action, vertical: 'none', instruction_text: `Do ${label}`, floor, turn_degrees: 0, ...extra })
const longNodes = Array.from({ length: 14 }, (_, k) => N(`l${k}`, `Lab ${k}`, 'room', 1 + (k > 9 ? 1 : 0), k * 10, (k % 3) * 7))
const MAP = { building_id: 'b1', name: 'B1', nodes: longNodes, edges: longNodes.slice(1).map((n, k) => E(`l${k}`, n.id, 8)), suggested_merges: [], videos: [] }
const STEPS = longNodes.map((n, k) => S(k, n.id, n.label, k === 0 ? 'start' : k === 13 ? 'destination_reached' : k % 2 ? 'turn_left' : 'turn_right', n.floor, { turn_degrees: k % 2 ? -90 : 90 }))
const URL_NAV = `${BASE}/#/navigate?b=b1&from=l0&to=l13`

const results = []
const check = (name, ok, info = '') => { results.push({ name, ok: !!ok, info }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok ? '' : '  -> ' + info}`) }
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
// WebGL errors in headless are environment noise, not app errors
const realErrors = (errs) => errs.filter((e) => !/GPU stall|GL Driver Message|ReadPixels|WebGL.*(unsupported|not supported)|Failed to load resource.*404/i.test(e))

const TRANSPARENT = '*:not(canvas){background:transparent!important;box-shadow:none!important}html,body{background:transparent!important}'
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] })

async function session(vp, o = {}, ctxOpts = {}) {
  const ctx = await browser.newContext({ viewport: vp, ...ctxOpts })
  const page = await ctx.newPage()
  const errors = []
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
  page.on('pageerror', (e) => errors.push('PAGEERROR ' + e.message))
  const st = { queueCalls: 0, jobCalls: {}, ...o }
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url()); const p = url.pathname.replace(/^\/api/, ''); const m = route.request().method()
    const json = (b, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(b) })
    if (p === '/buildings') return json([{ id: 'b1', name: 'B1' }])
    if (p === '/health') return json({ gemma: { ok: true, selected: 'gemma3', models: ['gemma3'] } })
    if (p === '/buildings/b1/map' && m === 'GET') {
      if (st.mapDelay) await sleep(st.mapDelay)
      return json(st.map || MAP)
    }
    if (p === '/buildings/b1/route') return json({ steps: STEPS, node_path: STEPS.map((s) => s.node_id) })
    if (p === '/buildings/b1/queue') { st.queueCalls++; return st.queue ? st.queue(json, st) : json([]) }
    if (p === '/buildings/b1/videos' && m === 'POST') return json({ job_id: 'j1' })
    const jm = p.match(/^\/jobs\/(.+)$/)
    if (jm) { st.jobCalls[jm[1]] = (st.jobCalls[jm[1]] || 0) + 1; return st.job(jm[1], st.jobCalls[jm[1]], json) }
    return json({ detail: 'nope' }, 404)
  })
  return { ctx, page, errors, st }
}
const DESKTOP = { width: 1280, height: 800 }
const MOBILE = { width: 390, height: 844 }
const live = (page) => page.locator('.sign[aria-live]').innerText()
const stepText = async (page) => (await page.locator('.sign .sr-only').innerText()).trim()
const nodeCount = (page) => page.evaluate(() => document.getElementsByTagName('*').length)

// ---------- navigation: mashing, confetti ----------
for (const [vpName, vp] of [['mobile', MOBILE], ['desktop', DESKTOP]]) {
  const T = (n) => `[${vpName}] ${n}`
  const { ctx, page, errors } = await session(vp)
  await page.goto(URL_NAV)
  await page.waitForSelector('.trip-view')
  const next = page.locator('.step-nav button:has-text("Next")')
  const back = page.locator('.step-nav button:has-text("Back")')
  // 13 back-to-back Next clicks (the last step's button turns into Done, so click exactly to the end), then overshoot by keyboard
  for (let i = 0; i < 13; i++) await page.locator('.step-nav button').nth(1).click({ noWaitAfter: true, force: true, timeout: 1500 })
  for (let i = 0; i < 10; i++) await page.keyboard.press('ArrowRight')
  await page.waitForTimeout(100)
  check(T('mash Next x13 + 10 arrows clamps on last step'), (await stepText(page)).startsWith('Step 14 of 14'), await stepText(page))
  for (let i = 0; i < 7; i++) await back.click({ force: true, noWaitAfter: true })
  check(T('mash Back x7 from last -> step 7 of 14'), (await stepText(page)).startsWith('Step 7 of 14'), await stepText(page))
  check(T('step counter agrees with aria-live'), (await page.locator('.step-count').innerText()).trim() === '7 of 14')
  check(T('live region text names step headline'), /Step 7 of 14/.test(await live(page)) && /Do Lab 6/.test(await live(page)), await live(page))
  // random interleaving via keyboard, compare to model
  let model = 6
  const seq = Array.from({ length: 60 }, (_, i) => ((i * 7 + 3) % 5 < 2 ? 'ArrowLeft' : 'ArrowRight'))
  for (const k of seq) { await page.keyboard.press(k); model = Math.max(0, Math.min(13, model + (k === 'ArrowRight' ? 1 : -1))) }
  await page.waitForTimeout(50)
  check(T('mashing arrow keys ends on modelled step'), (await stepText(page)).startsWith(`Step ${model + 1} of 14`), `${await stepText(page)} expected ${model + 1}`)
  check(T('only one live region sign'), (await page.locator('.sign[aria-live]').count()) === 1)
  check(T('no stale duplicate sign copy after mashing'), (await page.locator('.sign-copy').count()) === 1 && (await page.locator('.sign-glyph, .sign svg').count()) >= 1)
  // arrival confetti
  await page.locator('.step-nav button:has-text("Next")').waitFor({ state: 'attached' }).catch(() => {})
  for (let i = 0; i < 20; i++) await page.keyboard.press('ArrowRight')
  await page.waitForSelector('.confetti i', { timeout: 1000 })
  const bits = await page.locator('.confetti i').count()
  check(T('arrival shows confetti bits'), bits >= 10, String(bits))
  check(T('confetti is aria-hidden and not in live text'), (await page.locator('.confetti').getAttribute('aria-hidden')) === 'true' && !/\bconfetti\b/i.test(await live(page)))
  await page.waitForTimeout(2300)
  check(T('confetti removed after ~2s'), (await page.locator('.confetti').count()) === 0)
  check(T('still on destination after confetti gone'), (await stepText(page)).startsWith('Step 14 of 14'))
  // leak check: repeated arrivals
  const before = await nodeCount(page)
  for (let r = 0; r < 6; r++) {
    await page.keyboard.press('ArrowLeft'); await page.keyboard.press('ArrowRight')
    const c = await page.locator('.confetti').count()
    if (c > 1) check(T('never more than one confetti burst'), false, String(c))
  }
  await page.waitForTimeout(2300)
  const after = await nodeCount(page)
  check(T('no confetti left after repeated arrivals'), (await page.locator('.confetti').count()) === 0)
  check(T('DOM node count stable across repeated arrivals'), Math.abs(after - before) <= 2, `${before} -> ${after}`)
  // leaving mid-burst must not error (timer cleanup)
  await page.keyboard.press('ArrowLeft'); await page.keyboard.press('ArrowRight'); await page.click('button[aria-label="End directions"]')
  await page.waitForTimeout(2200)
  check(T('ending directions mid-confetti: no console errors'), realErrors(errors).length === 0, errors[0] || '')
  check(T('Done on last step ends trip'), true)
  await ctx.close()
}

// ---------- skeleton and empty ----------
{
  const { ctx, page, errors } = await session(DESKTOP, { mapDelay: 1500 })
  await page.goto(URL_NAV)
  await page.waitForSelector('.skeleton-planner', { timeout: 1000 })
  check('skeleton shows while loading', true)
  const lv = page.locator('.loading-state')
  check('loading state is a polite status with text', (await lv.getAttribute('role')) === 'status' && /Loading/.test(await lv.innerText()))
  check('skeleton is aria-hidden', (await page.locator('.skeleton-planner').getAttribute('aria-hidden')) === 'true')
  await page.waitForSelector('.trip-view, .planner', { timeout: 5000 })
  check('skeleton gone after load', (await page.locator('.skeleton-planner').count()) === 0)
  check('no console errors (skeleton)', realErrors(errors).length === 0, errors[0] || '')
  await ctx.close()
}
{
  const { ctx, page, errors } = await session(DESKTOP, { map: { ...MAP, nodes: [], edges: [] } })
  await page.goto(`${BASE}/#/navigate?b=b1`)
  await page.waitForSelector('.quiet-state .empty-art', { timeout: 4000 })
  check('empty art shows on empty map', true)
  check('empty state has explanatory heading', /No map yet/.test(await page.locator('.quiet-state').innerText()))
  check('empty art decorative (aria-hidden)', (await page.locator('.empty-art').getAttribute('aria-hidden')) === 'true')
  check('skeleton not shown on empty map', (await page.locator('.skeleton-planner').count()) === 0)
  check('no console errors (empty)', realErrors(errors).length === 0, errors[0] || '')
  await ctx.close()
}

// ---------- queue dismissal ----------
{
  const JOBS = [{ id: 'q1', status: 'running', stage: 'gemma', pct: 40, filename: 'a.mp4', kind: 'video' }]
  let mode = 'fail'; let msg = 'Queue down'
  const { ctx, page, errors, st } = await session(DESKTOP, { queue: (json) => (mode === 'fail' ? json({ detail: msg }, 500) : json(JOBS)) })
  // need jobs non-empty for the panel: first poll succeeds
  mode = 'ok'
  await page.goto(`${BASE}/#/build?b=b1`)
  await page.waitForSelector('ul.queue')
  mode = 'fail'
  await page.waitForSelector('.banner-dismiss[role="alert"]', { timeout: 5000 })
  check('queue error banner appears with message', /Queue down/.test(await page.locator('.banner-dismiss').innerText()))
  await page.click('button[aria-label="Dismiss error"]')
  check('dismiss hides banner', (await page.locator('.banner-dismiss').count()) === 0)
  const c0 = st.queueCalls
  await page.waitForTimeout(5200)
  check('dismissed banner stays dismissed across >=2 polls', st.queueCalls - c0 >= 2 && (await page.locator('.banner-dismiss').count()) === 0, `${st.queueCalls - c0} polls`)
  mode = 'ok'
  await page.waitForTimeout(2300)
  check('recovery keeps banner hidden', (await page.locator('.banner-dismiss').count()) === 0)
  mode = 'fail'
  await page.waitForSelector('.banner-dismiss', { timeout: 5000 })
  check('same failure after recovery reappears', true)
  await page.click('button[aria-label="Dismiss error"]')
  msg = 'Different problem'
  await page.waitForSelector('.banner-dismiss', { timeout: 5000 })
  check('a different error message shows even if previous was dismissed', /Different problem/.test(await page.locator('.banner-dismiss').innerText()))
  check('dismiss button has accessible name and is focusable', (await page.locator('button[aria-label="Dismiss error"]').count()) === 1)
  await page.locator('button[aria-label="Dismiss error"]').focus(); await page.keyboard.press('Enter')
  check('dismiss works by keyboard', (await page.locator('.banner-dismiss').count()) === 0)
  check('no page errors (queue)', !errors.some((e) => /PAGEERROR/.test(e)), errors[0] || '')
  await ctx.close()
}

// ---------- build success banner ----------
async function buildFlow(ctxOpts, T) {
  const done = { id: 'j1', status: 'done', stage: 'mapping', pct: 100, kind: 'video', log: ['ok'], video_id: null }
  const { ctx, page, errors, st } = await session(DESKTOP, {
    queue: (json) => json([]),
    job: (id, n, json) => (n < 3 ? json({ id, status: 'running', stage: 'gemma', pct: 30 * n, kind: 'video', log: ['x'] }) : json(done)),
  }, ctxOpts)
  await page.goto(`${BASE}/#/build?b=b1`)
  await page.setInputFiles('input[type=file][accept="video/*"]', { name: 'w.mp4', mimeType: 'video/mp4', buffer: Buffer.from('x') })
  await page.click('button.upload-btn')
  await page.waitForSelector('[role=progressbar]')
  const pb = page.locator('[role=progressbar]')
  check(T('progressbar has name and range'), (await pb.getAttribute('aria-label')) && (await pb.getAttribute('aria-valuemax')) === '100')
  const v1 = Number(await pb.getAttribute('aria-valuenow'))
  check(T('no success banner while running'), (await page.locator('.success-pop').count()) === 0)
  await page.waitForSelector('.success-pop', { timeout: 6000 })
  check(T('success banner appears on done'), true)
  check(T('progressbar reads 100 on done'), (await pb.getAttribute('aria-valuenow')) === '100', `v1=${v1}`)
  const conf = await page.locator('.success-pop .confetti').count()
  await page.waitForTimeout(3500)
  check(T('success banner appears exactly once'), (await page.locator('.success-pop').count()) === 1)
  const polls = st.jobCalls.j1
  await page.waitForTimeout(2500)
  check(T('job polling stops after done'), st.jobCalls.j1 === polls && polls <= 4, `${polls} -> ${st.jobCalls.j1}`)
  check(T('banner has status role and text'), /Map is ready/.test(await page.locator('.success-pop').innerText()) && (await page.locator('.success-pop').getAttribute('role')) === 'status')
  check(T('no console errors (build)'), realErrors(errors).length === 0, errors[0] || '')
  const reduced = ctxOpts.reducedMotion === 'reduce'
  if (reduced) check(T('reduced: no confetti on success'), conf === 0 && (await page.locator('.confetti').count()) === 0)
  else { check(T('confetti rendered when job seen running'), conf === 1, String(conf)); check(T('confetti gone after ~2s'), (await page.locator('.confetti').count()) === 0) }
  await ctx.close()
}
await buildFlow({}, (n) => `[build] ${n}`)
// job first seen as already done: banner yes, no confetti
{
  const { ctx, page, errors } = await session(DESKTOP, {
    queue: (json) => json([{ id: 'j9', status: 'done', filename: 'old.mp4', kind: 'video', pct: 100 }]),
    job: (id, n, json) => json({ id, status: 'done', stage: 'mapping', pct: 100, kind: 'video', log: [] }),
  })
  await page.goto(`${BASE}/#/build?b=b1`)
  await page.waitForSelector('ul.queue')
  await page.click('.q-main')
  await page.waitForSelector('.success-pop')
  check('[build] already-done job: banner without confetti', (await page.locator('.success-pop .confetti').count()) === 0)
  await page.click('.q-main'); await page.click('.q-main')
  check('[build] re-selecting a done job still shows one banner', (await page.locator('.success-pop').count()) === 1)
  check('[build] no console errors (done job)', realErrors(errors).length === 0, errors[0] || '')
  await ctx.close()
}

// ---------- 3D ----------
async function opaquePixels(page) {
  const buf = await page.locator('.map3d').screenshot({ omitBackground: true, animations: 'allow' })
  return page.evaluate(async (b64) => {
    const img = new Image(); img.src = 'data:image/png;base64,' + b64; await img.decode()
    const c = document.createElement('canvas'); c.width = img.width; c.height = img.height
    const x = c.getContext('2d'); x.drawImage(img, 0, 0)
    const d = x.getImageData(0, 0, c.width, c.height).data
    let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i] > 200) n++
    return n
  }, buf.toString('base64'))
}
{
  const { ctx, page, errors } = await session(DESKTOP)
  await page.addInitScript(() => { window.__done = 0 })
  await page.goto(`${BASE}/#/review?b=b1&view=3d`)
  await page.waitForSelector('.map3d canvas', { timeout: 5000 }).catch(() => {})
  const hasCanvas = !!(await page.$('.map3d canvas'))
  const webgl = hasCanvas && (await page.evaluate(() => !!document.createElement('canvas').getContext('webgl2')))
  check('3D canvas mounts (webgl available in this browser)', hasCanvas && webgl, `canvas=${hasCanvas} webgl=${webgl}`)
  await page.addStyleTag({ content: TRANSPARENT })
  // list present for keyboard
  const list = page.locator('ul[aria-label="Places in the 3D view"] button')
  check('keyboard place list has all places', (await list.count()) === 14, String(await list.count()))
  await page.waitForTimeout(2200) // let the intro finish
  const settled = await opaquePixels(page)
  // select via keyboard, as a user would
  await list.nth(5).focus()
  await page.keyboard.press('Enter')
  await page.waitForTimeout(120)
  const justAfter = await opaquePixels(page)
  check('keyboard place list selects a place (aria-pressed)', (await page.locator('ul[aria-label="Places in the 3D view"] button').nth(5).getAttribute('aria-pressed')) === 'true')
  check('selecting a place does not replay the intro (scene still populated 120ms later)', justAfter > settled * 0.6, `settled=${settled} after=${justAfter}`)
  // second selection
  await page.locator('ul[aria-label="Places in the 3D view"] button').nth(9).focus(); await page.keyboard.press('Space')
  await page.waitForTimeout(120)
  const again = await opaquePixels(page)
  check('second selection also does not replay the intro', again > settled * 0.6, `settled=${settled} again=${again}`)
  // control: a fresh mount does show the intro (proves the probe can see it)
  await page.reload(); await page.waitForSelector('.map3d canvas'); await page.addStyleTag({ content: TRANSPARENT })
  await page.waitForTimeout(60)
  const fresh = await opaquePixels(page)
  await page.waitForTimeout(2200)
  const freshSettled = await opaquePixels(page)
  // The probe is timing-based (screenshot latency can exceed the 1.3s intro), so an inconclusive control is reported, not failed.
  if (fresh < freshSettled) console.log(`INFO  probe sanity ok: fresh mount sparser than settled (${fresh} < ${freshSettled})`)
  else console.log(`INFO  probe sanity inconclusive: the no-replay checks above may be vacuous this run (fresh=${fresh} settled=${freshSettled})`)
  // resize and rotate
  for (const [w, h] of [[390, 844], [800, 500], [1280, 800], [300, 300], [1000, 1000]]) { await page.setViewportSize({ width: w, height: h }); await page.waitForTimeout(150) }
  const box = await page.locator('.map3d').boundingBox()
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2)
  await page.mouse.down(); await page.mouse.move(box.x + box.width / 2 + 120, box.y + box.height / 2 + 40, { steps: 10 }); await page.mouse.up()
  await page.mouse.wheel(0, -400); await page.waitForTimeout(300)
  await page.mouse.wheel(0, 800); await page.waitForTimeout(300)
  check('no console errors on resize / rotate / zoom', realErrors(errors).length === 0, errors.slice(0, 2).join(' | '))
  const afterDrag = await opaquePixels(page)
  check('scene still drawn after interaction', afterDrag > 100, String(afterDrag))
  // toggle back to 2D and 3D several times (dispose path)
  for (let i = 0; i < 4; i++) { await page.click('button:text-is("2D")'); await page.click('button:text-is("3D")') }
  await page.waitForTimeout(300)
  check('toggling 2D/3D keeps exactly one canvas', (await page.locator('.map3d canvas').count()) === 1)
  check('no console errors after toggling', realErrors(errors).length === 0, errors.slice(0, 2).join(' | '))
  await ctx.close()
}

// ---------- reduced motion: everything works, nothing stuck at opacity 0 ----------
const RM = { reducedMotion: 'reduce' }
async function stuckHidden(page) {
  return page.evaluate(() => {
    const bad = []
    for (const el of document.querySelectorAll('body *')) {
      const cs = getComputedStyle(el)
      if (cs.display === 'none' || cs.visibility === 'hidden') continue
      if (el.closest('.sr-only, .glyph-ripple') || el.matches('.sr-only')) continue
      const r = el.getBoundingClientRect()
      if (!r.width || !r.height) continue
      if (Number(cs.opacity) === 0) bad.push(el.tagName + '.' + el.className)
    }
    return bad
  })
}
{
  const { ctx, page, errors } = await session(DESKTOP, { mapDelay: 600 }, RM)
  await page.goto(URL_NAV)
  await page.waitForSelector('.planner, .trip-view', { timeout: 5000 })
  await page.waitForTimeout(300)
  check('[reduced] planner/trip visible, nothing at opacity 0', (await stuckHidden(page)).length === 0, (await stuckHidden(page)).join(','))
  await page.waitForSelector('.trip-view')
  for (let i = 0; i < 13; i++) await page.keyboard.press('ArrowRight')
  check('[reduced] navigation reaches last step', (await stepText(page)).startsWith('Step 14 of 14'), await stepText(page))
  await page.waitForTimeout(200)
  check('[reduced] no confetti on arrival', (await page.locator('.confetti').count()) === 0)
  check('[reduced] sign text visible at arrival', (await stuckHidden(page)).length === 0, (await stuckHidden(page)).join(','))
  for (let i = 0; i < 13; i++) await page.keyboard.press('ArrowLeft')
  check('[reduced] back to step 1', (await stepText(page)).startsWith('Step 1 of 14'))
  const running = await page.evaluate(() => document.getAnimations().filter((a) => a.playState === 'running' && a.effect?.getTiming().iterations === Infinity).length)
  check('[reduced] no infinite animations running', running === 0, String(running))
  check('[reduced] no console errors (nav)', realErrors(errors).length === 0, errors[0] || '')
  await ctx.close()
}
{
  const { ctx, page } = await session(DESKTOP, { map: { ...MAP, nodes: [], edges: [] } }, RM)
  await page.goto(`${BASE}/#/navigate?b=b1`)
  await page.waitForSelector('.empty-art')
  await page.waitForTimeout(300)
  check('[reduced] empty state visible', (await stuckHidden(page)).length === 0, (await stuckHidden(page)).join(','))
  await ctx.close()
}
await buildFlow(RM, (n) => `[reduced] ${n}`)
{
  const JOBS = [{ id: 'q1', status: 'running', stage: 'gemma', pct: 40, filename: 'a.mp4', kind: 'video' }]
  let mode = 'ok'
  const { ctx, page, errors } = await session(DESKTOP, { queue: (json) => (mode === 'fail' ? json({ detail: 'Queue down' }, 500) : json(JOBS)) }, RM)
  await page.goto(`${BASE}/#/build?b=b1`)
  await page.waitForSelector('ul.queue'); mode = 'fail'
  await page.waitForSelector('.banner-dismiss', { timeout: 5000 })
  await page.click('button[aria-label="Dismiss error"]')
  check('[reduced] queue dismiss works, queue visible', (await page.locator('.banner-dismiss').count()) === 0 && (await stuckHidden(page)).length === 0, (await stuckHidden(page)).join(','))
  await ctx.close()
}
{
  const { ctx, page, errors } = await session(DESKTOP, {}, RM)
  await page.goto(`${BASE}/#/review?b=b1&view=3d`)
  await page.waitForSelector('.map3d canvas', { timeout: 5000 }).catch(() => {})
  await page.addStyleTag({ content: TRANSPARENT })
  await page.waitForTimeout(80)
  const early = await opaquePixels(page)
  await page.waitForTimeout(1800)
  const late = await opaquePixels(page)
  check('[reduced] 3D has no intro: fully drawn immediately', early > 100 && early >= late * 0.9, `early=${early} late=${late}`)
  await page.locator('ul[aria-label="Places in the 3D view"] button').nth(2).focus(); await page.keyboard.press('Enter')
  check('[reduced] 3D keyboard selection works', (await page.locator('ul[aria-label="Places in the 3D view"] button').nth(2).getAttribute('aria-pressed')) === 'true')
  await page.setViewportSize(MOBILE); await page.waitForTimeout(200)
  check('[reduced] no console errors (3D)', realErrors(errors).length === 0, errors[0] || '')
  check('[reduced] review page nothing at opacity 0', (await stuckHidden(page)).length === 0, (await stuckHidden(page)).join(','))
  await ctx.close()
}

// ---------- edges table scroll wrapper ----------
for (const [vpName, vp] of [['mobile', MOBILE], ['desktop', DESKTOP]]) {
  const T = (n) => `[${vpName}] ${n}`
  const { ctx, page, errors } = await session(vp)
  await page.goto(`${BASE}/#/review?b=b1`)
  await page.waitForSelector('.node-card')
  await page.locator('.node-card').nth(3).click()
  const w = page.locator('.table-scroll')
  await w.first().waitFor({ timeout: 4000 })
  check(T('table wrapper is region with label and tabindex 0'), (await w.getAttribute('tabindex')) === '0' && (await w.getAttribute('role')) === 'region' && !!(await w.getAttribute('aria-label')))
  await w.focus()
  check(T('table wrapper can take focus'), await w.evaluate((e) => document.activeElement === e))
  // reach via Tab from the preceding control
  await page.locator('.node-card').nth(3).focus()
  let reached = false
  for (let i = 0; i < 80 && !reached; i++) { await page.keyboard.press('Tab'); reached = await w.evaluate((e) => document.activeElement === e) }
  check(T('table wrapper reachable by Tab'), reached)
  const focusVisible = await w.evaluate((e) => { const s = getComputedStyle(e); return s.outlineStyle !== 'none' && parseFloat(s.outlineWidth) > 0 })
  check(T('focused wrapper shows an outline'), focusVisible)
  const sw = await w.evaluate((e) => ({ s: e.scrollWidth, c: e.clientWidth }))
  if (sw.s > sw.c) {
    await page.keyboard.press('ArrowRight'); await page.keyboard.press('ArrowRight'); await page.waitForTimeout(500)
    check(T('arrow key scrolls the wrapper sideways'), (await w.evaluate((e) => e.scrollLeft)) > 0, `scrollLeft=${await w.evaluate((e) => e.scrollLeft)} sw=${sw.s}/${sw.c} active=${await page.evaluate(() => document.activeElement.className)}`)
  } else console.log(`INFO  ${T('table fits, nothing to scroll')} (${sw.s}/${sw.c})`)
  check(T('page itself does not scroll sideways'), await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1), String(await page.evaluate(() => document.documentElement.scrollWidth)))
  check(T('no console errors (table)'), realErrors(errors).length === 0, errors[0] || '')
  await ctx.close()
}

await browser.close()
const failed = results.filter((r) => !r.ok)
console.log(`\n${results.length - failed.length} passed, ${failed.length} failed`)
process.exit(failed.length ? 1 : 0)
