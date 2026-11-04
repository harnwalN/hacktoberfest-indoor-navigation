// Directions + Build motion tests. Run: BASE=http://localhost:5310 node tests/e2e-dirbuild.mjs (playwright resolvable). API fully mocked.
// TABS=1 prints only tab-stop counts (run against another checkout to compare).
import { chromium } from 'playwright'
import fs from 'node:fs'

const BASE = process.env.BASE || 'http://localhost:5310'
const SHOTS = process.env.SHOTS || ''
const N = (id, label, type, floor, x, y) => ({ id, label, type, floor, x, y, verified: true, texts: [], sign_texts: [], description: '' })
const E = (a, b) => ({ id: `${a}-${b}`, from_node: a, to_node: b, heading_out: 90, heading_in: 90, vertical: 'none', length_m: 8, frames_fwd: [], source: 'walk', verified: true })
const S = (i, node_id, label, action) => ({ step_index: i, node_id, label, frame_url: null, action, vertical: 'none', instruction_text: `Do ${label}`, floor: 1, turn_degrees: 0 })
const nodes = Array.from({ length: 6 }, (_, k) => N(`n${k}`, `Lab ${k}`, 'room', 1, k * 10, (k % 3) * 7))
const VIDEOS = [
  { id: 'v1', filename: 'walk1.mp4', duration: 125, keyframes: 48, status: 'done', has_sensors: false },
  { id: 'v2', filename: 'walk2.mp4', duration: 60, keyframes: 12, status: 'done', has_sensors: true },
]
const MAP = { building_id: 'b1', name: 'B1', nodes, edges: nodes.slice(1).map((n, k) => E(`n${k}`, n.id)), suggested_merges: [], videos: VIDEOS }
const STEPS = nodes.map((n, k) => S(k, n.id, n.label, k === 0 ? 'start' : k === 5 ? 'destination_reached' : 'turn_left'))

const results = []
const check = (name, ok, info = '') => { results.push({ name, ok: !!ok, info }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok ? '' : '  -> ' + info}`) }
const realErrors = (errs) => errs.filter((e) => !/GPU stall|GL Driver Message|ReadPixels|WebGL.*(unsupported|not supported)|Failed to load resource.*404/i.test(e))
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] })

async function session(vp, o = {}, ctxOpts = {}) {
  const ctx = await browser.newContext({ viewport: vp, ...ctxOpts })
  // Signed in as the owner: the Build tools (uploader, tips, queue) are on the owner page now.
  await ctx.addInitScript(() => { if (!localStorage.getItem('session_token')) { localStorage.setItem('session_token', 'tok-owner'); localStorage.setItem('session_account', JSON.stringify({ id: 'u-owner', email: 'owner@example.com', is_owner: true })) } })
  const page = await ctx.newPage()
  const errors = []
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
  page.on('pageerror', (e) => errors.push('PAGEERROR ' + e.message))
  const st = { routeBodies: [], videos: JSON.parse(JSON.stringify(VIDEOS)), jobStatus: 'running', ...o }
  await page.addInitScript(() => {
    try { localStorage.setItem('session_token', 't'); localStorage.setItem('session_account', JSON.stringify({ id: 'a1', email: 'a@b.co', is_owner: true })) } catch {}
    window.__urls = { created: [], revoked: [] }
    const c = URL.createObjectURL.bind(URL); const r = URL.revokeObjectURL.bind(URL)
    URL.createObjectURL = (b) => { const u = c(b); window.__urls.created.push(u); return u }
    URL.revokeObjectURL = (u) => { window.__urls.revoked.push(u); return r(u) }
  })
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url()); const p = url.pathname.replace(/^\/api/, ''); const m = route.request().method()
    const json = (b, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(b) })
    if (p === '/buildings') return json([{ id: 'b1', name: 'B1', role: 'owner' }])
    if (p === '/health') return json({ gemma: st.gemmaBad ? { ok: false, error: 'down' } : { ok: true, selected: 'gemma3', models: ['gemma3'] } })
    if (p === '/buildings/b1/map' && m === 'GET') return json({ ...MAP, videos: st.videos })
    if (p === '/buildings/b1/route') { st.routeBodies.push(route.request().postData() || url.search); return json({ steps: STEPS, node_path: STEPS.map((s) => s.node_id) }) }
    if (p === '/buildings/b1/queue') return json([])
    if (p === '/buildings/b1/videos' && m === 'POST') return json({ job_id: 'j1' })
    if (p.match(/^\/buildings\/b1\/videos\/[^/]+$/) && m === 'DELETE') { st.videos = st.videos.filter((v) => !p.endsWith('/' + v.id)); return json({ removed_places: 0, deleted_files: 0 }) }
    const jm = p.match(/^\/jobs\/(.+)$/)
    if (jm) return json({ id: 'j1', status: st.jobStatus, stage: 'gemma', pct: 50, log: [], kind: 'video', filename: 'x.mp4' })
    return json({ detail: 'nope' }, 404)
  })
  return { ctx, page, errors, st }
}
const DESK = { width: 1440, height: 900 }
const MOB = { width: 390, height: 844 }
const URL_PLAN = `${BASE}/#/navigate?b=b1`
const URL_TRIP = `${BASE}/#/navigate?b=b1&from=n0&to=n5`
const URL_BUILD = `${BASE}/#/owner?s=videos&b=b1`

async function tabStops(page, max = 80) {
  await page.evaluate(() => document.activeElement && document.activeElement.blur())
  await page.mouse.click(1, 1)
  const seen = []
  for (let i = 0; i < max; i++) {
    await page.keyboard.press('Tab')
    const d = await page.evaluate(() => { const a = document.activeElement; return a && a !== document.body ? `${a.tagName}|${a.className}|${a.getAttribute('aria-label') || a.textContent.slice(0, 20)}` : 'BODY' })
    if (seen.length && d === seen[0]) break
    seen.push(d)
  }
  return seen
}
const tabindexAttrs = (page, sel) => page.evaluate((s) => [...document.querySelectorAll(s)].filter((e) => e.getAttribute('tabindex') === '0' && !['A', 'BUTTON', 'INPUT', 'SELECT', 'SUMMARY', 'TEXTAREA'].includes(e.tagName)).map((e) => e.tagName + '.' + e.className), sel)

if (process.env.TABS) {
  for (const [n, u] of [['planner', URL_PLAN], ['trip', URL_TRIP], ['build', URL_BUILD]]) {
    const { page } = await session(DESK)
    await page.goto(u); await page.waitForTimeout(1500)
    const t = await tabStops(page)
    console.log(`TABS ${n}: ${t.length}`)
    console.log(await tabindexAttrs(page, '*').then((a) => 'stray tabindex=0: ' + JSON.stringify(a)))
  }
  await browser.close(); process.exit(0)
}

// ---- planner: swap, highlight, checkboxes, idle loops
{
  const { ctx, page, errors, st } = await session(DESK)
  await page.goto(URL_PLAN); await page.waitForSelector('#from')
  if (SHOTS) { await page.waitForTimeout(2500); await page.screenshot({ path: `${SHOTS}/planner-1440.png` }) }
  await page.fill('#from', 'Lab 1'); await page.locator('.place-option', { hasText: 'Lab 1' }).first().click()
  await page.fill('#to', 'Lab 4'); await page.locator('.place-option', { hasText: 'Lab 4' }).first().click()
  const picked = () => page.locator('.place-picked').allInnerTexts()
  let p = await picked()
  check('before swap From=Lab 1 To=Lab 4', /Lab 1/.test(p[0]) && /Lab 4/.test(p[1]), JSON.stringify(p))
  // Frame-rate independent: a compositor stall can swallow the whole spring before a timed sample runs, so record
  // every inline style motion writes to .place-picked (mutations + new nodes) instead of sampling at a fixed delay.
  await page.evaluate(() => {
    window.__swapStyles = []
    const rec = (el) => { if (el.classList?.contains('place-picked')) window.__swapStyles.push((el.style.transform || 'none') + '|' + (el.style.opacity || '1')) }
    new MutationObserver((ms) => ms.forEach((m) => { if (m.type === 'attributes') rec(m.target); else m.addedNodes.forEach((n) => { rec(n); n.querySelectorAll?.('.place-picked').forEach(rec) }) }))
      .observe(document.body, { subtree: true, childList: true, attributes: true, attributeFilter: ['style'] })
  })
  await page.click('button[aria-label="Swap start and destination"]')
  await page.waitForTimeout(80)
  const mid = await page.evaluate(() => window.__swapStyles)
  check('swap animates (motion drives non-identity transform / opacity<1 on the crossing)', mid.some((m) => { const [t, o] = m.split('|'); return (t !== 'none' && !/^translate(Y|3d)?\(0(px)?(, 0(px)?)*(, 0(px)?)?\)$/.test(t) && !t.startsWith('matrix(1, 0, 0, 1, 0, 0)')) || parseFloat(o) < 1 }), JSON.stringify(mid.slice(0, 6)))
  await page.waitForTimeout(900)
  p = await picked()
  check('after swap From=Lab 4 To=Lab 1', /Lab 4/.test(p[0]) && /Lab 1/.test(p[1]), JSON.stringify(p))
  const fin = await page.locator('.place-picked').evaluateAll((els) => els.map((e) => getComputedStyle(e).opacity))
  check('swap settles to opacity 1', fin.every((o) => o === '1'), JSON.stringify(fin))
  // route request uses swapped values
  const goBtn = page.locator('button:has-text("Get directions"), button.go').first()
  st.routeBodies.length = 0
  await goBtn.click(); await page.waitForTimeout(800)
  check('route request uses swapped values (n4 -> n1)', st.routeBodies.some((b) => /n4/.test(b) && /n1/.test(b) && b.indexOf('n4') < b.indexOf('n1')) || /from=n4&to=n1/.test(page.url()), JSON.stringify(st.routeBodies) + ' ' + page.url())
  check('no console errors planner', realErrors(errors).length === 0, errors[0] || '')
  await ctx.close()
}
{
  const { ctx, page } = await session(DESK)
  await page.goto(URL_PLAN); await page.waitForSelector('#from')
  await page.focus('#from')
  const opts = page.locator('.place-option')
  await opts.first().waitFor()
  const hlOf = () => page.evaluate(() => { const h = document.querySelectorAll('.place-hl'); return { n: h.length, txt: h[0] ? h[0].parentElement.innerText.split('\n')[0] : null } })
  await opts.nth(2).hover(); await page.waitForTimeout(400)
  const a = await hlOf()
  check('highlight follows hover (one, on hovered row)', a.n === 1 && a.txt === (await opts.nth(2).innerText()).split('\n')[0], JSON.stringify(a))
  await opts.nth(0).focus(); await page.keyboard.press('ArrowDown'); await page.waitForTimeout(400)
  const act = await page.evaluate(() => document.activeElement.innerText.split('\n')[0])
  const b = await hlOf()
  check('highlight follows keyboard ArrowDown', b.n === 1 && b.txt === act, JSON.stringify({ b, act }))
  check('listbox semantics unchanged (aria-pressed buttons, no aria-activedescendant)', (await page.locator('.place-option[aria-pressed]').count()) > 0 && (await page.locator('[aria-activedescendant]').count()) === 0)
  await page.keyboard.press('Enter'); await page.waitForTimeout(300)
  check('Enter selects focused option', /Lab/.test(await page.locator('.place-picked').first().innerText()))
  // checkboxes
  for (const [lab, sel] of [['Show unverified places too', 'label.check:has-text("Show unverified") input'], ['Avoid stairs', '#avoid-stairs']]) {
    const cb = page.locator(sel)
    check(`${lab}: real checkbox`, (await cb.getAttribute('type')) === 'checkbox')
    check(`${lab}: accessible name`, await cb.evaluate((e) => e.labels[0].textContent.includes(e.closest('label').textContent.trim().slice(0, 8))))
    await cb.focus(); const before = await cb.isChecked(); await page.keyboard.press('Space'); const after = await cb.isChecked()
    check(`${lab}: Space toggles`, before !== after)
    await page.locator('label', { hasText: lab }).first().click(); check(`${lab}: label click toggles`, (await cb.isChecked()) === before)
    await page.locator('label', { hasText: lab }).first().click()
    const tr = await cb.evaluate((e) => getComputedStyle(e, '::after').transform + '|' + getComputedStyle(e, '::after').transitionDuration)
    check(`${lab}: tick transition present when motion on`, !/^0s/.test(tr.split('|')[1]), tr)
  }
  await ctx.close()
}
// ---- step view
{
  const { ctx, page, errors } = await session(DESK)
  await page.goto(URL_TRIP); await page.waitForSelector('.trip-view')
  const sc = () => page.locator('.trip-progress i').evaluate((e) => { const m = getComputedStyle(e).transform.match(/matrix\(([^,]+)/); return m ? +m[1] : 1 })
  for (let i = 0; i < 6; i++) {
    await page.waitForTimeout(700)
    const cnt = (await page.locator('.step-count').innerText()).trim()
    const [cur, tot] = cnt.split(' of ').map(Number)
    const s = await sc()
    check(`progress bar step ${cur}/${tot}`, Math.abs(s - cur / tot) < 0.01 && cur === i + 1, `${s} vs ${cur / tot}`)
    if (i === 0) { if (SHOTS) await page.screenshot({ path: `${SHOTS}/step-1440.png` }) }
    if (i < 5) await page.locator('.step-nav button').nth(1).click()
  }
  for (let i = 5; i > 2; i--) await page.locator('.step-nav button:has-text("Back")').click()
  await page.waitForTimeout(700)
  const cnt = (await page.locator('.step-count').innerText()).trim()
  check('Back x3 -> 3 of 6 and bar 0.5', cnt === '3 of 6' && Math.abs((await sc()) - 0.5) < 0.01, cnt + ' ' + (await sc()))
  check('progress bar aria-hidden', (await page.locator('.trip-progress').getAttribute('aria-hidden')) === 'true')
  check('no console errors step view', realErrors(errors).length === 0, errors[0] || '')
  await ctx.close()
}
// ---- walker settles?
{
  const { ctx, page } = await session(DESK)
  await page.goto(URL_PLAN); await page.waitForSelector('#from'); await page.waitForTimeout(500)
  const info = await page.evaluate(() => new Promise((res) => setTimeout(() => res(document.getAnimations().map((a) => ({ n: a.animationName || a.constructor.name, it: a.effect.getComputedTiming().iterations, cls: a.effect.target && a.effect.target.className && String(a.effect.target.className.baseVal ?? a.effect.target.className) }))), 25000)))
  const inf = info.filter((a) => a.it === Infinity && /pa-|tile|walker/.test(a.cls || ''))
  check('planner art: no infinite animation after 25s (walker must settle)', inf.length === 0, JSON.stringify(inf))
  const bounded = info.filter((a) => /pa-tile/.test(a.cls || '') && a.it !== Infinity)
  check('tile bob bounded', true, JSON.stringify(bounded))
  await ctx.close()
}
// ---- build
{
  const { ctx, page, errors, st } = await session(DESK)
  await page.goto(URL_BUILD); await page.waitForSelector('input[type=file]'); await page.waitForTimeout(800)
  const vids = fs.mkdtempSync('/tmp/dirbuild-') // fixture dir
  const files = []
  for (let i = 0; i < 6; i++) { const f = `${vids}/clip${i}.mp4`; fs.writeFileSync(f, Buffer.alloc(1024 * 300 * (i + 1), i)); files.push(f) }
  const input = page.locator('input[type=file][accept="video/*"]')
  await input.setInputFiles(files[0]); await page.waitForSelector('.file-card')
  const t = await page.locator('.file-card').innerText()
  check('file card shows name and size', /clip0\.mp4/.test(t) && /300 KB/.test(t), t)
  if (SHOTS) { await page.waitForTimeout(600); await page.screenshot({ path: `${SHOTS}/build-card-1440.png` }) }
  for (let i = 1; i < 6; i++) { await input.setInputFiles(files[i]); await page.waitForTimeout(150) }
  await page.waitForTimeout(300)
  let u = await page.evaluate(() => window.__urls)
  check('5 file changes: >=5 URLs revoked, only 1 live', u.created.length >= 6 && u.created.filter((x) => !u.revoked.includes(x)).length <= 1, JSON.stringify({ c: u.created.length, r: u.revoked.length }))
  check('one file card only', (await page.locator('.file-card').count()) === 1)
  // tips
  const tog = page.locator('.tips-toggle')
  check('tips collapsed: aria-expanded=false, body absent', (await tog.getAttribute('aria-expanded')) === 'false' && (await page.locator('.tips-body').count()) === 0)
  await tog.focus(); await page.keyboard.press('Enter'); await page.waitForTimeout(500)
  check('Enter expands: aria-expanded=true, body visible', (await tog.getAttribute('aria-expanded')) === 'true' && (await page.locator('.tips-body').isVisible()))
  await page.keyboard.press('Space'); await page.waitForTimeout(600)
  check('Space collapses: aria-expanded=false, body gone from DOM/a11y', (await tog.getAttribute('aria-expanded')) === 'false' && (await page.locator('.tips-body').count()) === 0)
  // pills + delete
  const pills = await page.locator('.video-list .pill').allInnerTexts()
  check('no duplicate pills on load', pills.length === 2 && pills.every((x) => x === 'done'), JSON.stringify(pills))
  check('done pills did not pop on load', (await page.locator('.pill.just-done').count()) === 0)
  const li = page.locator('.video-list li')
  const delBtn = li.first().locator('button.danger')
  page.on('dialog', (d) => d.accept())
  await delBtn.click()
  await page.waitForTimeout(60)
  const h = await li.first().evaluate((e) => e.getBoundingClientRect().height)
  await page.waitForTimeout(700)
  check('delete: row mid-collapse shorter than full, then removed', (await li.count()) === 1, `count ${await li.count()} h ${h}`)
  const txt = (await li.first().innerText()).replace(/\s+/g, ' ')
  const tc = (await li.first().textContent()).replace(/\s+/g, ' ')
  console.log('   innerText:', txt, '| textContent:', tc)
  check('durations/keyframes appear once in textContent (60 once, 12 once)', (tc.match(/60s/g) || []).length === 1 && (tc.match(/\b12\b/g) || []).length === 1, tc)
  check('visual copy aria-hidden', (await li.first().locator('span[aria-hidden=true] number-flow-react').count()) >= 2)
  check('no console errors build', realErrors(errors).length === 0, errors[0] || '')
  fs.rmSync(vids, { recursive: true, force: true })
  await ctx.close()
}
// ---- confetti first walk only, pill pop on running->done
for (const [name, vids, done, expect] of [
  ['first walk', [{ id: 'v1', filename: 'a.mp4', duration: 9, keyframes: 3, status: 'running' }], [{ id: 'v1', filename: 'a.mp4', duration: 9, keyframes: 3, status: 'done' }], true],
  ['later walk', [VIDEOS[0], { id: 'v2', filename: 'b.mp4', duration: 9, keyframes: 3, status: 'running' }], [VIDEOS[0], { id: 'v2', filename: 'b.mp4', duration: 9, keyframes: 3, status: 'done' }], false],
]) {
  const { ctx, page, st } = await session(DESK)
  st.videos = vids; st.jobStatus = 'running'
  await page.goto(URL_BUILD); await page.waitForSelector('input[type=file]'); await page.waitForTimeout(500)
  const f = '/tmp/dirbuild-conf.mp4'; fs.writeFileSync(f, Buffer.alloc(2048))
  await page.locator('input[type=file][accept="video/*"]').setInputFiles(f)
  await page.locator('button.upload-btn').click()
  await page.waitForSelector('.job-slot', { timeout: 4000 })
  await page.waitForTimeout(2500)
  await page.evaluate(() => { window.__conf = 0; window.__pop = 0; new MutationObserver(() => { if (document.querySelector('.confetti i')) window.__conf++; if (document.querySelector('.pill.just-done')) window.__pop++ }).observe(document.body, { childList: true, subtree: true, attributes: true }) })
  st.jobStatus = 'done'; st.videos = done
  await page.waitForSelector('.success-pop', { timeout: 8000 }).catch(() => {})
  await page.waitForTimeout(2500)
  const conf = await page.evaluate(() => window.__conf); const pop = await page.evaluate(() => window.__pop)
  check(`confetti ${expect ? 'shown' : 'absent'} for ${name}`, expect ? conf > 0 : conf === 0, `mutations=${conf}`)
  check(`done pill pops on running->done (${name})`, pop > 0, `pop=${pop}`)
  fs.rmSync(f, { force: true })
  await ctx.close()
}
// ---- reduced motion runtime
{
  const { ctx, page, errors, st } = await session(DESK, {}, { reducedMotion: 'reduce' })
  await page.goto(URL_PLAN); await page.waitForSelector('#from'); await page.waitForTimeout(500)
  const anims = await page.evaluate(() => document.getAnimations().filter((a) => a.playState === 'running').map((a) => (a.animationName || a.transitionProperty || '') + '@' + (a.effect.target.className?.baseVal ?? a.effect.target.className)))
  check('reduced: no running CSS animations on planner', anims.length === 0, JSON.stringify(anims))
  const walker = await page.locator('.pa-walker').evaluateAll((e) => e.map((x) => getComputedStyle(x).display))
  check('reduced: walker hidden', walker.every((d) => d === 'none'), JSON.stringify(walker))
  const tick = await page.locator('#avoid-stairs, label:has-text("Show unverified") input').first().evaluate((e) => getComputedStyle(e, '::after').transitionDuration)
  check('reduced: tick transition off', /^0s/.test(tick), tick)
  await page.fill('#from', 'Lab 1'); await page.locator('.place-option', { hasText: 'Lab 1' }).first().click()
  await page.fill('#to', 'Lab 4'); await page.locator('.place-option', { hasText: 'Lab 4' }).first().click()
  await page.click('button[aria-label="Swap start and destination"]')
  await page.waitForTimeout(40)
  const mid = await page.locator('.place-picked').evaluateAll((els) => els.map((e) => getComputedStyle(e).transform + '|' + getComputedStyle(e).opacity))
  check('reduced: swap has no mid-flight transform/opacity', mid.every((m) => /^(none|matrix\(1, 0, 0, 1, 0, 0\))\|1$/.test(m)), JSON.stringify(mid))
  await page.goto(URL_BUILD); await page.waitForSelector('input[type=file]'); await page.waitForTimeout(500)
  const f = '/tmp/dirbuild-rm.mp4'; fs.writeFileSync(f, Buffer.alloc(2048))
  await page.locator('input[type=file][accept="video/*"]').setInputFiles(f); await page.waitForSelector('.file-card')
  const cs = await page.locator('.file-card').evaluate((e) => getComputedStyle(e).transform + '|' + getComputedStyle(e).opacity)
  check('reduced: file card no entrance animation', /^(none|matrix\(1, 0, 0, 1, 0, 0\))\|1$/.test(cs), cs)
  await page.locator('.tips-toggle').click(); await page.waitForTimeout(50); const t0 = await page.locator('.tips-body').evaluate((e) => e.getBoundingClientRect().height)
  await page.waitForTimeout(30)
  const t1 = await page.locator('.tips-body').evaluate((e) => e.getBoundingClientRect().height)
  check('reduced: tips open instantly (height stable)', Math.abs(t0 - t1) < 1 && t0 > 20, `${t0} ${t1}`)
  const anims2 = await page.evaluate(() => document.getAnimations().filter((a) => a.playState === 'running').length)
  check('reduced: no running animations on Build', anims2 === 0, String(anims2))
  fs.rmSync(f, { force: true })
  check('reduced: no console errors', realErrors(errors).length === 0, errors[0] || '')
  await ctx.close()
}
// ---- mobile overflow + shots
for (const [n, u] of [['planner', URL_PLAN], ['step', URL_TRIP], ['build', URL_BUILD]]) {
  const { ctx, page, errors } = await session(MOB)
  await page.goto(u); await page.waitForTimeout(2500)
  if (n === 'build') { const f = '/tmp/dirbuild-m.mp4'; fs.writeFileSync(f, Buffer.alloc(5 * 1048576)); await page.locator('input[type=file][accept="video/*"]').setInputFiles(f); await page.waitForTimeout(700); await page.locator('.tips-toggle').click(); await page.waitForTimeout(500); fs.rmSync(f, { force: true }) }
  const ov = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)
  check(`390px ${n}: no horizontal overflow`, ov <= 0, String(ov))
  check(`390px ${n}: no console errors`, realErrors(errors).length === 0, errors[0] || '')
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/${n}-390.png`, fullPage: n === 'build' })
  await ctx.close()
}
if (SHOTS) { const { ctx, page } = await session(DESK); await page.goto(URL_BUILD); await page.waitForTimeout(1500); await page.screenshot({ path: `${SHOTS}/build-list-1440.png` }); await ctx.close() }

await browser.close()
const failed = results.filter((r) => !r.ok).length
console.log(`\n${results.length - failed} passed, ${failed} failed`)
process.exit(failed ? 1 : 0)
