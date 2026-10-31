// Live run: viewer + editor + owner revoke against REAL backend. Env: BASE, OWNER_KEY, LINKS (json from live-owner.mjs)
import { chromium } from 'playwright'
import fs from 'node:fs'
const BASE = process.env.BASE, OWNER = process.env.OWNER_KEY
const links = JSON.parse(fs.readFileSync(process.env.LINKS, 'utf8'))
const tok = (l) => new URL(l.replace('/#', '/')).searchParams.get('t')
const VT = tok(links.viewer), ET = tok(links.editor)
const res = []
const check = (n, ok, i = '') => { res.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${n}${ok ? '' : '  -> ' + i}`) }
const browser = await chromium.launch()
const mk = async (name) => {
  const ctx = await browser.newContext({ viewport: { width: 1000, height: 900 } })
  const page = await ctx.newPage()
  const o = { ctx, page, logs: [], reqs: [], resp: [] }
  page.on('console', (m) => o.logs.push(m.text()))
  page.on('pageerror', (e) => o.logs.push('pageerror ' + e.message))
  page.on('request', (r) => o.reqs.push(r.url()))
  page.on('response', (r) => o.resp.push([r.status(), r.url()]))
  return o
}
const secretFree = (o, ...s) => s.every((t) => o.logs.every((l) => !l.includes(t)) && o.reqs.every((u) => !u.includes(t)))
const pick = async (page, id, text) => {
  await page.locator('#' + id).click()
  await page.locator(`#${id}-list .place-option`, { hasText: text }).first().click()
}
// ---------- viewer ----------
const v = await mk('viewer'); const vp = v.page
await vp.goto(links.viewer)
await vp.waitForSelector('nav.tabbar', { timeout: 10000 })
await vp.waitForTimeout(1500)
const vurl = vp.url()
check('viewer: token gone from URL', !vurl.includes(VT) && !vurl.includes('t='), vurl)
const vtabs = await vp.locator('nav.tabbar button').allInnerTexts()
check('viewer: Directions only', vtabs.length === 1 && /Directions/.test(vtabs[0]), vtabs.join('|'))
await vp.screenshot({ path: process.env.LINKS + '.viewer1.png' })
await pick(vp, 'from', 'Room 101')
await pick(vp, 'to', 'Room 224')
await vp.getByRole('button', { name: /Start directions/ }).click()
await vp.waitForSelector('.route-strip', { timeout: 10000 })
await vp.waitForTimeout(1500)
const badge1 = await vp.locator('.route-badge').allInnerTexts()
check('viewer: cross-floor route shows stairs badge', badge1.some((t) => /uses stairs/.test(t)), JSON.stringify(badge1))
const imgs = await vp.locator('img').evaluateAll((els) => els.map((e) => ({ src: e.getAttribute('src'), ok: e.complete && e.naturalWidth > 0 })))
console.log('imgs', imgs.length, imgs.slice(0, 2))
check('viewer: photos load with signed URLs', imgs.length > 0 && imgs.every((i) => i.ok) && imgs.every((i) => /\/bfiles\/.*\?exp=\d+&sig=[0-9a-f]+/.test(i.src)), JSON.stringify(imgs.slice(0, 3)))
await vp.screenshot({ path: process.env.LINKS + '.viewer2.png' })
const photoUrls = [...new Set(imgs.map((i) => i.src).filter((s) => s && s.includes('/bfiles/')))]
fs.writeFileSync(process.env.LINKS + '.photos', JSON.stringify(photoUrls))
// avoid stairs toggle
await vp.getByRole('button', { name: 'End directions' }).click()
await vp.waitForTimeout(500)
await vp.locator('#avoid-stairs').check()
await pick(vp, 'from', 'Room 101')
await pick(vp, 'to', 'Room 224')
await vp.getByRole('button', { name: /Start directions/ }).click()
await vp.waitForTimeout(2000)
const errTxt = await vp.locator('[role=alert], .notice.error, .route-error').allInnerTexts()
console.log('avoid error:', errTxt)
check('viewer: avoid-stairs shows step-free message', errTxt.some((t) => /No step-free route/.test(t)), JSON.stringify(errTxt))
check('viewer: "Show route with stairs" retry present', await vp.getByRole('button', { name: 'Show route with stairs' }).isVisible())
await vp.screenshot({ path: process.env.LINKS + '.viewer3.png' })
await vp.getByRole('button', { name: 'Show route with stairs' }).click()
await vp.waitForSelector('.route-strip', { timeout: 10000 })
check('viewer: retry shows stairs route', true)
check('viewer: toggle synced off after retry', !(await vp.locator('#avoid-stairs').isChecked().catch(() => false)))
// same-floor route with avoid on => step-free badge
await vp.getByRole('button', { name: 'End directions' }).click()
await vp.locator('#avoid-stairs').check()
await pick(vp, 'from', 'Room 101')
await pick(vp, 'to', 'Room 105')
await vp.getByRole('button', { name: /Start directions/ }).click()
await vp.waitForSelector('.route-strip', { timeout: 10000 })
await vp.waitForTimeout(800)
const badge2 = await vp.locator('.route-badge').allInnerTexts()
check('viewer: same-floor step-free badge', badge2.some((t) => /No stairs/.test(t)), JSON.stringify(badge2))
check('viewer: link has avoid=1', /avoid=1/.test(vp.url()), vp.url())
// viewer must not reach edit tabs by hash
await vp.goto(BASE + '/#/build?b=test-two-floor'); await vp.waitForTimeout(1200)
check('viewer: #/build falls back to Directions', !(await vp.getByText('Share this building').isVisible()) && /navigate/.test(vp.url()), vp.url())
check('viewer: no secrets in URLs/console', secretFree(v, VT, ET, OWNER), JSON.stringify(v.logs))
// ---------- editor ----------
const e = await mk('editor'); const ep = e.page
await ep.goto(links.editor)
await ep.waitForSelector('nav.tabbar', { timeout: 10000 }); await ep.waitForTimeout(1500)
check('editor: token gone from URL', !ep.url().includes(ET), ep.url())
const etabs = await ep.locator('nav.tabbar button').allInnerTexts()
check('editor: 4 tabs', etabs.length === 4, etabs.join('|'))
await ep.goto(BASE + '/#/build?b=test-two-floor'); await ep.waitForTimeout(1500)
check('editor: Share absent', !(await ep.getByText('Share this building').isVisible()))
check('editor: New building absent', !(await ep.getByText('New building').isVisible()))
// map edit via UI: open Map tab
await ep.goto(BASE + '/#/review?b=test-two-floor'); await ep.waitForTimeout(2500)
await ep.screenshot({ path: process.env.LINKS + '.editor1.png' })
console.log('editor map buttons:', (await ep.locator('main button').allInnerTexts()).slice(0, 40))
fs.writeFileSync(process.env.LINKS + '.state', JSON.stringify({ vtabs, etabs }))
// ---------- owner revokes viewer ----------
const o = await mk('owner'); const op = o.page
await op.goto(BASE + '/')
await op.getByRole('button', { name: /I own this server/ }).click()
await op.locator('form.access-form:not([hidden]) input').last().fill(OWNER)
await op.getByRole('button', { name: 'Save owner key' }).click()
await op.waitForTimeout(1500)
await op.goto(BASE + '/#/build?b=test-two-floor'); await op.waitForTimeout(1500)
await op.getByRole('button', { name: 'Revoke e2e-viewer' }).click()
await op.getByRole('button', { name: /Yes, revoke/ }).click()
await op.waitForTimeout(1200)
const row = await op.locator('.invite-row', { hasText: 'e2e-viewer' }).innerText()
check('owner: viewer invite shows Revoked', /Revoked/.test(row), row)
// viewer next action
await vp.goto(BASE + '/#/navigate?b=test-two-floor'); await vp.waitForTimeout(1500)
await vp.reload(); await vp.waitForTimeout(2000)
await vp.screenshot({ path: process.env.LINKS + '.viewer4.png' })
const acc = await vp.locator('.access-screen').count()
check('viewer after revoke: access screen shown', acc > 0, await vp.locator('main').innerText().catch(() => ''))
// old photo urls
const origin = new URL(BASE).origin
const codes = []
for (const u of photoUrls.slice(0, 5)) codes.push((await fetch(origin + u)).status)
console.log('old photo status after revoke', codes)
check('old viewer photo URLs return 404', codes.length > 0 && codes.every((c) => c === 404), codes.join())
// editor photos still ok (different invite, same building: all building URLs invalidated by design)
check('owner: no secrets', secretFree(o, OWNER, VT, ET))
check('editor: no secrets', secretFree(e, VT, ET, OWNER), JSON.stringify(e.logs))
console.log('viewer logs', v.logs.slice(0, 8)); console.log('editor logs', e.logs.slice(0, 8))
await browser.close()
process.exit(res.every(Boolean) ? 0 : 1)
