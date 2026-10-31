// Live: owner creates a viewer invite, opens the poster, decodes the QR, signs in with it. Env: BASE, OWNER_KEY, OUT (png prefix)
import { chromium } from 'playwright'
import { execFileSync } from 'node:child_process'
const BASE = process.env.BASE, OWNER = process.env.OWNER_KEY, OUT = process.env.OUT || '/tmp/poster'
const res = []
const check = (n, ok, i = '') => { res.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${n}${ok ? '' : '  -> ' + i}`) }
const browser = await chromium.launch()
const ctx = await browser.newContext({ viewport: { width: 1100, height: 900 } })
const page = await ctx.newPage(); const logs = [], reqs = []
page.on('console', (m) => logs.push(m.text())); page.on('request', (r) => reqs.push(r.url()))
await page.goto(BASE + '/'); await page.getByRole('button', { name: /I own this server/ }).click()
await page.locator('form.access-form:not([hidden]) input').last().fill(OWNER)
await page.getByRole('button', { name: 'Save owner key' }).click(); await page.waitForTimeout(1500)
await page.goto(BASE + '/#/build?b=test-two-floor'); await page.getByRole('heading', { name: /Share this building/ }).waitFor({ timeout: 8000 })
await page.getByRole('radio', { name: /^viewer/i }).check()
await page.getByLabel('Label (optional)').fill('Entrance poster live')
await page.getByRole('button', { name: 'Create link' }).click()
const made = await page.getByRole('region', { name: 'New invite link' }).getByLabel('Invite link').inputValue()
check('invite link made', /#\/join\?b=test-two-floor&t=/.test(made))
const token = new URL(made.replace('/#', '/')).searchParams.get('t')
const offer = page.getByRole('button', { name: 'Make a poster' })
await offer.focus(); await page.keyboard.press('Enter')
const dialog = page.getByRole('dialog', { name: /Entrance poster for/ })
await dialog.waitFor({ timeout: 5000 })
await page.waitForSelector('svg.poster-qr path')
check('poster opens via keyboard', true)
check('url has no token / owner key', !page.url().includes(token) && !page.url().includes(OWNER), page.url())
console.log('name:', await page.locator('h1.poster-name').innerText(), '| cta:', await page.locator('.poster-cta').innerText())
check('link text is the join link', (await page.locator('.poster-link').innerText()) === made)
const body = await page.locator('body').innerText()
check('owner key not on poster page', !body.includes(OWNER))
// focus trap
const inside = []
for (let i = 0; i < 6; i++) { await page.keyboard.press('Tab'); inside.push(await page.evaluate(() => { const a = document.activeElement; return a?.closest('.poster-root') ? true : a === document.body ? 'body' : a.tagName })) }
check('Tab never reaches the inert app (poster or browser chrome only)', inside.every((x) => x === true || x === 'body'), JSON.stringify(inside))
// decode
let decoded = await page.evaluate(async () => {
  if (!('BarcodeDetector' in window)) return null
  const svg = document.querySelector('svg.poster-qr'); const clone = svg.cloneNode(true); clone.setAttribute('width', '600'); clone.setAttribute('height', '600')
  const img = new Image(); img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(new XMLSerializer().serializeToString(clone)); await img.decode()
  const c = document.createElement('canvas'); c.width = 600; c.height = 600; c.getContext('2d').drawImage(img, 0, 0, 600, 600)
  const f = await new window.BarcodeDetector({ formats: ['qr_code'] }).detect(c); return f[0]?.rawValue ?? ''
})
let how = 'BarcodeDetector'
if (decoded === null) {
  how = 'opencv'
  await page.locator('svg.poster-qr').screenshot({ path: OUT + '.qr.png' })
  decoded = execFileSync(process.env.PY, ['-I', '-c', 'import cv2,sys;im=cv2.imread(sys.argv[1]);im=cv2.copyMakeBorder(im,40,40,40,40,cv2.BORDER_CONSTANT,value=(255,255,255));print(cv2.QRCodeDetector().detectAndDecode(im)[0])', OUT + '.qr.png']).toString().trim()
}
console.log('decoder:', how)
check('QR decodes to the invite link', decoded === made, decoded.replace(token, '<tok>'))
// print
await page.emulateMedia({ media: 'print' })
const vis = await page.evaluate(() => {
  const els = [...document.body.querySelectorAll('*')].filter((e) => { const r = e.getBoundingClientRect(); const cs = getComputedStyle(e); return r.width > 0 && r.height > 0 && cs.visibility !== 'hidden' && cs.display !== 'none' && e.children.length === 0 && e.textContent.trim() })
  return els.map((e) => e.textContent.trim().slice(0, 40)).filter((t) => !t.startsWith('http'))
})
console.log('visible text in print:', vis)
const out = await page.evaluate(() => ({ nav: document.querySelector('nav.tabbar') ? getComputedStyle(document.querySelector('nav.tabbar')).display : 'absent', root: getComputedStyle(document.getElementById('root')).display }))
check('print: only the poster has visible text', vis.every((t) => /Scan for directions|^[A-Z]/.test(t)) && vis.length <= 6 && !vis.some((t) => /Directions$|Map|Share|Create link/.test(t)), JSON.stringify(vis))
await page.pdf({ path: OUT + '.pdf' }).catch((e) => console.log('pdf err', e.message))
await page.emulateMedia({ media: 'screen' })
await page.screenshot({ path: OUT + '.screen.png' })
await page.keyboard.press('Escape'); await page.waitForTimeout(200)
check('Escape closes poster', (await dialog.count()) === 0)
check('focus back on Make a poster', await page.evaluate(() => document.activeElement?.textContent?.includes('Make a poster')))
// fresh context signs in as viewer with decoded link
const c2 = await browser.newContext(); const p2 = await c2.newPage(); const logs2 = [], reqs2 = []
p2.on('console', (m) => logs2.push(m.text())); p2.on('request', (r) => reqs2.push(r.url()))
await p2.goto(decoded); await p2.waitForSelector('nav.tabbar', { timeout: 10000 })
const tabs = await p2.locator('nav.tabbar button').allInnerTexts()
check('scan link signs in as viewer (Directions only)', tabs.length === 1 && /Directions/.test(tabs[0]), tabs.join('|'))
check('token gone from URL after sign-in', !p2.url().includes(token), p2.url())
check('no token in console/request URLs', logs2.every((l) => !l.includes(token)) && reqs2.every((u) => !u.includes(token)) && logs.every((l) => !l.includes(token) && !l.includes(OWNER)) && reqs.every((u) => !u.includes(token) && !u.includes(OWNER)))
console.log('logs', logs, logs2)
await browser.close()
process.exit(res.every(Boolean) ? 0 : 1)
