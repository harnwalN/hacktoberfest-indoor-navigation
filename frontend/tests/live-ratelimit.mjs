// Live: 21st report in an hour from one invite shows the try-again-later message. Env: BASE, OWNER_KEY
import { chromium } from 'playwright'
const BASE = process.env.BASE, OWNER = process.env.OWNER_KEY, B = 'test-two-floor'
const j = await (await fetch(`${BASE}/api/buildings/${B}/invites`, { method: 'POST', headers: { Authorization: 'Bearer ' + OWNER, 'Content-Type': 'application/json' }, body: JSON.stringify({ role: 'viewer', label: 'e2e-ratelimit' }) })).json()
const codes = []
for (let i = 0; i < 20; i++) {
  const r = await fetch(`${BASE}/api/buildings/${B}/reports`, { method: 'POST', headers: { Authorization: 'Bearer ' + j.token, 'Content-Type': 'application/json' }, body: JSON.stringify({ kind: 'other', note: 'rl ' + i }) })
  codes.push(r.status)
}
console.log('first 20 statuses', [...new Set(codes)])
const browser = await chromium.launch(); const page = await (await browser.newContext()).newPage()
const logs = []; page.on('console', (m) => logs.push(m.text()))
const st = []; page.on('response', (r) => { if (r.request().method() === 'POST' && r.url().includes('/reports')) st.push([r.status(), r.headers()['retry-after']]) })
await page.goto(`${BASE}/${j.join_path}`); await page.waitForSelector('nav.tabbar')
const pick = async (id, text) => { await page.locator('#' + id).click(); await page.locator(`#${id}-list .place-option`, { hasText: text }).first().click() }
await pick('from', 'Room 101'); await pick('to', 'Room 105')
await page.getByRole('button', { name: /Start directions/ }).click(); await page.waitForSelector('.route-strip')
await page.getByRole('button', { name: /Something.s wrong\?/ }).click()
const panel = page.locator('form[aria-label="Report a problem with this step"]')
await panel.getByRole('radio', { name: 'Something else' }).check(); await panel.locator('textarea').fill('21st')
await panel.getByRole('button', { name: 'Send' }).click()
await panel.locator('[role=alert]').first().waitFor({ timeout: 6000 }).catch(() => {})
const msg = await panel.locator('[role=alert]').first().innerText().catch(() => '')
console.log('ui:', msg, st, logs)
await browser.close()
process.exit(codes.every((c) => c === 200 || c === 201) && /try again later/i.test(msg) && st[0]?.[0] === 429 ? 0 : 1)
