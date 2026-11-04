// Live: a map edit saved through the UI, no sign-in. Env: BASE
import { chromium } from 'playwright'
const BASE = process.env.BASE
const browser = await chromium.launch()
const page = await (await browser.newContext({ viewport: { width: 1000, height: 900 } })).newPage()
const logs = []; page.on('console', (m) => logs.push(m.text()))
const puts = []; page.on('response', (r) => { if (r.request().method() === 'PUT') puts.push([r.status(), r.url()]) })
await page.goto(BASE + '/#/review?b=test-two-floor'); await page.waitForSelector('nav.tabbar'); await page.waitForTimeout(2000)
await page.locator('.node-card', { hasText: 'Room 101' }).first().click()
const inp = page.locator('input[value="Room 101"]').first()
await inp.fill('Room 101 edited-by-e2e')
await page.getByRole('button', { name: 'Save changes' }).click()
await page.waitForTimeout(1500)
console.log('PUT', puts, 'saved-label', await page.getByRole('button', { name: /Saved|Save changes/ }).innerText())
console.log('logs', logs)
await browser.close()
process.exit(puts.length && puts[0][0] === 200 ? 0 : 1)
