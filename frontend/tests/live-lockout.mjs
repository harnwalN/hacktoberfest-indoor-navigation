// Live: lockout shows 429 message in the UI. Env: BASE
import { chromium } from 'playwright'
const browser = await chromium.launch()
const page = await (await browser.newContext()).newPage()
const codes = []; page.on('response', (r) => { if (r.url().includes('/api/')) codes.push(r.status()) })
await page.goto(process.env.BASE + '/')
await page.getByRole('button', { name: /I own this server/ }).click()
let text = ''
for (let i = 0; i < 8; i++) {
  await page.locator('form.access-form:not([hidden]) input').last().fill('bad-owner-key-attempt-number-' + i)
  await page.getByRole('button', { name: 'Save owner key' }).click()
  await page.waitForTimeout(700)
  text = (await page.locator('[role=alert]').allInnerTexts()).join(' | ')
  console.log(i, text)
  if (/wait/i.test(text)) break
}
console.log('codes', codes)
await browser.close()
process.exit(/wait a minute/i.test(text) ? 0 : 1)
