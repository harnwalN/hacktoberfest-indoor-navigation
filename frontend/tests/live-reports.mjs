// Live: reports flow + poster against the REAL backend and built SPA. Env: BASE, OWNER_KEY, REPORTS (number to send for rate test, optional)
import { chromium } from 'playwright'
const BASE = process.env.BASE, OWNER = process.env.OWNER_KEY
const B = 'test-two-floor'
const res = []
const check = (n, ok, i = '') => { res.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${n}${ok ? '' : '  -> ' + i}`) }
const api = async (method, path, body) => {
  const r = await fetch(BASE + '/api' + path, { method, headers: { Authorization: 'Bearer ' + OWNER, 'Content-Type': 'application/json' }, body: body && JSON.stringify(body) })
  return r.json()
}
const mkInvite = async (role, label) => {
  const j = await api('POST', `/buildings/${B}/invites`, { role, label })
  return j
}
const v = await mkInvite('viewer', 'e2e-live-reports-viewer'); const e = await mkInvite('editor', 'e2e-live-reports-editor')
console.log('invite response keys', Object.keys(v))
const link = (j) => j.link || j.url || `${BASE}/#/join?b=${B}&t=${j.token}`
const VL = link(v), EL = link(e)
const VT = v.token || new URL(VL.replace('/#', '/')).searchParams.get('t'), ET = e.token || new URL(EL.replace('/#', '/')).searchParams.get('t')
const browser = await chromium.launch()
const mk = async () => {
  const ctx = await browser.newContext({ viewport: { width: 1100, height: 900 } })
  const page = await ctx.newPage(); const o = { ctx, page, logs: [], reqs: [], resp: [] }
  page.on('console', (m) => o.logs.push(m.text())); page.on('pageerror', (x) => o.logs.push('pageerror ' + x.message))
  page.on('request', (r) => o.reqs.push(r.method() + ' ' + r.url())); page.on('response', (r) => o.resp.push([r.status(), r.url()]))
  return o
}
const pick = async (page, id, text) => { await page.locator('#' + id).click(); await page.locator(`#${id}-list .place-option`, { hasText: text }).first().click() }
const clean = (o, ...s) => s.every((t) => o.logs.every((l) => !l.includes(t)) && o.reqs.every((u) => !u.includes(t)))

// viewer files a report
const V = await mk(); const vp = V.page
await vp.goto(VL); await vp.waitForSelector('nav.tabbar', { timeout: 10000 }); await vp.waitForTimeout(1200)
await pick(vp, 'from', 'Room 101'); await pick(vp, 'to', 'Room 105')
await vp.getByRole('button', { name: /Start directions/ }).click(); await vp.waitForSelector('.route-strip', { timeout: 10000 })
await vp.getByRole('button', { name: 'Next' }).click().catch(() => {})
await vp.waitForTimeout(500)
const stepText = await vp.locator('.step-count').innerText().catch(() => '?')
console.log('step', stepText)
const posts = []; vp.on('request', (r) => { if (r.method() === 'POST' && r.url().includes('/reports')) posts.push(r.postData()) })
await vp.getByRole('button', { name: /Something.s wrong\?/ }).click()
const panel = vp.locator('form[aria-label="Report a problem with this step"]')
await panel.getByRole('radio', { name: 'Directions are wrong' }).check()
await panel.locator('textarea').fill('live e2e: turn is wrong')
await panel.getByRole('button', { name: 'Send' }).click()
await vp.locator('.report-thanks').waitFor({ timeout: 8000 }).catch(() => {})
check('viewer: thanks message', /editors will check/.test(await vp.locator('.report-thanks').innerText().catch(() => '')))
console.log('POST body', posts)
const body = JSON.parse(posts[0] || '{}')
check('report body has kind/note/node_id/step_index', body.kind && body.note && body.node_id && typeof body.step_index === 'number', posts[0])
check('report body has no token', !(posts[0] || '').includes(VT) && !(posts[0] || '').includes(OWNER))
const placeId = body.node_id
const sentNodeLabel = (await api('GET', `/buildings/${B}/map`)).nodes?.find((n) => n.id === placeId)?.label
console.log('node label', sentNodeLabel)
// viewer sees no reports UI and made no GET /reports or /access calls
await vp.goto(BASE + '/#/review?b=' + B).catch(() => {}); await vp.waitForTimeout(1500)
check('viewer: no GET /reports', !V.reqs.some((r) => r.startsWith('GET') && r.includes('/reports')), V.reqs.filter((r) => r.includes('/reports')).join(';'))
check('viewer: no /access calls', !V.reqs.some((r) => r.includes('/access')), V.reqs.filter((r) => r.includes('/access')).join(';'))
check('viewer: no Reports heading/badge', (await vp.getByRole('heading', { name: /^Reports/ }).count()) === 0 && (await vp.locator('.count-badge, .tab-badge').count()) === 0)
check('viewer: no secrets in console/urls', clean(V, VT, OWNER), V.logs.join('|'))

// editor
const E = await mk(); const ep = E.page
await ep.goto(EL); await ep.waitForSelector('nav.tabbar', { timeout: 10000 }); await ep.waitForTimeout(1500)
const tab = ep.locator('nav.tabbar button', { hasText: /^Map/ })
check('editor: tab badge 1', (await ep.locator('.tab-badge').first().innerText().catch(() => '')) === '1', await tab.innerText())
await tab.click(); await ep.getByRole('heading', { name: /^Reports/ }).waitFor({ timeout: 8000 })
const item = ep.getByRole('list', { name: 'Open reports' }).locator('.report-item').first()
const txt = await item.innerText(); console.log('item:', txt.replace(/\n/g, ' | '))
check('editor: place label shown', sentNodeLabel && txt.includes(sentNodeLabel), sentNodeLabel)
check('editor: kind + note shown', txt.includes('Directions are wrong') && txt.includes('turn is wrong'))
check('editor: reporter not a token', !txt.includes(VT))
await item.getByRole('button', { name: 'Show on map' }).click(); await ep.waitForTimeout(800)
const h3 = await ep.locator('.node-editor h3').innerText().catch(() => '')
check('editor: show on map selects the place', sentNodeLabel && h3.includes(sentNodeLabel), h3)
await ep.getByRole('heading', { name: /^Reports/ }).scrollIntoViewIfNeeded()
await ep.getByRole('list', { name: 'Open reports' }).locator('.report-item').first().getByRole('button', { name: 'Mark fixed' }).click()
await ep.getByRole('group', { name: 'Mark as fixed' }).locator('textarea').fill('fixed by live e2e')
await ep.getByRole('button', { name: 'Confirm fixed' }).click(); await ep.waitForTimeout(1500)
check('editor: badge gone after fix', (await ep.locator('.tab-badge').count()) === 0 || (await ep.locator('.tab-badge').first().innerText()) === '0', await tab.innerText())
await ep.getByRole('button', { name: 'Resolved', exact: true }).click(); await ep.waitForTimeout(1000)
check('editor: resolved shows fix note', (await ep.locator('.report-resolution').first().innerText().catch(() => '')) === 'Fix: fixed by live e2e')
await ep.getByRole('button', { name: 'Reopen' }).first().click(); await ep.waitForTimeout(1500)
check('editor: badge back to 1 after reopen', (await ep.locator('.tab-badge').first().innerText().catch(() => '')) === '1')
const open = await api('GET', `/buildings/${B}/reports?status=open`)
check('server: report open again with history', open.length === 1 && open[0].history?.length >= 3, JSON.stringify(open).slice(0, 300))
check('editor: no secrets in console/urls', clean(E, ET, OWNER), E.logs.join('|'))
console.log('editor logs', E.logs, 'viewer logs', V.logs)
await browser.close()
process.exit(res.every(Boolean) ? 0 : 1)
