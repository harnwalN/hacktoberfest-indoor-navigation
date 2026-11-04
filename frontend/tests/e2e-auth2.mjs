// Contributor sign-in (email link), identity only. Run: node tests/e2e-auth2.mjs  (needs `playwright` resolvable and a
// vite dev server at BASE, default http://localhost:5199). The API is mocked with page.route, answering the auth
// routes the way the backend contract says: POST /api/auth/request-link {email} (always 200), POST /api/auth/verify
// {token} -> {session_token, account:{id,email}}, GET /api/auth/me (Bearer) -> {account:{id,email,moderator_of}, is_owner, blocked} or 401 (the real nested shape).
// One scenario answers the older flat shape {id,email,is_owner} to check it still works.
// Covers: request-link fires with the email and shows "check your email", the link token verifies and stores the
// session (and is removed from the URL), the header goes only on the two write calls (walk upload, building
// creation) and never on GET, uploads and building creation prompt sign-in first without a session, a bad link,
// sign out, and an expired session (stored session refused on load, and on a write).
import { chromium } from 'playwright'

const BASE = process.env.BASE || 'http://localhost:5199'
const ORIGIN = new URL(BASE).origin
const EMAIL = 'ada@example.com'
const TOKEN_KEY = 'session_token'
const ACCOUNT_KEY = 'session_account'
const SESSION = 'sess-123'
const CHECK_MAIL = 'Check your email for a sign-in link'
const NEED_UPLOAD = 'Sign in to upload a walk'
const BAD_LINK = 'That sign-in link is invalid or has expired'
const ENDED = 'Your sign-in ended'

const MAP = { building_id: 'b1', name: 'Hall B', nodes: [], edges: [], suggested_merges: [], videos: [] }

const results = []
const check = (name, ok, info = '') => {
  results.push({ name, ok: !!ok, info })
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok ? '' : '  -> ' + info}`)
}
const realErrors = (errs) => errs.filter((e) => !/Failed to load resource/.test(e))
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

const browser = await chromium.launch()

// One browser context per scenario. opts: { stored, uploadStatus, sessionValid, owner, flatMe } (owner: the account is the
// owner; flatMe: /auth/me answers the flat shape).
async function session(opts = {}) {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 } })
  const page = await ctx.newPage()
  const errors = []
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
  page.on('pageerror', (e) => errors.push('PAGEERROR ' + e.message))
  const st = {
    requests: [], // { method, path, auth, body }
    uploadStatus: 200,
    sessionValid: true,
    ...opts,
  }
  await ctx.route('**/api/**', async (route) => {
    const req = route.request()
    const url = new URL(req.url())
    const p = url.pathname.replace(/^\/api/, '')
    const m = req.method()
    const auth = req.headers()['authorization'] || null
    let body = null
    try {
      body = req.postDataJSON()
    } catch {
      body = null
    }
    st.requests.push({ method: m, path: p, auth, body })
    const json = (b, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(b) })

    if (p === '/health') return json({ gemma: { models: [] } })
    if (p === '/buildings' && m === 'GET') return json([{ id: 'b1', name: 'Hall B' }])
    if (p === '/buildings' && m === 'POST') {
      if (auth !== `Bearer ${SESSION}`) return json({ detail: 'sign in' }, 401)
      return json({ building_id: 'b2' }, 201)
    }
    if (p === '/buildings/b1/map') return json(MAP)
    if (p === '/buildings/b1/queue') return json([])
    if (p === '/buildings/b1/videos' && m === 'POST') {
      if (st.uploadStatus !== 200) return json({ detail: 'sign in' }, st.uploadStatus)
      return json({ job_id: 'j1' }, 202)
    }
    if (p === '/auth/request-link' && m === 'POST') return json({ message: 'If that address can sign in, a link is on its way.' })
    if (p === '/auth/verify' && m === 'POST') {
      if (body?.token === 'good') return json({ session_token: SESSION, account: { id: 'acc1', email: EMAIL } })
      return json({ detail: 'invalid or expired' }, 400)
    }
    if (p === '/auth/me' && m === 'GET') {
      if (st.sessionValid && auth === `Bearer ${SESSION}`) {
        const acct = { id: 'acc1', email: EMAIL, moderator_of: [] }
        return json(st.flatMe ? { ...acct, is_owner: !!st.owner } : { account: acct, is_owner: !!st.owner, blocked: false })
      }
      return json({ detail: 'sign in' }, 401)
    }
    if (p.startsWith('/jobs/')) return json({ id: 'j1', status: 'queued', stages: [], log: [] })
    return route.fulfill({ status: 404, body: '{}' })
  })
  if (opts.stored) {
    await page.addInitScript(([k, v, a, owner]) => {
      if (!sessionStorage.getItem('seeded')) {
        sessionStorage.setItem('seeded', '1')
        localStorage.setItem(k, v)
        localStorage.setItem(a, JSON.stringify({ id: 'acc1', email: 'ada@example.com', is_owner: owner }))
      }
    }, [TOKEN_KEY, opts.stored, ACCOUNT_KEY, !!opts.owner])
  }
  return { ctx, page, errors, st }
}

const dialog = (page) => page.getByRole('dialog', { name: 'Sign in' })
const chipEmail = (page) => page.locator('.account-email')
const signInBtn = (page) => page.locator('.account-btn', { hasText: 'Sign in' })
const storedToken = (page) => page.evaluate((k) => localStorage.getItem(k), TOKEN_KEY)
const sendAuthOf = (st, path) => st.requests.filter((r) => r.path === path)
// The Contribute tab (id build) is where a contributor uploads. Building creation is on the owner page now.
const pickBuildTab = async (page) => {
  await page.goto(`${BASE}/#/build?b=b1`)
  await page.locator('input[type=file][accept="video/*"]').waitFor({ timeout: 5000 })
}

// 1. Browsing stays anonymous; the sign-in button is there; no auth call without a stored session.
{
  const { ctx, page, errors, st } = await session()
  await page.goto(`${BASE}/#/navigate?b=b1`)
  await signInBtn(page).waitFor({ timeout: 5000 })
  check('anon: the Sign in button shows with no session', (await signInBtn(page).count()) === 1)
  await sleep(300)
  check('anon: no /auth/me call without a stored session', sendAuthOf(st, '/auth/me').length === 0, JSON.stringify(st.requests))
  const gets = st.requests.filter((r) => r.method === 'GET')
  check('anon: GET requests carry no Authorization header', gets.length > 0 && gets.every((r) => !r.auth), JSON.stringify(gets.slice(0, 3)))
  check('anon: no page errors', realErrors(errors).length === 0, realErrors(errors).join('|').slice(0, 200))
  await ctx.close()
}

// 2. Sign-in sheet: validation, request-link call, "check your email".
{
  const { ctx, page, errors, st } = await session()
  await page.goto(`${BASE}/#/navigate?b=b1`)
  await signInBtn(page).click()
  // The header Sign in is a link to the sign-in page now (S4): no sheet opens from the header.
  await page.waitForURL(/#\/sign-in$/, { timeout: 3000 })
  await page.getByRole('heading', { name: /Welcome to/ }).waitFor({ timeout: 3000 })
  check('request: header Sign in goes to the sign-in page, not the sheet', (await dialog(page).count()) === 0 && (await page.getByRole('heading', { name: /Welcome to/ }).count()) === 1)
  await page.getByLabel('Email').fill('not-an-email')
  await page.getByRole('button', { name: 'Email me a link' }).click()
  await page.getByText('Enter a valid email address.').waitFor({ timeout: 3000 })
  check('request: a bad address is refused client side', sendAuthOf(st, '/auth/request-link').length === 0)
  await page.getByLabel('Email').fill(`  ${EMAIL} `)
  await page.getByRole('button', { name: 'Email me a link' }).click()
  await page.getByText(CHECK_MAIL).waitFor({ timeout: 3000 })
  // S1: after a successful send, focus moves to the sent-note heading.
  check('request: focus moves to the sent-note heading', await page.evaluate(() => document.activeElement?.tagName === 'H2' && document.activeElement.textContent.includes('Check your email')))
  const req = sendAuthOf(st, '/auth/request-link')
  check('request: POST /auth/request-link fired once with the trimmed email', req.length === 1 && req[0].body?.email === EMAIL, JSON.stringify(req))
  check('request: the request carries no Authorization header', req.length === 1 && !req[0].auth)
  check('request: the sheet says to check email, not signed in yet', (await chipEmail(page).count()) === 0 && (await storedToken(page)) === null)
  check('request: no page errors', realErrors(errors).length === 0, realErrors(errors).join('|').slice(0, 200))
  await ctx.close()
}

// 3. No session: the Contribute tab asks to sign in (no upload form), and the owner page (building creation) is not allowed.
{
  const { ctx, page, errors, st } = await session()
  await page.goto(`${BASE}/#/build?b=b1`)
  await page.getByRole('button', { name: 'Sign in to contribute' }).waitFor({ timeout: 5000 })
  check('gate: signed out, Contribute shows the sign-in prompt and no upload form', (await page.locator('input[type=file]').count()) === 0)
  await page.getByRole('button', { name: 'Sign in to contribute' }).click()
  await dialog(page).waitFor({ timeout: 3000 })
  check('gate: the prompt opens sign-in with its reason', (await page.getByText('Sign in to contribute a walk').count()) === 1)
  await page.keyboard.press('Escape')
  await dialog(page).waitFor({ state: 'detached', timeout: 3000 }).catch(() => {})
  check('gate: Escape closes the sheet', (await dialog(page).count()) === 0)
  await page.goto(`${BASE}/#/owner?s=buildings&b=b1`)
  await page.getByText('This page is not allowed').waitFor({ timeout: 3000 })
  check('gate: signed out, the owner page (building creation) is not allowed', (await page.getByText('This page is not allowed').count()) === 1)
  await sleep(200)
  check('gate: no video POST and no building POST without a session', sendAuthOf(st, '/buildings/b1/videos').length === 0 && st.requests.filter((r) => r.method === 'POST' && r.path === '/buildings').length === 0)
  check('gate: no page errors', realErrors(errors).length === 0, realErrors(errors).join('|').slice(0, 200))
  await ctx.close()
}

// 4. A good link: verify stores the session, the token leaves the URL, the chip shows the email, and a reload keeps it.
{
  const { ctx, page, errors, st } = await session()
  await page.goto(`${BASE}/?token=good#/navigate?b=b1`)
  await page.locator('.account-email').waitFor({ timeout: 5000 })
  const v = sendAuthOf(st, '/auth/verify')
  check('link: POST /auth/verify fired once with the token', v.length === 1 && v[0].body?.token === 'good', JSON.stringify(v))
  check('link: the session is stored under session_token', (await storedToken(page)) === SESSION, String(await storedToken(page)))
  const acct = await page.evaluate((k) => localStorage.getItem(k), ACCOUNT_KEY)
  check('link: the account is stored under session_account', acct && JSON.parse(acct).email === EMAIL, acct)
  check('link: the token is gone from the address bar', !(await page.url()).includes('token='), await page.url())
  check('link: the app stays on its tab and building', (await page.url()).includes('#/navigate?b=b1'), await page.url())
  check('link: the chip shows the signed-in email', (await chipEmail(page).textContent()) === EMAIL)
  check('link: the Sign out button shows', (await page.getByRole('button', { name: 'Sign out' }).count()) === 1)

  await page.reload()
  await chipEmail(page).waitFor({ timeout: 5000 })
  const me = sendAuthOf(st, '/auth/me')
  check('reload: /auth/me checks the stored session with the bearer header', me.length >= 1 && me.every((r) => r.auth === `Bearer ${SESSION}`), JSON.stringify(me))
  check('reload: still signed in', (await chipEmail(page).textContent()) === EMAIL)
  check('no page errors', realErrors(errors).length === 0, realErrors(errors).join('|').slice(0, 200))
  await ctx.close()
}

// 4b. A stored session on reload: the chip shows the email from the nested /auth/me, and the flat shape still works.
for (const flatMe of [false, true]) {
  const { ctx, page, errors } = await session({ stored: SESSION, flatMe })
  await page.goto(`${BASE}/#/navigate?b=b1`)
  await chipEmail(page).waitFor({ timeout: 5000 }).catch(() => {})
  await sleep(300)
  const label = flatMe ? 'flat shape' : 'nested shape'
  check(`reload (${label}): the chip shows the signed-in email`, (await chipEmail(page).textContent()) === EMAIL, await chipEmail(page).textContent().catch(() => 'no chip'))
  const stored = JSON.parse((await page.evaluate((k) => localStorage.getItem(k), ACCOUNT_KEY)) || 'null')
  check(`reload (${label}): stored account is flat with email and is_owner`, stored?.email === EMAIL && stored?.is_owner === false && stored?.account === undefined, JSON.stringify(stored))
  check(`reload (${label}): no page errors`, realErrors(errors).length === 0, realErrors(errors).join('|').slice(0, 200))
  await ctx.close()
}

// 5. Signed in: the upload and building creation carry the header; GETs still do not.
{
  const { ctx, page, errors, st } = await session({ stored: SESSION })
  await pickBuildTab(page)
  await chipEmail(page).waitFor({ timeout: 5000 })
  await page.locator('input[type=file][accept="video/*"]').setInputFiles({ name: 'walk.mp4', mimeType: 'video/mp4', buffer: Buffer.from('x') })
  await page.getByRole('button', { name: /upload/i }).first().click()
  await sleep(500)
  const up = sendAuthOf(st, '/buildings/b1/videos')
  check('upload: the video POST carries Authorization: Bearer <session>', up.length === 1 && up[0].auth === `Bearer ${SESSION}`, JSON.stringify(up))
  check('upload: no sign-in sheet when signed in', (await dialog(page).count()) === 0)
  // The building list is the one GET that carries the bearer (the owner sees every building); other GETs carry none.
  // /me/uploads is the signed-in contributor's own list, so it carries the bearer too.
  const gets = st.requests.filter((r) => r.method === 'GET' && !r.path.startsWith('/auth/') && r.path !== '/buildings' && r.path !== '/me/uploads')
  check('signed in: GET requests still carry no Authorization header', gets.every((r) => !r.auth), JSON.stringify(gets.slice(0, 3)))
  check('signed in: no page errors', realErrors(errors).length === 0, realErrors(errors).join('|').slice(0, 200))
  await ctx.close()
}

// 5b. Owner: building creation carries the bearer header (it is the owner's, on the owner page).
{
  const { ctx, page, errors, st } = await session({ stored: SESSION, owner: true })
  await page.goto(`${BASE}/#/owner?s=buildings&b=b1`)
  await page.getByPlaceholder('e.g. Engineering Hall').waitFor({ timeout: 5000 })
  await page.getByPlaceholder('e.g. Engineering Hall').fill('New Hall')
  await page.getByRole('button', { name: 'Create', exact: true }).click()
  await sleep(500)
  const bc = st.requests.filter((r) => r.method === 'POST' && r.path === '/buildings')
  check('create: the building POST carries the bearer header', bc.length === 1 && bc[0].auth === `Bearer ${SESSION}`, JSON.stringify(bc))
  check('create: no sign-in sheet when the owner creates', (await dialog(page).count()) === 0)
  check('signed in owner: no page errors', realErrors(errors).length === 0, realErrors(errors).join('|').slice(0, 200))
  await ctx.close()
}

// 6. Expired session: refused on load (cleared, no sheet), and a write refused with 401 opens sign-in again.
{
  const { ctx, page } = await session({ stored: 'old-token', sessionValid: false })
  await page.goto(`${BASE}/#/navigate?b=b1`)
  await signInBtn(page).waitFor({ timeout: 5000 })
  await sleep(300)
  check('expired on load: the stored session is cleared', (await storedToken(page)) === null)
  check('expired on load: the chip falls back to Sign in and no sheet opens', (await dialog(page).count()) === 0)
  await ctx.close()
}
{
  const { ctx, page, st } = await session({ stored: SESSION, uploadStatus: 401 })
  await pickBuildTab(page)
  await chipEmail(page).waitFor({ timeout: 5000 })
  await page.locator('input[type=file][accept="video/*"]').setInputFiles({ name: 'walk.mp4', mimeType: 'video/mp4', buffer: Buffer.from('x') })
  await page.getByRole('button', { name: /upload/i }).first().click()
  await dialog(page).waitFor({ timeout: 3000 })
  check('expired on write: the sheet says the sign-in ended', (await page.getByText(ENDED).count()) === 1)
  check('expired on write: the stored session is cleared', (await storedToken(page)) === null)
  check('expired on write: the upload was sent with the old header', sendAuthOf(st, '/buildings/b1/videos')[0]?.auth === `Bearer ${SESSION}`)
  await ctx.close()
}

// 7. Bad link: the sheet explains it, and no session is stored.
{
  const { ctx, page, errors } = await session()
  await page.goto(`${BASE}/?token=stale#/navigate?b=b1`)
  await dialog(page).waitFor({ timeout: 5000 })
  check('bad link: the sheet says the link is invalid or expired', (await page.getByText(BAD_LINK).count()) === 1)
  check('bad link: no session is stored', (await storedToken(page)) === null)
  check('bad link: the token is removed from the address bar', !(await page.url()).includes('token='))
  check('bad link: no page errors', realErrors(errors).length === 0, realErrors(errors).join('|').slice(0, 200))
  await ctx.close()
}

// 8. Sign out clears the session; the next write prompts again.
{
  const { ctx, page, st } = await session({ stored: SESSION })
  await page.goto(`${BASE}/#/navigate?b=b1`)
  await chipEmail(page).waitFor({ timeout: 5000 })
  await page.getByRole('button', { name: 'Sign out' }).click()
  await signInBtn(page).waitFor({ timeout: 3000 })
  check('sign out: the session is cleared', (await storedToken(page)) === null)
  check('sign out: the account is cleared', (await page.evaluate((k) => localStorage.getItem(k), ACCOUNT_KEY)) === null)
  check('sign out: the chip shows Sign in again', (await signInBtn(page).count()) === 1)
  await page.goto(`${BASE}/#/owner?s=buildings&b=b1`)
  await page.getByText('This page is not allowed').waitFor({ timeout: 3000 })
  check('sign out: the owner page is not allowed again', (await page.getByText('This page is not allowed').count()) === 1)
  check('sign out: no building POST after sign out', st.requests.filter((r) => r.method === 'POST' && r.path === '/buildings').length === 0)
  await ctx.close()
}

await browser.close()
const failed = results.filter((r) => !r.ok)
console.log(`\n${results.length - failed.length} passed, ${failed.length} failed of ${results.length}`)
process.exit(failed.length ? 1 : 0)
