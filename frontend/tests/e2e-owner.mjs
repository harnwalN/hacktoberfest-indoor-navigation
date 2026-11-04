// Visitor / contributor / owner split. Run: BASE=http://127.0.0.1:5410 node tests/e2e-owner.mjs
// (needs `playwright` resolvable and a vite dev server at BASE). The API is mocked with page.route; the mock answers
// /auth/me by the bearer token: "tok-owner" is the owner, "tok-member" a signed-in contributor.
// Covers: visibility of tabs and tools per viewer (signed out, contributor, owner); the contributor's Contribute page
// and My uploads chips; the owner page (moderation approve, reject with rollback, Videos & map, buildings create and
// delete through the in-page dialog, users block / unblock, reports); the not-allowed screen.
import { chromium } from 'playwright'

const BASE = process.env.BASE || 'http://localhost:5199'
const OWNER = { id: 'u-owner', email: 'owner@example.com', is_owner: true }
const MEMBER = { id: 'u-member', email: 'walker@example.com', is_owner: false }

const NODES = [
  { id: 'n1', label: 'Lobby', floor: 1, type: 'entrance', x: 0, y: 0, verified: true, texts: [], sign_texts: [], description: 'Main lobby' },
  { id: 'n2', label: 'Room 101', floor: 1, type: 'room', x: 4, y: 0, verified: false, texts: [], sign_texts: [] },
]
const EDGES = [{ id: 'e1', from_node: 'n1', to_node: 'n2', heading_out: 0, heading_in: 0, vertical: 'none', frames_fwd: [], frames_rev: null }]
const MAP = () => ({ building_id: 'b1', name: 'Hall B', nodes: NODES.map((n) => ({ ...n })), edges: EDGES.map((e) => ({ ...e })), suggested_merges: [], videos: [] })

const MOD_ITEMS = [
  { building_id: 'b1', building_name: 'Hall B', video_id: 'v1', filename: 'lobby-walk.mp4', uploader_email: 'walker@example.com', status: 'pending', reason: 'Possible face in frame 3', frames: ['/fake/f1.jpg', '/fake/f2.jpg', '/fake/f3.jpg', '/fake/f4.jpg', '/fake/f5.jpg', '/fake/f6.jpg', '/fake/f7.jpg'], created_at: '2026-11-03T10:00:00Z' },
  { building_id: 'b1', building_name: 'Hall B', video_id: 'v2', filename: 'stairs.mp4', uploader_email: 'other@example.com', status: 'pending', reason: '', frames: [], created_at: '2026-11-03T11:00:00Z' },
]

// A 1x1 PNG for the moderation frames (served by the mock, so the thumbnails really load).
const PNG = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=', 'base64')

const results = []
const check = (name, ok, info = '') => {
  results.push({ name, ok: !!ok, info })
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok ? '' : '  -> ' + info}`)
}
const realErrors = (errs) => errs.filter((e) => !/Failed to load resource/.test(e))
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

const browser = await chromium.launch()

// viewer: 'anon' | 'member' | 'owner'. opts: { moderateFail, blockFail, deleteFail, moderationStatus }.
async function session(viewer, opts = {}, hash = '#/navigate?b=b1') {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } })
  if (viewer !== 'anon') {
    const acct = viewer === 'owner' ? OWNER : MEMBER
    const token = viewer === 'owner' ? 'tok-owner' : 'tok-member'
    await ctx.addInitScript(([t, a]) => {
      if (!localStorage.getItem('session_token')) {
        localStorage.setItem('session_token', t)
        localStorage.setItem('session_account', JSON.stringify(a))
      }
    }, [token, acct])
  }
  await ctx.route('**/fake/**', (route) => route.fulfill({ status: 200, contentType: 'image/png', body: PNG }))
  const page = await ctx.newPage()
  const errors = []
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
  page.on('pageerror', (e) => errors.push('PAGEERROR ' + e.message))
  page.on('dialog', (d) => d.accept())
  const st = {
    requests: [],
    moderation: MOD_ITEMS.map((i) => ({ ...i })),
    users: [
      { email: 'walker@example.com', uploads: 3, live: 1, rejected: 1, blocked: false, last_seen: '2026-11-02T09:00:00Z' },
      { email: 'other@example.com', uploads: 1, live: 0, rejected: 0, blocked: false, last_seen: null },
    ],
    buildings: [
      { id: 'b1', name: 'Hall B', nodes: 2, has_map: true },
      { id: 'b2', name: 'Annex', nodes: 0, has_map: false },
    ],
    ...opts,
  }
  // Visitors only see buildings with a map (contract); owner sees all.
  const listFor = (viewerToken) => (viewerToken === 'tok-owner' ? st.buildings : st.buildings.filter((b) => b.has_map))
  await ctx.route('**/api/**', async (route) => {
    const req = route.request()
    const url = new URL(req.url())
    const p = url.pathname.replace(/^\/api/, '')
    const m = req.method()
    const auth = req.headers()['authorization'] || null
    const token = auth ? auth.replace('Bearer ', '') : null
    let body = null
    try { body = req.postDataJSON() } catch { body = null }
    st.requests.push({ method: m, path: p, query: url.search, auth, body })
    const json = (b, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(b) })
    const forbid = () => json({ detail: 'Owner only' }, 403)

    if (p === '/health') return json({ gemma: { models: [], ok: false, error: 'off' } })
    if (p === '/auth/me') {
      const nest = (a) => ({ account: { id: a.id, email: a.email }, is_owner: !!a.is_owner, blocked: false })
      if (token === 'tok-owner') return json(nest(OWNER))
      if (token === 'tok-member') return json(nest(MEMBER))
      return json({ detail: 'no' }, 401)
    }
    if (p === '/buildings' && m === 'GET') return json(listFor(token))
    if (p === '/buildings' && m === 'POST') {
      if (token !== 'tok-owner') return forbid()
      const b = { id: 'b3', name: body?.name || 'New', nodes: 0, has_map: false }
      st.buildings.push(b)
      return json({ building_id: 'b3' }, 201)
    }
    const bm = /^\/buildings\/([^/]+)$/.exec(p)
    if (bm && m === 'DELETE') {
      if (token !== 'tok-owner') return forbid()
      if (st.deleteFail) return json({ detail: 'Delete failed on the server' }, 500)
      st.buildings = st.buildings.filter((b) => b.id !== bm[1])
      return route.fulfill({ status: 204, body: '' })
    }
    // Any building id: the owner can switch to a building created during the test.
    const bmap = /^\/buildings\/([^/]+)\/map$/.exec(p)
    if (bmap && m === 'GET') return json({ ...MAP(), building_id: bmap[1], name: st.buildings.find((b) => b.id === bmap[1])?.name || bmap[1] })
    if (/^\/buildings\/[^/]+\/queue$/.test(p)) return json([])
    if (/^\/buildings\/[^/]+\/reports$/.test(p) && m === 'GET') return json([{ id: 'r1', node_id: 'n2', kind: 'wrong_direction', status: 'open', note: 'Wrong turn here', node_label: 'Room 101', created_at: '2026-11-03T08:00:00Z' }])
    if (p === '/buildings/b1/videos' && m === 'POST') {
      if (!token) return json({ detail: 'sign in' }, 401)
      return json({ job_id: 'j1' }, 202)
    }
    if (p === '/me/uploads') {
      if (!token) return json({ detail: 'sign in' }, 401)
      return json([
        { building_id: 'b1', building_name: 'Hall B', video_id: 'u1', filename: 'walk-a.mp4', status: 'processing', created_at: '2026-11-03T12:00:00Z' },
        { building_id: 'b1', building_name: 'Hall B', video_id: 'u2', filename: 'walk-b.mp4', status: 'live', created_at: '2026-11-02T12:00:00Z' },
        { building_id: 'b1', building_name: 'Hall B', video_id: 'u3', filename: 'walk-c.mp4', status: 'needs_review', reason: 'Low light in stairwell', created_at: '2026-11-01T12:00:00Z' },
        { building_id: 'b1', building_name: 'Hall B', video_id: 'u4', filename: 'walk-d.mp4', status: 'rejected', reason: 'Shows a face without consent', created_at: '2026-10-31T12:00:00Z' },
      ])
    }
    if (p === '/owner/moderation') {
      if (token !== 'tok-owner') return forbid()
      const want = url.searchParams.get('status') || 'pending'
      return json(st.moderation.filter((i) => want === 'all' || i.status === want))
    }
    const mm = /^\/owner\/videos\/([^/]+)\/([^/]+)\/moderation$/.exec(p)
    if (mm && m === 'POST') {
      if (token !== 'tok-owner') return forbid()
      await sleep(250) // a slow server: the optimistic state is visible while the request is out
      if (st.moderateFail) return json({ detail: 'Moderation service unavailable' }, 500)
      const item = st.moderation.find((i) => i.video_id === mm[2])
      const next = { approve: 'approved', reject: 'rejected', hide: 'hidden' }[body?.action]
      if (item && next) item.status = next
      return json({ ...item, status: next })
    }
    if (p === '/owner/users') {
      if (token !== 'tok-owner') return forbid()
      return json(st.users)
    }
    const bl = /^\/owner\/users\/([^/]+)\/block$/.exec(p)
    if (bl && m === 'POST') {
      if (token !== 'tok-owner') return forbid()
      if (st.blockFail) return json({ detail: 'Block failed' }, 500)
      const email = decodeURIComponent(bl[1])
      const u = st.users.find((x) => x.email === email)
      if (u) u.blocked = !!body?.blocked
      return json({ ...u })
    }
    return route.fulfill({ status: 404, body: '{}' })
  })
  await page.goto(BASE + '/' + hash)
  await page.waitForSelector('nav.tabbar')
  await sleep(400)
  return { ctx, page, errors, st }
}

const tabLabels = (page) => page.locator('nav.tabbar button').allInnerTexts().then((a) => a.map((t) => t.trim()))
const sends = (st, method, re) => st.requests.filter((r) => r.method === method && re.test(r.path))

// ---------- 1. Visitor (signed out) ----------
{
  const { ctx, page, errors, st } = await session('anon', {}, '#/navigate?b=b1')
  const tabs = await tabLabels(page)
  check('visitor: nav has Directions, Map, Record, Contribute', JSON.stringify(tabs) === JSON.stringify(['Directions', 'Map', 'Record', 'Contribute']), JSON.stringify(tabs))
  check('visitor: no Owner tab in the nav', !tabs.includes('Owner'))

  const picker = await page.locator('select[aria-label="Building"] option').allInnerTexts()
  check('visitor: picker shows only buildings the API returns (with a map)', picker.length === 1 && picker[0] === 'Hall B', JSON.stringify(picker))

  await page.locator('nav.tabbar button', { hasText: 'Map' }).click()
  await page.waitForSelector('.map-panel')
  await sleep(300)
  check('visitor map: 2D/3D switch and place list are shown', (await page.getByRole('button', { name: '3D', exact: true }).count()) === 1)
  check('visitor map: no Save, Link videos or Add photo', (await page.getByRole('button', { name: /Save changes|Saved|Link videos/ }).count()) === 0 && (await page.getByText('Add photo').count()) === 0)
  check('visitor map: no Add connection, no Delete place', (await page.getByText('Add connection').count()) === 0 && (await page.getByRole('button', { name: 'Delete', exact: true }).count()) === 0)
  check('visitor map: no reports list', (await page.getByText(/Reports are|Open reports|Something's wrong/).count()) === 0 && (await page.locator('.reports, .report-list').count()) === 0)
  await page.getByRole('button', { name: /Lobby/ }).first().click()
  await sleep(300)
  check('visitor map: picking a place shows its details without edit fields', (await page.locator('.node-editor input').count()) === 0 && (await page.getByText('Floor 1').count()) >= 1)

  await page.locator('nav.tabbar button', { hasText: 'Contribute' }).click()
  await sleep(400)
  check('visitor contribute: sign-in prompt, no upload form', (await page.getByText('Sign in to contribute').count()) >= 1 && (await page.locator('input[type=file]').count()) === 0)

  await page.goto(BASE + '/#/owner')
  await sleep(400)
  check('visitor: #/owner shows the not-allowed screen', (await page.getByText('This page is not allowed').count()) === 1)
  check('visitor: no owner endpoint is called', st.requests.every((r) => !r.path.startsWith('/owner/')), JSON.stringify(st.requests.filter((r) => r.path.startsWith('/owner/')).map((r) => r.path)))
  check('visitor: no console errors', realErrors(errors).length === 0, errors[0] || '')
  await ctx.close()
}

// ---------- 2. Contributor (signed in, not owner) ----------
{
  const { ctx, page, errors, st } = await session('member', {}, '#/build?b=b1')
  const tabs = await tabLabels(page)
  check('contributor: nav has Contribute and no Owner', tabs.includes('Contribute') && !tabs.includes('Owner'), JSON.stringify(tabs))
  check('contributor: upload form shows the walk, sensor and floor fields', (await page.locator('input[type=file][accept="video/*"]').count()) === 1 && (await page.getByLabel(/Starts on floor/).count()) === 1)
  check('contributor: no queue, Inspect, Rebuild or job controls', (await page.getByText(/Rebuild map|Inspect|Run next|Processed videos/).count()) === 0 && (await page.locator('ul.queue').count()) === 0)
  check('contributor: Record link is there', (await page.getByRole('button', { name: 'Open Record' }).count()) === 1)

  await page.waitForSelector('.ow-uploads .ow-chip')
  const chips = await page.locator('.ow-uploads .ow-chip').allInnerTexts()
  check('contributor: My uploads shows the four friendly statuses', ['Processing', 'Live', 'Needs review', 'Rejected'].every((c) => chips.some((t) => t.includes(c))), JSON.stringify(chips))
  check('contributor: rejected upload shows its reason', (await page.getByText('Shows a face without consent').count()) === 1)
  check('contributor: needs review shows its reason', (await page.getByText('Low light in stairwell').count()) === 1)
  check('contributor: no processed-video internals in My uploads', !/keyframes|gyro|\bdone\b|job|Inspect/i.test(await page.locator('.ow-uploads').innerText()))

  await page.locator('input[type=file][accept="video/*"]').setInputFiles({ name: 'mine.mp4', mimeType: 'video/mp4', buffer: Buffer.alloc(2048) })
  await page.getByRole('button', { name: 'Upload walk' }).click()
  await sleep(600)
  const up = sends(st, 'POST', /^\/buildings\/b1\/videos$/)
  check('contributor: upload is sent with the bearer header', up.length === 1 && up[0].auth === 'Bearer tok-member', JSON.stringify(up))
  check('contributor: upload toast', (await page.getByText('Walk uploaded. It shows under My uploads.').count()) === 1)

  await page.goto(BASE + '/#/owner')
  await sleep(500)
  check('contributor: #/owner shows the not-allowed screen', (await page.getByText('This page is not allowed').count()) === 1)
  check('contributor: no owner endpoint is called', st.requests.every((r) => !r.path.startsWith('/owner/')))
  check('contributor: no console errors', realErrors(errors).length === 0, errors[0] || '')
  await ctx.close()
}

// ---------- 3. Owner ----------
{
  const { ctx, page, errors, st } = await session('owner', {}, '#/navigate?b=b1')
  const tabs = await tabLabels(page)
  check('owner: nav has Owner and no Contribute', tabs.includes('Owner') && !tabs.includes('Contribute'), JSON.stringify(tabs))
  const picker = await page.locator('select[aria-label="Building"] option').allInnerTexts()
  check('owner: picker lists every building, empty ones too', picker.includes('Annex') && picker.includes('Hall B'), JSON.stringify(picker))

  await page.locator('nav.tabbar button', { hasText: 'Owner' }).click()
  await page.waitForSelector('.ow-cards li.ow-card')
  check('owner: moderation is the default section (Pending)', (await page.locator('.ow-mod h3').innerText()) === 'Moderation queue' && sends(st, 'GET', /^\/owner\/moderation$/)[0]?.query === '?status=pending')
  const first = page.locator('.ow-cards li.ow-card').first()
  check('owner: card shows up to 6 frames', (await first.locator('img.ow-frame').count()) === 6, String(await first.locator('img.ow-frame').count()))
  check('owner: card shows filename, building, uploader and bot reason', (await first.innerText()).includes('lobby-walk.mp4') && (await first.innerText()).includes('Hall B') && (await first.innerText()).includes('walker@example.com') && (await first.innerText()).includes('Possible face in frame 3'))

  // Approve: optimistic removal and a toast.
  await first.getByRole('button', { name: 'Approve' }).click()
  await sleep(450)
  check('owner moderation: approve removes the card (Pending filter)', (await page.locator('.ow-cards li.ow-card').count()) === 1)
  check('owner moderation: approve toast', (await page.getByText('Approved: lobby-walk.mp4').count()) === 1)
  const ap = sends(st, 'POST', /^\/owner\/videos\/b1\/v1\/moderation$/)
  check('owner moderation: approve sends {action: approve}', ap.length === 1 && ap[0].body?.action === 'approve', JSON.stringify(ap))

  // Reject with a server failure: the card comes back and an error toast says so.
  st.moderateFail = true
  await page.locator('.ow-cards li.ow-card').first().getByRole('button', { name: 'Reject' }).click()
  await sleep(60)
  const optimisticGone = (await page.locator('.ow-cards li.ow-card').count()) === 0
  await sleep(600)
  check('owner moderation: reject removes the card while the request is out', optimisticGone)
  check('owner moderation: a failed reject brings the card back', (await page.locator('.ow-cards li.ow-card').count()) === 1, String(await page.locator('.ow-cards li.ow-card').count()))
  check('owner moderation: failed reject shows an error toast', (await page.getByText(/Could not reject stairs.mp4/).count()) === 1)
  st.moderateFail = false

  // Hide, on the All filter: the card stays and shows its new chip.
  await page.getByRole('button', { name: 'All', exact: true }).click()
  await sleep(400)
  check('owner moderation: All filter lists everything', (await page.locator('.ow-cards li.ow-card').count()) >= 2)
  await page.locator('.ow-cards li.ow-card').filter({ hasText: 'stairs.mp4' }).getByRole('button', { name: 'Hide' }).click()
  await sleep(500)
  check('owner moderation: hide on All keeps the card with a Hidden chip', (await page.locator('.ow-cards li.ow-card').filter({ hasText: 'stairs.mp4' }).locator('.ow-chip').innerText()) === 'Hidden')

  // Videos & map: the Build content and the map editing tools.
  await page.getByRole('button', { name: 'Videos & map' }).click()
  await page.waitForSelector('.map-panel')
  await sleep(300)
  check('owner videos: Rebuild map, queue, Link videos, Add photo and Save are there',
    (await page.getByRole('button', { name: /Rebuild map/ }).count()) === 1 && (await page.getByRole('button', { name: /Link videos/ }).count()) === 1
    && (await page.getByText('Add photo').count()) === 1 && (await page.getByRole('button', { name: /Saved|Save changes/ }).count()) === 1)
  check('owner videos: the map is in edit mode (Add connection)', (await page.getByText('Add connection').count()) === 1)
  check('owner videos: the walk upload form is there', (await page.getByText('Add a walkthrough').count()) === 1)
  check('owner url: the section is kept in the hash', (await page.url()).includes('#/owner?') && (await page.url()).includes('s=videos'), await page.url())

  // Buildings: list all, create, delete with the in-page dialog.
  await page.getByRole('button', { name: 'Buildings', exact: true }).click()
  await sleep(300)
  check('owner buildings: lists the empty building too', (await page.getByText('No places yet').count()) === 1 && (await page.locator('.ow-building').count()) === 2, `${await page.getByText('No places yet').count()} / ${await page.locator('.ow-building').count()} / ${(await page.locator('.ow-buildings').innerText().catch(() => 'none')).slice(0, 300)}`)
  await page.getByRole('textbox', { name: 'New building name' }).fill('Library')
  await page.getByRole('button', { name: 'Create', exact: true }).click()
  await sleep(500)
  check('owner buildings: create posts the name', sends(st, 'POST', /^\/buildings$/).some((r) => r.body?.name === 'Library' && r.auth === 'Bearer tok-owner'))

  await page.getByRole('button', { name: 'Delete Annex' }).click()
  await sleep(350)
  const dlg = page.getByRole('alertdialog')
  check('owner buildings: delete opens an in-page dialog, not window.confirm', (await dlg.count()) === 1 && (await dlg.innerText()).includes('Delete Annex'))
  check('owner buildings: the app behind is inert while the dialog is open', await page.evaluate(() => !!document.querySelector('.app > header')?.inert))
  await page.keyboard.press('Escape')
  await sleep(350)
  check('owner buildings: Escape closes the dialog without deleting', (await page.getByRole('alertdialog').count()) === 0 && sends(st, 'DELETE', /^\/buildings\/b2$/).length === 0)
  check('owner buildings: focus returns to the Delete button', await page.evaluate(() => document.activeElement?.getAttribute('aria-label') === 'Delete Annex'))

  await page.getByRole('button', { name: 'Delete Annex' }).click()
  await sleep(350)
  await page.getByRole('alertdialog').getByRole('button', { name: 'Delete building' }).click()
  await sleep(600)
  check('owner buildings: confirm sends DELETE for the building', sends(st, 'DELETE', /^\/buildings\/b2$/).length === 1)
  check('owner buildings: delete toast and the building is gone', (await page.getByText('Deleted "Annex"').count()) === 1 && (await page.getByRole('button', { name: 'Delete Annex' }).count()) === 0)

  // Reports (its own section).
  await page.getByRole('button', { name: 'Reports', exact: true }).click()
  await sleep(400)
  check('owner reports: the reports list is shown', (await page.getByText('Wrong turn here').count()) >= 1, (await page.locator('main').innerText()).slice(0, 400))

  // Users: block and unblock.
  await page.getByRole('button', { name: 'Users', exact: true }).click()
  await page.waitForSelector('.ow-table')
  const row = page.locator('.ow-table tbody tr').filter({ hasText: 'walker@example.com' })
  await row.getByRole('button', { name: 'Block' }).click()
  await sleep(500)
  check('owner users: block posts {blocked: true}', sends(st, 'POST', /\/owner\/users\/walker%40example\.com\/block$/).some((r) => r.body?.blocked === true))
  check('owner users: the row shows Blocked and an Unblock button', (await row.getByText('Blocked').count()) === 1 && (await row.getByRole('button', { name: 'Unblock' }).count()) === 1)
  await row.getByRole('button', { name: 'Unblock' }).click()
  await sleep(500)
  check('owner users: unblock posts {blocked: false}', sends(st, 'POST', /\/owner\/users\/walker%40example\.com\/block$/).some((r) => r.body?.blocked === false))

  check('owner: no console errors', realErrors(errors).length === 0, errors[0] || '')
  await ctx.close()
}

// ---------- 4. Owner: block failure shows a toast, the row keeps its state ----------
{
  const { ctx, page, st } = await session('owner', { blockFail: true }, '#/owner?s=users&b=b1')
  await page.waitForSelector('.ow-table')
  const row = page.locator('.ow-table tbody tr').filter({ hasText: 'walker@example.com' })
  await row.getByRole('button', { name: 'Block' }).click()
  await sleep(500)
  check('owner users: a failed block keeps the row unblocked and says so', (await row.getByText('Blocked').count()) === 0 && (await page.getByText(/Could not block/).count()) === 1, JSON.stringify(st.requests.slice(-1)))
  await ctx.close()
}

// ---------- 5. Owner: a delete that fails keeps the building ----------
{
  const { ctx, page } = await session('owner', { deleteFail: true }, '#/owner?s=buildings&b=b1')
  await page.waitForSelector('.ow-building')
  await page.getByRole('button', { name: 'Delete Annex' }).click()
  await sleep(300)
  await page.getByRole('alertdialog').getByRole('button', { name: 'Delete building' }).click()
  await sleep(600)
  check('owner buildings: a failed delete says so and keeps the building', (await page.getByText(/Could not delete "Annex"/).count()) === 1 && (await page.getByRole('button', { name: 'Delete Annex' }).count()) === 1)
  await ctx.close()
}

// ---------- 6. Owner page on a refresh while the session is still being checked ----------
{
  const { ctx, page } = await session('owner', {}, '#/owner?s=users&b=b1')
  await page.waitForSelector('.ow-table')
  check('owner: a deep link opens its section', (await page.locator('.ow-table').count()) === 1)
  await ctx.close()
}

// ---------- 7. Owner: unknown dates, the section from the hash (also on a hash change), Block hidden for is_owner rows ----------
{
  const undated = (id, created_at) => ({ building_id: 'b1', building_name: 'Hall B', video_id: id, filename: `${id}.mp4`, uploader_email: 'walker@example.com', status: 'pending', reason: '', frames: [], created_at })
  const { ctx, page, errors, st } = await session('owner', {
    moderation: [undated('n1', null), undated('n2', 0), undated('n3', 'not-a-date'), undated('n4', '2026-11-03T10:00:00Z')],
    users: [
      { email: 'walker@example.com', uploads: 3, live: 1, rejected: 1, blocked: false, last_seen: null },
      { email: 'cofounder@example.com', uploads: 0, live: 0, rejected: 0, blocked: false, last_seen: null, is_owner: true },
      { email: OWNER.email, uploads: 0, live: 0, rejected: 0, blocked: false, last_seen: null },
    ],
  }, '#/navigate?b=b1')
  await page.locator('nav.tabbar button', { hasText: 'Owner' }).click()
  await page.waitForSelector('.ow-cards li.ow-card')
  const cards = await page.locator('.ow-cards li.ow-card').allInnerTexts()
  check('owner dates: a null, zero or invalid created_at shows "Date unknown"', cards.slice(0, 3).every((t) => t.includes('Date unknown')), JSON.stringify(cards.map((t) => t.slice(-60))))
  check('owner dates: no card shows the 1970 epoch (Dec 31)', cards.every((t) => !/Dec 31|1969|1970/.test(t)), JSON.stringify(cards.map((t) => t.slice(-60))))
  check('owner dates: a real created_at still shows its date', cards[3].includes('Nov') && !cards[3].includes('Date unknown'), cards[3].slice(-80))

  // Hash: the section is read on load and follows the hash after that (a hash change while the page is open).
  await page.getByRole('button', { name: 'Users', exact: true }).click()
  await page.waitForSelector('.ow-table')
  check('owner hash: switching to Users writes s=users into the hash', (await page.url()).includes('s=users'), await page.url())
  await page.evaluate(() => { window.location.hash = '#/owner?s=moderation' })
  await sleep(400)
  check('owner hash: a hash change to s=moderation opens Moderation', (await page.getByRole('button', { name: 'Moderation', exact: true }).getAttribute('aria-pressed')) === 'true' && (await page.locator('.ow-mod').count()) === 1)
  await page.evaluate(() => { window.location.hash = '#/owner?s=users' })
  await page.waitForSelector('.ow-table')
  check('owner hash: a hash change to s=users opens Users', (await page.locator('.ow-table').count()) === 1)
  await page.evaluate(() => { window.location.hash = '#/owner?s=reports&b=b1' })
  await sleep(400)
  check('owner hash: a hash change to s=reports opens Reports', (await page.getByRole('button', { name: 'Reports', exact: true }).getAttribute('aria-pressed')) === 'true')
  await page.getByRole('button', { name: 'Buildings', exact: true }).click()
  await sleep(300)
  check('owner hash: switching to Buildings keeps the building and writes s=buildings', (await page.url()).includes('s=buildings') && (await page.url()).includes('b=b1'), await page.url())

  // Block: hidden for the owner-flagged row (is_owner), shown for an ordinary row.
  await page.getByRole('button', { name: 'Users', exact: true }).click()
  await page.waitForSelector('.ow-table')
  const coRow = page.locator('.ow-table tbody tr').filter({ hasText: 'cofounder@example.com' })
  const wRow = page.locator('.ow-table tbody tr').filter({ hasText: 'walker@example.com' })
  check('owner users: no Block button on a row with is_owner: true', (await coRow.count()) === 1 && (await coRow.getByRole('button', { name: 'Block' }).count()) === 0)
  check('owner users: Block is still shown for an ordinary row', (await wRow.getByRole('button', { name: 'Block' }).count()) === 1)
  check('owner users: no Block on the signed-in owner row (no is_owner flag)', (await page.locator('.ow-table tbody tr').filter({ hasText: OWNER.email }).getByRole('button', { name: 'Block' }).count()) === 0)
  check('owner: no console errors (dates, hash, users)', realErrors(errors).length === 0, errors[0] || '')
  await ctx.close()
}

const failed = results.filter((r) => !r.ok)
console.log(`\n${results.length - failed.length}/${results.length} checks passed`)
await browser.close()
process.exit(failed.length ? 1 : 0)
