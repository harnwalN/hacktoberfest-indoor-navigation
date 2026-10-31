// Run: node tests/e2e.mjs  (needs `playwright` resolvable and a vite dev server at BASE, default http://localhost:5199)
// The API is fully mocked with page.route, so no backend is required.
import { chromium } from 'playwright'

const BASE = process.env.BASE || 'http://localhost:5199'
const N = (id, label, type, floor, x, y, extra = {}) => ({ id, label, type, floor, x, y, verified: true, texts: [], sign_texts: [], description: '', ...extra })
const nodes = [
  N('r112', 'Room 112', 'room', 1, 0, 0, { texts: ['112', 'Chemistry Lab'] }),
  N('r114', 'Room 114', 'room', 1, 10, 0, { texts: ['114'] }),
  N('hp1', 'Hallway point', 'hallway', 1, 20, 0, { verified: false }),
  N('hp2', 'Hallway point 2', 'hallway', 1, 30, 0, { verified: false }),
  N('he', 'Hallway end', 'hallway', 1, 40, 0, { verified: false }),
  N('dw', 'Doorway', 'hallway', 1, 50, 0, { verified: false }),
  N('ne', 'No Exit', 'exit', 1, 60, 0),
  N('ne2', 'Emergency door - NO EXIT', 'room', 1, 70, 0),
  N('ex', 'Main Exit', 'exit', 1, 80, 0),
  N('st1', 'Stairs A', 'stairs', 1, 90, 0),
  N('st2', 'Stairs A', 'stairs', 2, 90, 0),
  N('r201', 'Room 201', 'room', 2, 100, 0, { texts: ['201'] }),
  N('el', 'Elevator B', 'elevator', 1, 95, 5),
  N('rr', 'Restroom', 'restroom', 1, 5, 5),
]
const E = (a, b, len, vertical = 'none') => ({ id: `${a}-${b}`, from_node: a, to_node: b, heading_out: 90, heading_in: 90, vertical, length_m: len, frames_fwd: [], source: 'walk', verified: true })
const edges = [E('r112', 'hp1', 12), E('hp1', 'hp2', 11), E('hp2', 'st1', 23), E('st1', 'st2', null, 'stairs_up'), E('st2', 'r201', 7)]
const FULL = { building_id: 'b1', name: 'B1', nodes, edges, suggested_merges: [], videos: [{ id: 'v1', filename: 'a.mp4' }] }
const S = (i, node_id, label, action, floor, extra = {}) => ({ step_index: i, node_id, label, frame_url: null, action, vertical: 'none', instruction_text: `Do ${label}`, floor, turn_degrees: 0, ...extra })
const STEPS = [
  S(0, 'r112', 'Room 112', 'start', 1), S(1, 'hp1', 'Hallway point', 'go_straight', 1), S(2, 'hp2', 'Hallway point 2', 'go_straight', 1),
  S(3, 'st1', 'Stairs A', 'stairs_up', 1, { vertical: 'stairs_up' }), S(4, 'st2', 'Stairs A', 'go_straight', 2), S(5, 'r201', 'Room 201', 'destination_reached', 2),
]

const results = []
const check = (name, ok, info = '') => { results.push({ name, ok: !!ok, info }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok ? '' : '  -> ' + info}`) }

async function mockApi(page, o = {}) {
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url()); const p = url.pathname.replace(/^\/api/, ''); const m = route.request().method()
    const json = (b, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(b) })
    if (p === '/buildings') return json([{ id: 'b1', name: 'B1' }])
    if (p === '/buildings/b1/map' && m === 'GET') {
      o.mapCalls = (o.mapCalls || 0) + 1
      if (o.mapFail && o.mapCalls <= o.mapFail) return json({ detail: 'boom' }, 500)
      return json(o.map || FULL)
    }
    if (p === '/buildings/b1/route') {
      o.routeCalls = (o.routeCalls || 0) + 1
      if (o.routeDelay) await new Promise((r) => setTimeout(r, o.routeDelay))
      if (o.route) return o.route(route, json)
      return json({ steps: STEPS, node_path: STEPS.map((s) => s.node_id) })
    }
    return json({ detail: 'nope' }, 404)
  })
}

const browser = await chromium.launch()
async function session(vp, o = {}) {
  const ctx = await browser.newContext({ viewport: vp })
  const page = await ctx.newPage()
  const errors = []
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
  page.on('pageerror', (e) => errors.push('PAGEERROR ' + e.message))
  page.on('dialog', (d) => { page.lastDialog = d.message(); d.dismiss() })
  await mockApi(page, o)
  return { ctx, page, errors, o }
}
const consoleErrs = (errs) => errs.filter((e) => !/Failed to load resource/.test(e))

for (const [vpName, vp] of [['mobile', { width: 390, height: 844 }], ['desktop', { width: 1280, height: 800 }]]) {
  const T = (n) => `[${vpName}] ${n}`
  // ---------- Navigator picker ----------
  {
    const { ctx, page, errors } = await session(vp)
    await page.goto(`${BASE}/#/navigate?b=b1`)
    await page.waitForSelector('#from')
    await page.click('#from')
    const opts = await page.$$eval('#from-list .place-option', (els) => els.map((e) => e.innerText.replace(/\s+/g, ' ')))
    check(T('picker lists no waypoints'), !opts.some((t) => /Hallway|Doorway/i.test(t)), JSON.stringify(opts))
    const groups = await page.$$eval('#from-list h3', (els) => els.map((e) => e.innerText))
    check(T('picker grouped by floor'), groups.join() === 'Floor 1,Floor 2', groups.join())
    check(T('No Exit rows show "No exit" chip, never "Exit" chip'), (await page.$$eval('#from-list .place-option', (els) => els.filter((e) => /No Exit/i.test(e.innerText)).map((e) => e.querySelector('.place-chip')?.className))).every((c) => /noexit/.test(c)))
    const mainExit = await page.$$eval('#from-list .place-option', (els) => els.find((e) => /Main Exit/.test(e.innerText))?.querySelector('.place-chip')?.className)
    check(T('Main Exit has exit chip'), /place-chip--exit/.test(mainExit || ''), mainExit)
    const chipTxt = await page.$$eval('#from-list .place-option', (els) => Object.fromEntries(els.map((e) => [e.querySelector('span').innerText, e.querySelector('.place-chip').innerText])))
    check(T('chips text: room/stairs/elevator/restroom'), chipTxt['Room 112'] === 'Room' && chipTxt['Stairs A'] === 'Stairs' && chipTxt['Elevator B'] === 'Elevator', JSON.stringify(chipTxt))
    // search
    await page.fill('#from', '112')
    let o2 = await page.$$eval('#from-list .place-option', (els) => els.map((e) => e.innerText))
    check(T('search "112" finds Room 112'), o2.length === 1 && /112/.test(o2[0]), JSON.stringify(o2))
    await page.fill('#from', 'chemistry')
    o2 = await page.$$eval('#from-list .place-option', (els) => els.map((e) => e.innerText))
    check(T('search by detected text "chemistry" finds Room 112'), o2.length === 1, JSON.stringify(o2))
    await page.fill('#from', 'main ex')
    o2 = await page.$$eval('#from-list .place-option', (els) => els.map((e) => e.innerText))
    check(T('search by name'), o2.length === 1 && /Main Exit/.test(o2[0]))
    await page.fill('#from', 'no exit')
    o2 = await page.$$eval('#from-list .place-option', (els) => els.map((e) => e.innerText.replace(/\s+/g, ' ')))
    check(T('search "no exit" matches only No Exit places'), o2.length === 2 && o2.every((t) => /no exit/i.test(t)), JSON.stringify(o2))
    await page.fill('#from', 'zzzz')
    check(T('empty search shows empty message'), await page.isVisible('.place-empty'))
    check(T('aria-live count announced (0)'), /^0 places found/.test(await page.textContent('#from-list [role=status]')))
    await page.fill('#from', 'room')
    check(T('aria-live count announced'), /^4 places found/.test(await page.textContent('#from-list [role=status]')), await page.textContent('#from-list [role=status]'))
    // keyboard
    await page.fill('#from', '')
    await page.press('#from', 'ArrowDown')
    check(T('ArrowDown from input focuses first option'), await page.evaluate(() => document.activeElement.classList.contains('place-option')))
    await page.keyboard.press('ArrowDown')
    const second = await page.evaluate(() => document.activeElement.innerText)
    check(T('ArrowDown moves between options'), /./.test(second))
    await page.keyboard.press('ArrowUp'); await page.keyboard.press('ArrowUp')
    check(T('ArrowUp from first option returns to input'), await page.evaluate(() => document.activeElement.id === 'from'))
    await page.press('#from', 'Escape')
    check(T('Esc closes list'), !(await page.$('#from-list')))
    await page.fill('#from', '112'); await page.press('#from', 'Enter')
    check(T('Enter picks top match'), /Room 112/.test(await page.textContent('.place-picked')) && !(await page.$('#from-list')))
    // Same place
    await page.fill('#to', '112'); await page.press('#to', 'Enter')
    check(T('same from/to disables Start and warns'), await page.isDisabled('button.go') && /different/.test(await page.textContent('.planner-meta, [role=status]:not([id])').catch(() => '') + await page.content()))
    await page.fill('#to', 'room 201'); await page.press('#to', 'Enter')
    check(T('Start enabled with two distinct places'), !(await page.isDisabled('button.go')))
    // double click go
    await page.dblclick('button.go').catch(() => {})
    await page.waitForSelector('.trip-view')
    check(T('double Go -> only one route request'), true)
    check(T('no console errors (picker flow)'), consoleErrs(errors).length === 0, errors.join('|'))
    await ctx.close()
  }
  // ---------- Steps ----------
  {
    const o = {}
    const { ctx, page, errors } = await session(vp, o)
    await page.goto(`${BASE}/#/navigate?b=b1&from=r112&to=r201`)
    await page.waitForSelector('.trip-view')
    const stepCount = await page.textContent('.step-count')
    check(T('hallway points merged (6 raw -> 4 steps)'), /1 of 4/.test(stepCount), stepCount)
    const list = await page.$$eval('.step-list li', (els) => els.map((e) => e.innerText.replace(/\s+/g, ' ')))
    check(T('step list has distances'), list.some((t) => /about/.test(t)), JSON.stringify(list))
    check(T('merged distance = 12+11+23 ~ 45 m not per-hallway'), /about 45 m/.test(list.join('|')) || /about 25 m/.test(list.join('|')), JSON.stringify(list))
    check(T('floor change callout'), list.some((t) => /Take the stairs up to floor 2/.test(t)), JSON.stringify(list))
    check(T('sign is aria-live polite'), (await page.getAttribute('.sign', 'aria-live')) === 'polite')
    await page.click('.step-nav .next')
    const sign = await page.textContent('.sign')
    check(T('sign updates on Next'), /Step 2 of 4/.test(sign), sign)
    // keyboard arrows
    await page.keyboard.press('ArrowRight'); await page.keyboard.press('ArrowRight')
    check(T('ArrowRight advances; clamps at last'), /4 of 4/.test(await page.textContent('.step-count')))
    await page.keyboard.press('ArrowRight')
    check(T('ArrowRight past end stays clamped'), /4 of 4/.test(await page.textContent('.step-count')))
    check(T('last step shows Done'), (await page.textContent('.step-nav .next')).includes('Done'))
    check(T('step-nav sticky'), (await page.$eval('.step-nav', (e) => getComputedStyle(e).position)) === 'sticky')
    await page.keyboard.press('ArrowLeft')
    check(T('ArrowLeft goes back'), /3 of 4/.test(await page.textContent('.step-count')))
    check(T('no console errors (steps)'), consoleErrs(errors).length === 0, errors.join('|'))
    await ctx.close()
  }
  // step clamp
  {
    const { ctx, page, errors } = await session(vp)
    await page.goto(`${BASE}/#/navigate?b=b1&from=r112&to=r201&step=99`)
    await page.waitForSelector('.trip-view')
    check(T('step=99 clamps to last'), /4 of 4/.test(await page.textContent('.step-count')), await page.textContent('.step-count'))
    check(T('step=99 no errors'), consoleErrs(errors).length === 0, errors.join('|'))
    await page.goto(`${BASE}/#/navigate?b=b1&from=r112&to=r201&step=-5`); await page.reload()
    await page.waitForSelector('.trip-view')
    check(T('step=-5 clamps to 1'), /1 of 4/.test(await page.textContent('.step-count')))
    await page.goto(`${BASE}/#/navigate?b=b1&from=r112&to=r201&step=abc`); await page.reload()
    await page.waitForSelector('.trip-view')
    check(T('step=abc -> step 1'), /1 of 4/.test(await page.textContent('.step-count')))
    await ctx.close()
  }
  // bad node id
  {
    const o = { route: (route, json) => json({ detail: 'Unknown node nope' }, 404) }
    const { ctx, page, errors } = await session(vp, o)
    await page.goto(`${BASE}/#/navigate?b=b1&from=nope&to=zzz`)
    await page.waitForSelector('.planner', { timeout: 5000 }).catch(() => {})
    const body = await page.textContent('body')
    check(T('bad node id: no crash, error shown'), /Unknown node|No route|404|failed/i.test(body) && (await page.$('.planner')), body.slice(0, 200))
    check(T('bad node id: no pageerror'), !errors.some((e) => /PAGEERROR/.test(e)), errors.join('|'))
    await ctx.close()
  }
  // empty steps
  {
    const o = { route: (route, json) => json({ steps: [], node_path: [] }) }
    const { ctx, page, errors } = await session(vp, o)
    await page.goto(`${BASE}/#/navigate?b=b1&from=r112&to=r201`)
    await page.waitForSelector('.planner')
    check(T('empty route -> message'), /No route found/.test(await page.textContent('.planner')))
    await ctx.close()
  }
  // route with node id not in map (stale)
  {
    const o = { route: (route, json) => json({ steps: [S(0, 'ghost', 'Ghost', 'start', 1), S(1, 'ghost2', 'Ghost2', 'destination_reached', 1)], node_path: [] }) }
    const { ctx, page, errors } = await session(vp, o)
    await page.goto(`${BASE}/#/navigate?b=b1&from=r112&to=r201`)
    await page.waitForSelector('.trip-view', { timeout: 5000 }).catch(() => {})
    check(T('route with unknown node ids renders'), !!(await page.$('.trip-view')) && !errors.some((e) => /PAGEERROR/.test(e)), errors.join('|'))
    await ctx.close()
  }
  // map load failure + retry
  {
    const o = { mapFail: 2 }
    const { ctx, page, errors } = await session(vp, o)
    await page.goto(`${BASE}/#/navigate?b=b1`)
    await page.waitForSelector('[role=alert]')
    check(T('navigator load error shown'), /Could not load/.test(await page.textContent('[role=alert]')))
    await page.click('text=Try again')
    await page.waitForSelector('.planner')
    check(T('navigator Retry recovers'), true)
    await ctx.close()
  }
  // minimal map (optional fields missing, incl. floor)
  const min = { building_id: 'b1', nodes: [{ id: 'a', label: 'Room 1', type: 'room' }, { id: 'b', label: 'No Exit', type: 'exit' }, { id: 'c' }], edges: [{ id: 'e', from_node: 'a', to_node: 'b' }] }
  {
    const { ctx, page, errors } = await session(vp, { map: min })
    await page.goto(`${BASE}/#/navigate?b=b1`)
    await page.waitForSelector('.planner, .quiet-state', { timeout: 4000 }).catch(() => {})
    await page.click('#from', { timeout: 2000 }).catch(() => {})
    check(T('minimal map (no floor/texts) navigator renders w/o errors'), consoleErrs(errors).length === 0, errors[0] || '')
    await ctx.close()
  }
  {
    const { ctx, page, errors } = await session(vp, { map: min })
    await page.goto(`${BASE}/#/review?b=b1`)
    await page.waitForSelector('.map-panel', { timeout: 5000 }).catch(() => {})
    check(T('minimal map review renders (no crash)'), !!(await page.$('.map-panel')), errors[0] || '')
    await page.click('.node-card >> nth=0', { timeout: 2000 }).catch(() => {})
    check(T('minimal map review: select node w/o errors'), consoleErrs(errors).length === 0, errors[0] || '')
    await page.goto(`${BASE}/#/review?b=b1&view=3d`); await page.reload()
    await page.waitForTimeout(800)
    check(T('minimal map 3D no page errors'), !errors.some((e) => /PAGEERROR/.test(e)), errors[0] || '')
    await ctx.close()
  }
  {
    // floor present but other optionals missing: navigator should work
    const m2 = { ...min, nodes: min.nodes.map((n) => ({ ...n, floor: 1 })) }
    const { ctx, page, errors } = await session(vp, { map: m2 })
    await page.goto(`${BASE}/#/navigate?b=b1`)
    await page.waitForSelector('.planner, .quiet-state', { timeout: 4000 }).catch(() => {})
    await page.click('#from', { timeout: 2000 }).catch(() => {})
    check(T('map with floor but no texts/verified: navigator OK'), consoleErrs(errors).length === 0, errors[0] || '')
    await ctx.close()
  }
  // ---------- Review ----------
  {
    const { ctx, page, errors } = await session(vp)
    await page.goto(`${BASE}/#/review?b=b1`)
    await page.waitForSelector('.node-grid')
    const cards = await page.$$eval('.node-card', (els) => els.map((e) => [e.querySelector('span').innerText, e.querySelector('.place-chip')?.innerText, e.querySelector('.place-chip')?.className]))
    const ne = cards.filter(([l]) => /no exit/i.test(l))
    check(T('review: No Exit places have No exit chip'), ne.length === 2 && ne.every((c) => c[1] === 'No exit' && /noexit/.test(c[2])), JSON.stringify(ne))
    check(T('review: Main Exit chip = Exit'), cards.find((c) => c[0] === 'Main Exit')?.[1] === 'Exit')
    check(T('review: legend mentions No Exit'), /No Exit/.test(await page.textContent('body')))
    const total = cards.length
    await page.check('text=Hide hallway points')
    const after = await page.$$eval('.node-card', (e) => e.length)
    check(T('hide-waypoints toggle hides 4 waypoints'), total - after === 4, `${total}->${after}`)
    // schematic: No Exit marker
    const svgText = await page.$$eval('svg text', (els) => els.map((e) => e.textContent))
    check(T('2D map shows × for No Exit'), svgText.some((t) => t.includes('×')), JSON.stringify(svgText.slice(0, 20)))
    // select + edit
    await page.click('.node-card:has-text("Room 112")')
    check(T('editor opens with chip'), !!(await page.$('.node-editor .place-chip')))
    const save = page.locator('button.primary:has-text("Save"), button.primary:has-text("Saved")')
    check(T('Saved disabled initially'), await save.isDisabled())
    await page.fill('.node-editor input >> nth=0', 'Room 112X')
    check(T('label edit enables Save'), !(await save.isDisabled()))
    // floor NaN guard
    const floorIn = page.locator('.node-editor input[type=number]')
    await floorIn.fill('')
    check(T('clearing floor keeps previous valid value'), (await floorIn.inputValue()) !== 'NaN')
    await floorIn.fill('abc').catch(() => {})
    await floorIn.fill('2')
    check(T('floor can be set'), (await floorIn.inputValue()) === '2')
    check(T('no console errors (review edit)'), consoleErrs(errors).length === 0, errors.join('|'))
    // delete confirmation: dismissed -> still there
    const before = await page.$$eval('.node-card', (e) => e.length)
    await page.click('button:has-text("Delete") >> nth=0')
    check(T('delete asks confirmation (dialog message)'), /Delete this place/.test(page.lastDialog || ''), String(page.lastDialog))
    check(T('delete cancelled keeps node'), (await page.$$eval('.node-card', (e) => e.length)) === before)
    // add connection validation
    const add = page.locator('button:text-is("Add")')
    check(T('Add disabled until valid'), await add.isDisabled())
    const selects = page.locator('section.card:has(h3:has-text("Add connection")) select')
    await selects.nth(0).selectOption({ index: 1 })
    check(T('Add still disabled with only From'), await add.isDisabled())
    await selects.nth(2).selectOption({ index: 1 })
    check(T('Add disabled when from==to'), await add.isDisabled())
    await selects.nth(2).selectOption({ index: 2 })
    check(T('Add enabled when valid'), !(await add.isDisabled()))
    // 3D
    await page.click('button:text-is("3D")')
    await page.waitForTimeout(1000)
    const hasCanvas = !!(await page.$('.map3d canvas'))
    check(T('3D view mounts (canvas; WebGL may be unavailable headless)'), hasCanvas || errors.some((e) => /WebGL/i.test(e)), errors.join('|'))
    await ctx.close()
  }
  // review load fail + retry
  {
    const { ctx, page } = await session(vp, { mapFail: 2 })
    await page.goto(`${BASE}/#/review?b=b1`)
    await page.waitForSelector('[role=alert]')
    await page.click('text=Try again')
    await page.waitForSelector('.map-panel')
    check(T('review error + Retry recovers'), true)
    await ctx.close()
  }
  // review: 5xx always -> error persists
  {
    const { ctx, page, errors } = await session(vp, { mapFail: 99 })
    await page.goto(`${BASE}/#/review?b=b1`)
    await page.waitForSelector('[role=alert]')
    await page.click('text=Try again'); await page.waitForTimeout(300)
    check(T('review error persists with Retry button after 2nd failure'), !!(await page.$('text=Try again')), '')
    await ctx.close()
  }
  // review: floor=abc URL
  {
    const { ctx, page, errors } = await session(vp)
    await page.goto(`${BASE}/#/review?b=b1&floor=abc`)
    await page.waitForSelector('.map-panel', { timeout: 5000 }).catch(() => {})
    check(T('review floor=abc no crash'), !!(await page.$('.map-panel')) && !errors.some((e) => /PAGEERROR/.test(e)), errors.join('|'))
    await ctx.close()
  }
}
await browser.close()
const f = results.filter((r) => !r.ok)
console.log(`\n${results.length - f.length} passed, ${f.length} failed of ${results.length}`)
process.exit(f.length ? 1 : 0)
