// Runs the mocked Playwright suites one after another against a local server, then stops the server.
//
//   node tests/run-all.mjs e2e       vite dev server; all mocked suites in SUITES.e2e
//   node tests/run-all.mjs offline   production build + vite preview; the offline suites in SUITES.offline
//
// Called from package.json as `npm run test:e2e` and `npm run test:offline`.
// The server listens on 127.0.0.1 on the first free port in 5470-5479. Each suite gets BASE set to that server.
// Exit code: 0 when every suite passes, 1 when any suite fails or times out, 2 on a usage or setup error.
//
// Not included: e2e-owner-realbackend.mjs. It needs a running backend (MAIL_BACKEND=log, OWNER_EMAILS=owner@test.local,
// a backend log file) and a vite dev server with API_TARGET pointing at it. Run it by hand:
//   LOG=/path/be.log BASE=http://127.0.0.1:5431 API=http://127.0.0.1:8030 node tests/e2e-owner-realbackend.mjs
import { spawn } from 'node:child_process'
import fs from 'node:fs'
import http from 'node:http'
import net from 'node:net'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const TESTS_DIR = path.dirname(fileURLToPath(import.meta.url))
const FRONTEND_DIR = path.resolve(TESTS_DIR, '..')
const VITE_BIN = path.join(FRONTEND_DIR, 'node_modules', 'vite', 'bin', 'vite.js')
const OFFLINE_OUT = path.join(FRONTEND_DIR, 'node_modules', '.cache', 'e2e-offline-dist')
const PORT_FIRST = 5470
const PORT_LAST = 5479
const SUITE_TIMEOUT_MS = 10 * 60 * 1000
const SERVER_READY_MS = 90 * 1000

const SUITES = {
  e2e: [
    'e2e.mjs',
    'e2e-motion.mjs',
    'e2e-round2.mjs',
    'e2e-stepfree.mjs',
    'e2e-reports.mjs',
    'e2e-auth2.mjs',
    'e2e-write-auth.mjs',
    'e2e-dirbuild.mjs',
    'e2e-motion-map.mjs',
    'e2e-map3d-perf.mjs',
    'e2e-final-cross.mjs',
    'e2e-owner.mjs',
  ],
  offline: ['e2e-offline.mjs', 'e2e-offline-adversarial.mjs'],
}

const mode = process.argv[2]
if (!SUITES[mode]) {
  console.error(`usage: node tests/run-all.mjs ${Object.keys(SUITES).join(' | ')}`)
  process.exit(2)
}
for (const file of SUITES[mode]) {
  if (!fs.existsSync(path.join(TESTS_DIR, file))) {
    console.error(`[run-all] missing suite file: tests/${file}`)
    process.exit(2)
  }
}

const log = (msg) => console.log(`[run-all] ${msg}`)

// First port in the range that nobody is listening on.
function findFreePort() {
  return new Promise((resolve, reject) => {
    const tryPort = (port) => {
      if (port > PORT_LAST) return reject(new Error(`no free port in ${PORT_FIRST}-${PORT_LAST}`))
      const srv = net.createServer()
      srv.once('error', () => tryPort(port + 1))
      srv.listen({ port, host: '127.0.0.1', exclusive: true }, () => {
        srv.close(() => resolve(port))
      })
    }
    tryPort(PORT_FIRST)
  })
}

function waitForHttp(url, child) {
  const deadline = Date.now() + SERVER_READY_MS
  return new Promise((resolve, reject) => {
    const attempt = () => {
      if (child.exitCode !== null) return reject(new Error(`server exited early with code ${child.exitCode}`))
      if (Date.now() > deadline) return reject(new Error(`server not answering at ${url} after ${SERVER_READY_MS / 1000}s`))
      const req = http.get(url, (res) => {
        res.resume()
        if (res.statusCode && res.statusCode < 500) return resolve()
        setTimeout(attempt, 300)
      })
      req.on('error', () => setTimeout(attempt, 300))
      req.setTimeout(2000, () => req.destroy())
    }
    attempt()
  })
}

// Stop exactly the child we started, by its PID handle. Never by name or pattern.
function stopChild(child) {
  return new Promise((resolve) => {
    if (!child || child.exitCode !== null || child.signalCode !== null) return resolve()
    const force = setTimeout(() => child.kill('SIGKILL'), 5000)
    child.once('exit', () => {
      clearTimeout(force)
      resolve()
    })
    child.kill('SIGTERM')
  })
}

// The suite runs in its own process group. When it ends (pass, fail or crash), the whole group is killed, so a browser
// left behind by a crashed suite does not keep running and slow the next suite.
let currentSuite = null
function runNode(args, env, cwd) {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, args, { cwd, env, stdio: "inherit", detached: true })
    currentSuite = child
    const killGroup = (signal) => {
      try { process.kill(-child.pid, signal) } catch { /* group already gone */ }
    }
    const timer = setTimeout(() => {
      log(`timeout after ${SUITE_TIMEOUT_MS / 60000} min, stopping ${args.at(-1)}`)
      killGroup("SIGTERM")
    }, SUITE_TIMEOUT_MS)
    const done = (code, signal) => {
      clearTimeout(timer)
      killGroup("SIGKILL")
      currentSuite = null
      resolve({ code: code ?? 1, signal })
    }
    child.on("exit", (code, signal) => done(code, signal))
    child.on("error", () => done(1, null))
  })
}

async function startServer(port) {
  if (mode === 'e2e') {
    // Dev server: hot-module build, same as `npm run dev`.
    return spawn(process.execPath, [VITE_BIN, '--host', '127.0.0.1', '--port', String(port), '--strictPort'], {
      cwd: FRONTEND_DIR,
      stdio: ['ignore', 'inherit', 'inherit'],
    })
  }
  // Offline: the service worker only registers in production builds, so build, then serve the build.
  log('building production bundle for the offline suites')
  const build = await runNode(
    [VITE_BIN, 'build', '--outDir', OFFLINE_OUT, '--emptyOutDir'],
    process.env,
    FRONTEND_DIR,
  )
  if (build.code !== 0) throw new Error(`vite build failed with code ${build.code}`)
  return spawn(
    process.execPath,
    [VITE_BIN, 'preview', '--host', '127.0.0.1', '--port', String(port), '--strictPort', '--outDir', OFFLINE_OUT],
    { cwd: FRONTEND_DIR, stdio: ['ignore', 'inherit', 'inherit'] },
  )
}

async function main() {
  let server = null
  const onSignal = () => {
    if (currentSuite) { try { process.kill(-currentSuite.pid, "SIGKILL") } catch { /* gone */ } }
    stopChild(server).finally(() => process.exit(130))
  }
  process.on('SIGINT', onSignal)
  process.on('SIGTERM', onSignal)

  let results = []
  try {
    const port = await findFreePort()
    const base = `http://127.0.0.1:${port}`
    log(`starting ${mode === 'e2e' ? 'vite dev server' : 'vite preview of a production build'} at ${base}`)
    server = await startServer(port)
    await waitForHttp(`${base}/`, server)
    log(`server up (pid ${server.pid}), running ${SUITES[mode].length} suites sequentially`)

    for (const file of SUITES[mode]) {
      log(`--- ${file}`)
      const started = Date.now()
      const r = await runNode([path.join('tests', file)], { ...process.env, BASE: base }, FRONTEND_DIR)
      const ok = r.code === 0 && r.signal === null
      results.push({ file, ok, secs: ((Date.now() - started) / 1000).toFixed(1), code: r.code, signal: r.signal })
      log(`${ok ? 'PASS' : 'FAIL'} ${file} (${((Date.now() - started) / 1000).toFixed(1)}s, exit ${r.code}${r.signal ? `, signal ${r.signal}` : ''})`)
    }
  } catch (err) {
    log(`setup error: ${err.message}`)
    await stopChild(server)
    process.exit(2)
  }

  await stopChild(server)
  log('server stopped')

  const failed = results.filter((r) => !r.ok)
  console.log('\n[run-all] summary')
  for (const r of results) console.log(`  ${r.ok ? 'PASS' : 'FAIL'}  ${r.file}  ${r.secs}s`)
  console.log(`[run-all] ${results.length - failed.length}/${results.length} suites passed`)
  if (failed.length) {
    console.log(`[run-all] failed: ${failed.map((r) => r.file).join(', ')}`)
    process.exit(1)
  }
  process.exit(0)
}

main()
