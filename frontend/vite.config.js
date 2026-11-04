import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev: proxy API + frames to FastAPI so the app only ever uses relative URLs (works on the phone too).
// API_TARGET points the dev server at another machine's backend (e.g. a teammate's tunnel URL), so
// frontend work needs no local Gemma or map data:  API_TARGET=https://xyz.trycloudflare.com npm run dev
// Build: emit straight into backend/static so FastAPI serves app + API from one origin.
const API = process.env.API_TARGET || 'http://127.0.0.1:8000'

// The lazy pages' chunks (and what they import, plus their CSS) are listed in /lazy-chunks.json. public/sw.js
// precaches that list at install, so every page works offline after one online visit, not only the pages visited.
function lazyChunkList() {
  return {
    name: 'waypoint-lazy-chunk-list',
    apply: 'build',
    generateBundle(_options, bundle) {
      const js = new Set()
      const css = new Set()
      const visit = (name) => {
        if (js.has(name)) return
        const chunk = bundle[name]
        if (!chunk || chunk.type !== 'chunk' || chunk.isEntry) return // the entry is in the shell already
        js.add(name)
        for (const c of chunk.viteMetadata?.importedCss ?? []) css.add(c)
        for (const i of chunk.imports) visit(i)
      }
      for (const item of Object.values(bundle)) if (item.type === 'chunk' && item.isDynamicEntry) visit(item.fileName)
      const files = [...js, ...css].map((f) => `/${f}`).sort()
      this.emitFile({ type: 'asset', fileName: 'lazy-chunks.json', source: JSON.stringify(files) })
    },
  }
}

export default defineConfig({
  plugins: [react(), lazyChunkList()],
  server: {
    host: true,
    allowedHosts: true,
    proxy: {
      '/api': { target: API, changeOrigin: true },
      '/bfiles': { target: API, changeOrigin: true },
    },
  },
  build: { outDir: '../backend/static', emptyOutDir: true },
})
