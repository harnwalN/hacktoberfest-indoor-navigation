import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev: proxy API + frames to FastAPI so the app only ever uses relative URLs (works on the phone too).
// API_TARGET points the dev server at another machine's backend (e.g. a teammate's tunnel URL), so
// frontend work needs no local Gemma or map data:  API_TARGET=https://xyz.trycloudflare.com npm run dev
// Build: emit straight into backend/static so FastAPI serves app + API from one origin.
const API = process.env.API_TARGET || 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
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
