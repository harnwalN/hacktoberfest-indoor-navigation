import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev: proxy API + frames to FastAPI so the app only ever uses relative URLs (works on the phone too).
// Build: emit straight into backend/static so FastAPI serves app + API from one origin.
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    allowedHosts: true,
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/bfiles': 'http://127.0.0.1:8000',
    },
  },
  build: { outDir: '../backend/static', emptyOutDir: true },
})
