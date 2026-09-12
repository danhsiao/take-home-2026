import path from 'node:path'

import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': path.resolve(import.meta.dirname, './src') },
  },
  server: {
    // The FastAPI catalogue runs as a separate process in development. Proxying keeps
    // the frontend origin-relative, so no CORS handling or base-URL config is needed.
    proxy: { '/api': 'http://localhost:8000' },
  },
})
