import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/optimize': 'http://localhost:8080',
      '/job': 'http://localhost:8080',
      '/health': 'http://localhost:8080',
      '/memory': 'http://localhost:8080',
      '/patterns': 'http://localhost:8080',
      '/parse-page': 'http://localhost:8080',
      '/score-layout': 'http://localhost:8080',
      '/optimize-block': 'http://localhost:8080',
      '/apply-edit': 'http://localhost:8080',
      '/export': 'http://localhost:8080',
      '/gaze-analysis': 'http://localhost:8080',
    },
  },
})
