import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  define: {
    // Expose env vars to the browser bundle
    'import.meta.env.VITE_API_BASE_URL': JSON.stringify(process.env.VITE_API_BASE_URL || 'http://localhost:8000'),
    'import.meta.env.VITE_DASHBOARD_URL': JSON.stringify(process.env.VITE_DASHBOARD_URL || 'http://localhost:8501'),
  }
})
