/// <reference types="vitest" />
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',   // expose on LAN for phone access — see memory
                       // "dev-server-bind-all-interfaces"
    port: 5173,
    strictPort: true,  // fail loudly rather than picking another port
    proxy: {
      '/api':   { target: 'http://localhost:8765', changeOrigin: true },
      '/thumb': { target: 'http://localhost:8765', changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    chunkSizeWarningLimit: 800,
  },
  test: {
    // happy-dom is faster than jsdom and ships its own DOM
    // implementation — jsdom doesn't implement IntersectionObserver
    // anyway, so the env choice only affects React rendering, not the
    // observer mock.
    environment: 'happy-dom',
    // The Grid pulls in AppContext, App.tsx, asset endpoints, etc.
    // None of that needs to load for a single-hook test — keep the
    // test surface tiny.
    include: ['src/**/*.test.{ts,tsx}'],
  },
});