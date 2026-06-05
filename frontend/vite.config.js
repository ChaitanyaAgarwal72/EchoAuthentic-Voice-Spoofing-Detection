import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    // onnxruntime-web uses dynamic require() for its WASM paths — excluding it
    // from Vite's pre-bundler keeps those paths intact so we can redirect them
    // to CDN at runtime via ort.env.wasm.wasmPaths.
    exclude: ['onnxruntime-web'],
  },
})


