import { defineConfig } from 'vite'

export default defineConfig(async () => {
  let reactPlugin = null
  try {
    // Optional: load React plugin if available
    const mod = await import('@vitejs/plugin-react')
    reactPlugin = mod && (mod.default ? mod.default() : mod())
  } catch {
    // Plugin not installed; fall back to default config
  }
  return {
    plugins: reactPlugin ? [reactPlugin] : [],
  }
})
