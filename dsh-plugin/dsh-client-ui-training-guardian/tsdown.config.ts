/**
 * Minimal two-config tsdown setup for DSH plugin.
 *
 * Target A (host): ESM for Node/cordis → dist/
 * Target B (client): CJS loader-factory for DSH browser → dist-client/
 */

import { defineConfig } from 'tsdown'

const CLIENT_ID = '@linxin666/dsh-client-ui-training-guardian'

export default defineConfig([
  // Host half: plain ESM
  {
    entry: { index: './src/index.ts' },
    outDir: 'dist',
    format: 'esm',
    platform: 'node',
    target: 'node22',
    dts: true,
    sourcemap: true,
  },

  // Client half: CJS + __ModuleLoader__.load wrapper
  {
    entry: { 'client/index': './src/client/index.ts' },
    outDir: 'dist-client',
    format: 'cjs',
    platform: 'browser',
    target: 'es2022',
    sourcemap: true,
    external: ['react', 'react/jsx-runtime', 'react-dom'],
    outputOptions: {
      banner: [
        `window.__ModuleLoader__.load({`,
        `  id: ${JSON.stringify(CLIENT_ID)},`,
        `  factory: (require) => {`,
        `    var module = { exports: {} };`,
        `    var exports = module.exports;`,
      ].join('\n'),
      footer: '    return module.exports;\n  }\n});',
    },
  },
])
