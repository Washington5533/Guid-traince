import { defineConfig } from 'tsdown'

export default defineConfig({
  entry: {
    index: './src/index.ts',
    'client/index': './src/client/index.ts',
  },
  format: 'esm',
  dts: true,
  sourcemap: true,
  clean: true,
})
