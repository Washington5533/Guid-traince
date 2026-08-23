import { describe, it, expect } from 'vitest'
import { zh, en } from '../src/client/locales'
import type { TgKey } from '../src/client/locales'

describe('locales', () => {
  it('both locales contain every key in TgKey union', () => {
    // Collect all valid keys by iterating the union type's values.
    // We use a brute-force approach: pick the value set from one locale,
    // then assert the other has the same set.
    const zhKeys = new Set(Object.keys(zh) as TgKey[])
    const enKeys = new Set(Object.keys(en) as TgKey[])

    expect([...zhKeys].sort()).toEqual([...enKeys].sort())

    for (const k of zhKeys) {
      expect(zh[k]).toBeTruthy()
      expect(en[k]).toBeTruthy()
    }
  })

  it('zh and en are not identical (useful i18n sanity check)', () => {
    let diff = 0
    for (const k of Object.keys(zh) as TgKey[]) {
      if (zh[k] !== en[k]) diff++
    }
    expect(diff).toBeGreaterThan(0)
  })
})
