import { describe, expect, it, vi } from 'vitest'
import { apply } from '../src/client/index'
import { DEFAULT_SETTINGS } from '../src/client/settings/SettingsCard'

/**
 * PLUGIN_STANDARDS §7 integration tests:
 * apply() must run on a mock ctx, inject slots without throwing,
 * and register i18n + the plugin skill.
 */

function mockCtx() {
  const injected: string[] = []
  const registered: unknown[] = []
  const localeCalls: unknown[] = []
  const skillCalls: Array<[string, Record<string, unknown>]> = []

  const scope = {
    getSnapshot: () => ({ status: 'ready', value: { ...DEFAULT_SETTINGS } }),
    setValue: vi.fn(),
  }

  const ctx = {
    locale: {
      register: vi.fn((ns: string, dict: unknown) => {
        localeCalls.push([ns, dict])
      }),
      t: (key: string) => key,
    },
    settingsScope: {
      bind: vi.fn(() => scope),
    },
    slots: {
      inject: vi.fn((name: string, factory: () => unknown) => {
        injected.push(name)
        return factory()
      }),
      register: vi.fn((opts: unknown, component: unknown) => {
        registered.push({ opts, component })
        return component
      }),
    },
    skills: {
      register: vi.fn((id: string, def: Record<string, unknown>) => {
        skillCalls.push([id, def])
      }),
    },
  }

  return { ctx, injected, registered, localeCalls, skillCalls }
}

describe('apply() integration', () => {
  it('applies to a mock DSH ctx without throwing', () => {
    const { ctx } = mockCtx()
    expect(() => apply(ctx as never)).not.toThrow()
  })

  it('injects both DSH slots', () => {
    const { ctx, injected, registered } = mockCtx()
    apply(ctx as never)

    expect(injected.sort()).toEqual([
      'conversation.session.header.actions',
      'settings.plugin.item',
    ])
    expect(registered).toHaveLength(2)
  })

  it('registers locales and the plugin skill', () => {
    const { ctx, localeCalls, skillCalls } = mockCtx()
    apply(ctx as never)

    expect(localeCalls).toHaveLength(1)
    const [ns, dict] = localeCalls[0] as [string, { zh: unknown; en: unknown }]
    expect(ns).toBe('training-guardian')
    expect(Object.keys(dict.zh)).toHaveLength(Object.keys(dict.en).length)

    expect(skillCalls).toHaveLength(1)
    const [skillId, def] = skillCalls[0]
    expect(skillId).toBe('training-guardian')
    expect(def.name).toBeTruthy()
    expect(def.description).toBeTruthy()
    expect(typeof def.invoke).toBe('function')
  })

  it('degrades gracefully when settings scope is unavailable', () => {
    const { ctx } = mockCtx()
    const broken = { ...ctx, settingsScope: { bind: vi.fn(() => { throw new Error('no scope') }) } }
    expect(() => apply(broken as never)).not.toThrow()
  })
})
