/**
 * DSH browser-side entry for Training Guardian.
 *
 * Registered by the host half (src/index.ts) as a `dsh.client` entry.
 * Reads settings from the DSH settings scope, constructs the SSE client,
 * and injects the sidebar panel + settings card into DSH slots.
 */

import { createElement } from 'react'
import type { Context } from '@deepseek-ai/dsh-client-runtime'
import type { SettingsScope } from '@deepseek-ai/dsh-client-runtime/client'
import { TrainingGuardianAction } from './panel/TrainingGuardianAction'
import { SettingsCard, DEFAULT_SETTINGS, TgSettingsCardController } from './settings/SettingsCard'
import { SseClient } from './sse/client'
import { zh, en } from './locales'

/**
 * Services this bundle needs on the fiber `ctx` before apply() runs,
 * provided by the DSH client bundles listed in package.json
 * `dsh.client.inject`: `slots` (sidebar/panel injection), `locale`
 * (i18n registration) and `settingsScope` (settings binding).
 */
export const inject = ['slots', 'locale', 'settingsScope']

/**
 * DSH client plugin entry: the cordis loader invokes this with the fiber
 * `ctx` proxy directly (no options object), and every service it touches
 * must be declared in {@link inject}.
 */
export function apply(ctx: Context): void {
  const localeNs = 'training-guardian'

  // ---------- i18n ----------
  ctx.locale.register(localeNs, { zh, en })

  const t = (key: string): string => ctx.locale.t(`${localeNs}:${key}`)

  // ---------- settings ----------
  // The DSH client runtime exposes bind() on ctx.settingsScope when the
  // ui-settings bundle provides a settings scope; fall back to a
  // runtime-level bind if it is unavailable in this runtime version.
  const settingsBinder = (ctx.settingsScope ?? ctx) as {
    bind<S>(spec: { default: S; namespace: string }): SettingsScope<S>
  }

  let settingsScope: SettingsScope<{
    serverUrl: string
    authToken: string
    sessionId: string
    autoConnect: boolean
  }> | null = null

  try {
    settingsScope = settingsBinder.bind({
      default: DEFAULT_SETTINGS,
      namespace: 'training-guardian',
    })
  } catch {
    // Settings scope not available in this runtime version.
  }

  const controller = new TgSettingsCardController(settingsScope ?? {
    getSnapshot: () => ({ ...DEFAULT_SETTINGS } as never),
    setValue: () => {},
  } as unknown as SettingsScope<typeof DEFAULT_SETTINGS>)

  // ---------- SSE client ----------
  const sse = new SseClient({
    url: controller.getSnapshot().serverUrl,
    authToken: controller.getSnapshot().authToken || undefined,
  })

  // ---------- react to settings changes ----------
  controller.subscribe(() => {
    const s = controller.getSnapshot()
    sse.setSessionId(s.sessionId || null)
    if (sse.getStatus() !== 'connected') {
      sse.disconnect()
      sse.connect()
    }
  })

  // ---------- approve / reject -> REST ----------
  async function post(path: string, body: unknown): Promise<void> {
    const base = controller.getSnapshot().serverUrl.replace(/\/$/, '')
    const url = `${base}${path}`
    const headers: Record<string, string> = { 'Content-Type': 'application/json' }
    const token = controller.getSnapshot().authToken
    if (token) headers['Authorization'] = `Bearer ${token}`
    const res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    })
    if (!res.ok) {
      throw new Error(`REST ${path} failed: ${res.status}`)
    }
  }

  // Approve/reject are session-scoped on the guardian remote server:
  // POST /api/sessions/{session_id}/approve|reject with { action_id, reason? }.
  const requireSessionId = (): string => {
    const sessionId = controller.getSnapshot().sessionId.trim()
    if (sessionId === '') {
      throw new Error('决策审批需要先设置训练会话 ID（Training Guardian 设置 → sessionId）')
    }
    return sessionId
  }

  const onApprove = (actionId: string) =>
    post(`/api/sessions/${encodeURIComponent(requireSessionId())}/approve`, { action_id: actionId })
  const onReject = (actionId: string, reason: string) =>
    post(`/api/sessions/${encodeURIComponent(requireSessionId())}/reject`, { action_id: actionId, reason })

  // ---------- inject slots ----------
  // Session-header action: opens the monitoring panel popover. The seats are
  // declared by the dsh client runtime bundles (ui-conversation for
  // `conversation.session.header.actions`, ui-settings-plugins for the keyed
  // `settings.plugin.item` card), not invented by the plugin.
  const sseUrl = controller.getSnapshot().serverUrl

  try {
    ctx.slots.inject('conversation.session.header.actions', () =>
      ctx.slots.register({
        name: 'conversation.session.header.actions',
        id: 'training-guardian',
        order: 10,
      }, () => createElement(TrainingGuardianAction, {
        sse,
        sessionId: controller.getSnapshot().sessionId || null,
        serverUrl: sseUrl,
        modelEntry: undefined,
        projectDir: undefined,
        t,
        onApprove,
        onReject,
      })))
  } catch {
    // Slot not declared in this runtime; ignore gracefully.
  }

  // Settings card in the plugin configuration tab, keyed by the settings
  // namespace this plugin edits.
  try {
    ctx.slots.inject('settings.plugin.item', () =>
      ctx.slots.register({
        name: 'settings.plugin.item',
        key: 'training-guardian',
      }, () => createElement(SettingsCard, { controller, t })))
  } catch {
    // Optional slot — ignore when unavailable.
  }
}
