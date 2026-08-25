/**
 * DSH browser-side entry for Training Guardian.
 *
 * Registered by the host half (src/index.ts) as a `dsh.client` entry.
 * Reads settings from the DSH settings scope, constructs the SSE client,
 * and injects the sidebar panel + settings card into DSH slots.
 */

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
    modelEntry: string
    projectDir: string
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
    // url/token 变更必须重建 EventSource，不能只改 sessionId
    if (sse.setEndpoint({ url: s.serverUrl, authToken: s.authToken || undefined })) {
      sse.disconnect()
      sse.connect()
    } else if (sse.getStatus() !== 'connected') {
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
    if (token) headers['X-Auth-Token'] = token
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
  // Session-header action: opens the monitoring panel popover. Follows the
  // ui-jobs pattern — the inject factory returns business props, register
  // takes the React component directly, locale is declared on the options
  // so the framework wires `t` into the component props automatically.
  // NB: do NOT inject a custom `t` here — it would shadow the framework's
  // translator (ctx.locale.t is not part of the fiber surface in older
  // runtimes and crashes the slot entry).
  try {
    ctx.slots.inject('conversation.session.header.actions', () =>
      ctx.slots.register({
        name: 'conversation.session.header.actions',
        id: 'training-guardian',
        order: 10,
        locale: 'training-guardian',
        inject: (): Record<string, unknown> => ({
          sse,
          sessionId: controller.getSnapshot().sessionId || null,
          serverUrl: controller.getSnapshot().serverUrl,
          authToken: controller.getSnapshot().authToken || undefined,
          modelEntry: controller.getSnapshot().modelEntry || undefined,
          projectDir: controller.getSnapshot().projectDir || undefined,
          onApprove,
          onReject,
        }),
      }, TrainingGuardianAction))
    console.log('[TG] slot registration succeeded')
  } catch (e) {
    console.error('[TG] slot registration failed:', e)
  }

  // Settings card in the plugin configuration tab, keyed by the settings
  // namespace this plugin edits. `t` is framework-wired via `locale:` —
  // see the note on the header slot above.
  try {
    ctx.slots.inject('settings.plugin.item', () =>
      ctx.slots.register({
        name: 'settings.plugin.item',
        key: 'training-guardian',
        locale: 'training-guardian',
        inject: (): Record<string, unknown> => ({
          controller,
        }),
      }, SettingsCard))
    console.log('[TG] settings slot registration succeeded')
  } catch (e) {
    console.error('[TG] settings slot registration failed:', e)
  }

  // ---------- skill registration ----------
  // Runtime skill registration is best-effort: only newer DSH runtimes
  // expose a `skills` fiber service. The skill is always declared in
  // cordis.patch.yml `meta.skills` (the canonical channel); this block adds
  // the `invoke` behavior when the runtime supports it. Never let it fail
  // the boot, and stay quiet on runtimes without the service.
  try {
    const skills = (ctx as unknown as { skills?: { register(id: string, def: Record<string, unknown>): void } }).skills
    if (skills) {
      skills.register('training-guardian', {
        id: 'training-guardian',
        name: 'Training Guardian',
        description: 'Real-time training metrics, GPU status, anomaly feed, and sub-agent decision approval',
        whenToUse: 'Use when the user asks about training status, GPU utilization, loss curves, anomalies, or sub-agent decisions',
        modelInvocable: true,
        userInvocable: true,
        invoke: () => {
          // Click the header action button to open the panel popover.
          const btn = document.querySelector(
            '[data-slot-id="training-guardian"], [aria-label*="Guardian"], button:has-text("Training Guardian")'
          )
          ;(btn as HTMLButtonElement | null)?.click()
        },
      })
      console.log('[TG] skill registration succeeded')
    }
  } catch (e) {
    // e.g. "cannot get property skills without inject" on runtimes where the
    // skill system is not a fiber service — manifest declaration covers it.
    console.log('[TG] runtime skill service unavailable; manifest declaration applies')
  }
}
