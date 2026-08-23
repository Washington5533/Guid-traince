import type { FormEvent } from 'react'
import { useState, useEffect, useCallback } from 'react'
import type { SettingsScope } from '@deepseek-ai/dsh-client-runtime/client'

export interface TrainingGuardianSettings {
  serverUrl: string
  authToken: string
  sessionId: string
  autoConnect: boolean
}

export const DEFAULT_SETTINGS: TrainingGuardianSettings = {
  serverUrl: 'http://localhost:8765',
  authToken: '',
  sessionId: '',
  autoConnect: true,
}

export type TgSettingsCardFace = {
  onChange(settings: Partial<TrainingGuardianSettings>): void
}

export class TgSettingsCardController {
  private scope: SettingsScope<TrainingGuardianSettings>
  private listeners = new Set<() => void>()

  constructor(scope: SettingsScope<TrainingGuardianSettings>) {
    this.scope = scope
  }

  getSnapshot(): TrainingGuardianSettings {
    const raw = this.scope.getSnapshot()
    if (raw.status === 'ready' && raw.value) {
      return { ...DEFAULT_SETTINGS, ...raw.value }
    }
    return { ...DEFAULT_SETTINGS }
  }

  subscribe(fn: () => void): () => void {
    this.listeners.add(fn)
    return () => { this.listeners.delete(fn) }
  }

  update(partial: Partial<TrainingGuardianSettings>): void {
    const current = this.getSnapshot()
    const next = { ...current, ...partial }
    this.scope.setValue(next)
    this.listeners.forEach(fn => { try { fn() } catch { /* swallow */ } })
  }

  dispose(): void {
    this.listeners.clear()
  }

  inject(): ReturnType<typeof import('react').createElement> {
    // The actual JSX is rendered by the React component registered via slots.
    // This method exists for slot injection compatibility.
    return null as unknown as ReturnType<typeof import('react').createElement>
  }
}

interface SettingsCardProps {
  controller: TgSettingsCardController
  t: (key: string) => string
}

export function SettingsCard({ controller, t }: SettingsCardProps) {
  const [settings, setSettings] = useState<TrainingGuardianSettings>(controller.getSnapshot())
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    const unsub = controller.subscribe(() => {
      setSettings(controller.getSnapshot())
    })
    return unsub
  }, [controller])

  const handleSubmit = useCallback((e: FormEvent) => {
    e.preventDefault()
    controller.update(settings)
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }, [controller, settings])

  const handleChange = useCallback((field: keyof TrainingGuardianSettings, value: string | boolean) => {
    setSettings(prev => ({ ...prev, [field]: value }))
  }, [])

  return (
    <form onSubmit={handleSubmit} style={{ padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div>
        <label style={{ display: 'block', fontSize: 13, fontWeight: 500, marginBottom: 4 }}>
          {t('settings.serverUrl')}
        </label>
        <input
          type="text"
          value={settings.serverUrl}
          onChange={e => handleChange('serverUrl', e.target.value)}
          placeholder={t('settings.serverUrlHint')}
          style={{
            width: '100%', padding: '6px 8px', fontSize: 13,
            border: '1px solid var(--border, #ddd)', borderRadius: 4,
            background: 'var(--bg, #fff)', color: 'var(--text, #333)',
          }}
        />
      </div>

      <div>
        <label style={{ display: 'block', fontSize: 13, fontWeight: 500, marginBottom: 4 }}>
          {t('settings.authToken')}
        </label>
        <input
          type="password"
          value={settings.authToken}
          onChange={e => handleChange('authToken', e.target.value)}
          placeholder={t('settings.authTokenHint')}
          style={{
            width: '100%', padding: '6px 8px', fontSize: 13,
            border: '1px solid var(--border, #ddd)', borderRadius: 4,
            background: 'var(--bg, #fff)', color: 'var(--text, #333)',
          }}
        />
      </div>

      <div>
        <label style={{ display: 'block', fontSize: 13, fontWeight: 500, marginBottom: 4 }}>
          {t('settings.sessionId')}
        </label>
        <input
          type="text"
          value={settings.sessionId}
          onChange={e => handleChange('sessionId', e.target.value)}
          placeholder={t('settings.sessionIdHint')}
          style={{
            width: '100%', padding: '6px 8px', fontSize: 13,
            border: '1px solid var(--border, #ddd)', borderRadius: 4,
            background: 'var(--bg, #fff)', color: 'var(--text, #333)',
          }}
        />
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <input
          type="checkbox"
          id="tg-auto-connect"
          checked={settings.autoConnect}
          onChange={e => handleChange('autoConnect', e.target.checked)}
        />
        <label htmlFor="tg-auto-connect" style={{ fontSize: 13 }}>
          {t('settings.autoConnect')}
        </label>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <button
          type="submit"
          style={{
            padding: '6px 16px', fontSize: 13, cursor: 'pointer',
            background: 'var(--accent, #007acc)', color: '#fff',
            border: 'none', borderRadius: 4,
          }}
        >
          {saved ? t('settings.saved') : 'Save'}
        </button>
      </div>
    </form>
  )
}
