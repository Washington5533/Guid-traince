import { useState, useEffect, useCallback, useRef } from 'react'
import type { SseClient, SseError } from '../sse/client'
import type { TgKey } from '../locales'
import { MetricsTab } from './MetricsTab'
import { GpuTab } from './GpuTab'
import { AnomaliesTab } from './AnomaliesTab'
import { DecisionsTab } from './DecisionsTab'
import { ArchTab, type ArchNarration } from './ArchTab'
import { HistoryTab } from './HistoryTab'
import {
  MAX_HISTORY,
  loadMetricsHistory,
  saveMetricsHistory,
  clearMetricsHistory,
  registerLocalSession,
} from '../storage'

export interface TrainingPanelProps {
  sse: SseClient
  sessionId: string | null
  t: (key: TgKey) => string
  onApprove: (actionId: string) => void | Promise<void>
  onReject: (actionId: string, reason: string) => void | Promise<void>
  serverUrl?: string
  authToken?: string
  modelEntry?: string
  projectDir?: string
  autoConnect?: boolean
}

type Tab = 'overview' | 'gpu' | 'anomalies' | 'decisions' | 'arch' | 'history'

const TABS: { key: Tab; labelKey: TgKey }[] = [
  { key: 'overview', labelKey: 'tab.overview' },
  { key: 'gpu', labelKey: 'tab.gpu' },
  { key: 'anomalies', labelKey: 'tab.anomalies' },
  { key: 'decisions', labelKey: 'tab.decisions' },
  { key: 'arch', labelKey: 'tab.arch' },
  { key: 'history', labelKey: 'tab.history' },
]

/** Map SseError.kind to a locale key for the diagnosis headline. */
const ERROR_KIND_KEY: Record<string, TgKey> = {
  unauthorized: 'conn.unauthorized',
  not_found: 'conn.notFound',
  server_error: 'conn.serverError',
  unreachable: 'conn.unreachable',
  exhausted: 'conn.exhausted',
  unknown: 'conn.unknown',
}

const ERROR_HINT_KEY: Record<string, TgKey> = {
  unauthorized: 'conn.hintUnauthorized',
  not_found: 'conn.hintNotFound',
  server_error: 'conn.hintServerError',
  unreachable: 'conn.hintUnreachable',
}

/** Copyable remediation commands shown inside the advice block. */
const CMD_WATCH = 'guarftrain watch --remote -- python train.py --epochs 50'
const cmdRemote = (port: string) => `guarftrain remote --port ${port}`

/**
 * Per-failure-kind actionable advice: a localized suggestion plus, where the
 * fix is "start the server", the exact command to run on the training machine.
 */
const ERROR_ADVICE: Record<string, { key: TgKey; cmd?: 'remote' | 'watch' }> = {
  unreachable: { key: 'conn.adviceUnreachable', cmd: 'remote' },
  exhausted: { key: 'conn.adviceExhausted', cmd: 'remote' },
  unauthorized: { key: 'conn.adviceUnauthorized' },
  not_found: { key: 'conn.adviceNotFound' },
  server_error: { key: 'conn.adviceServerError', cmd: 'watch' },
}

export function TrainingPanel({ sse, sessionId, t, onApprove, onReject, serverUrl = '', authToken, modelEntry, projectDir, autoConnect = true }: TrainingPanelProps) {
  const [activeTab, setActiveTab] = useState<Tab>('overview')
  const [metrics, setMetrics] = useState<Record<string, unknown>>({})
  // Effective storage key: use sessionId when set, fall back to 'auto' so
  // data always persists — even when the user hasn't configured a session ID.
  const effectiveKey = (sessionId && sessionId.trim()) || 'auto'

  const [metricsHistory, setMetricsHistory] = useState<Array<Record<string, unknown>>>(() => {
    // Restore persisted history for current session on mount.
    return loadMetricsHistory(effectiveKey)
  })
  const [gpuStatus, setGpuStatus] = useState<Record<string, unknown> | null>(null)
  const [anomalies, setAnomalies] = useState<Array<Record<string, unknown>>>([])
  const [pendingActions, setPendingActions] = useState<Array<Record<string, unknown>>>([])
  const [connectionStatus, setConnectionStatus] = useState<string>('disconnected')
  const [connError, setConnError] = useState<SseError | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [archNarration, setArchNarration] = useState<ArchNarration | null>(null)
  // Transient "copied" feedback for the advice command box.
  const [copied, setCopied] = useState(false)

  // Debounced persistence of metrics history to localStorage + registry.
  const persistTimer = useRef<number | null>(null)
  useEffect(() => {
    if (metricsHistory.length === 0) return
    if (persistTimer.current) window.clearTimeout(persistTimer.current)
    persistTimer.current = window.setTimeout(() => {
      saveMetricsHistory(effectiveKey, metricsHistory)
      // Register in the local session registry so HistoryTab can find it.
      const last = metricsHistory[metricsHistory.length - 1] || {}
      registerLocalSession(effectiveKey, metricsHistory.length, last)
    }, 500)
    return () => { if (persistTimer.current) window.clearTimeout(persistTimer.current) }
  }, [metricsHistory, effectiveKey])

  // Reload history when the effective session key changes.
  const prevKeyRef = useRef<string>(effectiveKey)
  useEffect(() => {
    if (prevKeyRef.current !== effectiveKey) {
      prevKeyRef.current = effectiveKey
      setMetricsHistory(loadMetricsHistory(effectiveKey))
    }
  }, [effectiveKey])

  useEffect(() => {
    const unsubs: Array<() => void> = []

    unsubs.push(sse.onStatusChange((status) => {
      setConnectionStatus(status)
    }))

    // Subscribe to diagnosed connection errors from the SSE client.
    unsubs.push(sse.onError((err) => {
      setConnError(err)
    }))

    // Merge incoming metrics fields rather than overwriting the whole object.
    // Also accumulate into history for live chart rendering.
    unsubs.push(sse.on('metrics', (data) => {
      const d = data as Record<string, unknown>
      setMetrics(prev => ({ ...prev, ...d }))
      setMetricsHistory(prev => {
        const next = [...prev, d]
        return next.length > MAX_HISTORY ? next.slice(next.length - MAX_HISTORY) : next
      })
    }))

    unsubs.push(sse.on('gpu_status', (data) => {
      setGpuStatus(data as Record<string, unknown>)
    }))

    unsubs.push(sse.on('anomaly', (data) => {
      setAnomalies(prev => [data as Record<string, unknown>, ...prev].slice(0, 50))
    }))

    unsubs.push(sse.on('decision', (data) => {
      setPendingActions(prev => [data as Record<string, unknown>, ...prev].slice(0, 20))
    }))

    // Agent-driven arch analysis: narration arrives via SSE after REST returns.
    unsubs.push(sse.on('arch_analysis', (data) => {
      const d = data as Record<string, unknown>
      setArchNarration({
        narration: (d.narration as string) ?? null,
        error: (d.error as string) ?? null,
        model_name: (d.model_name as string) ?? '',
        total_params: (d.total_params as number) ?? 0,
        bottleneck_count: (d.bottleneck_count as number) ?? 0,
      })
    }))

    // Respect autoConnect setting: false means start in idle state.
    if (autoConnect) {
      sse.connect()
    } else {
      sse.goIdle()
    }
    return () => {
      unsubs.forEach(fn => { try { fn() } catch { /* swallow */ } })
    }
  }, [sse])

  // Wrap approve/reject with error feedback so failures don't become
  // unhandled promise rejections.
  const safeApprove = useCallback((actionId: string) => {
    setActionError(null)
    try {
      const result = onApprove(actionId)
      if (result && typeof result.catch === 'function') {
        result.catch(e => setActionError(String(e)))
      }
    } catch (e) {
      setActionError(String(e))
    }
  }, [onApprove])

  const safeReject = useCallback((actionId: string, reason: string) => {
    setActionError(null)
    try {
      const result = onReject(actionId, reason)
      if (result && typeof result.catch === 'function') {
        result.catch(e => setActionError(String(e)))
      }
    } catch (e) {
      setActionError(String(e))
    }
  }, [onReject])

  const connected = connectionStatus === 'connected'
  const idle = connectionStatus === 'idle'

  // Build the connection status bar content.
  const statusContent = (() => {
    if (connected) return null
    // Idle state: neutral blue, manual connect button.
    if (idle) {
      return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
            <span>{t('panel.idle')}</span>
            <button
              onClick={() => sse.connect()}
              style={{
                padding: '1px 10px', fontSize: 10, cursor: 'pointer',
                background: 'var(--accent, #007acc)', color: '#fff',
                border: 'none', borderRadius: 3,
              }}
            >
              {t('panel.idleConnect')}
            </button>
          </div>
          <div style={{ fontSize: 10, opacity: 0.75, textAlign: 'center' }}>
            {t('panel.idleHint')}{' '}
            <code style={{ fontFamily: 'ui-monospace, Consolas, monospace' }}>{CMD_WATCH}</code>
          </div>
        </div>
      )
    }
    if (connectionStatus === 'connecting') return t('panel.connecting')
    // Disconnected or error — show diagnosis if available.
    if (connError) {
      const kindKey = ERROR_KIND_KEY[connError.kind] || 'conn.unknown'
      const hintKey = ERROR_HINT_KEY[connError.kind]
      const advice = ERROR_ADVICE[connError.kind]
      const port = /:(\d+)(\/|$)/.exec(serverUrl)?.[1] || '8765'
      const adviceCmd = advice?.cmd === 'watch'
        ? CMD_WATCH
        : advice?.cmd === 'remote'
          ? cmdRemote(port)
          : null
      const attemptInfo = connError.kind !== 'exhausted'
        ? ` (${connError.attempt}/${connError.maxAttempts})`
        : ''
      return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span>{t(kindKey)}{attemptInfo}</span>
            {connError.status !== null && (
              <span style={{ opacity: 0.7 }}>HTTP {connError.status}</span>
            )}
          </div>
          {hintKey && (
            <div style={{ fontSize: 10, opacity: 0.7 }}>{t(hintKey)}</div>
          )}
          {connError.detail && (
            <div style={{ fontSize: 10, opacity: 0.6 }}>{connError.detail}</div>
          )}
          {advice && (
            <div style={{
              marginTop: 4, padding: '6px 8px', borderRadius: 4,
              background: 'rgba(0,0,0,0.25)', border: '1px solid var(--border, #333)',
              display: 'flex', flexDirection: 'column', gap: 4, textAlign: 'left',
            }}>
              <div style={{ fontSize: 11 }}>
                <span style={{ fontWeight: 600 }}>{t('conn.adviceLabel')}</span>
                <span style={{ opacity: 0.85 }}> — {t(advice.key)}</span>
              </div>
              {adviceCmd && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <span style={{ fontSize: 10, opacity: 0.6 }}>{t('conn.adviceCmdLabel')}:</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <code style={{
                      flex: 1, fontFamily: 'ui-monospace, Consolas, monospace', fontSize: 11,
                      background: 'rgba(0,0,0,0.35)', padding: '3px 6px', borderRadius: 3,
                      overflowX: 'auto', whiteSpace: 'nowrap',
                    }}>
                      {adviceCmd}
                    </code>
                    <button
                      onClick={() => {
                        navigator.clipboard?.writeText(adviceCmd).catch(() => { /* clipboard denied */ })
                        setCopied(true)
                        window.setTimeout(() => setCopied(false), 2000)
                      }}
                      style={{
                        padding: '2px 8px', fontSize: 10, cursor: 'pointer',
                        background: 'var(--accent, #007acc)', color: '#fff',
                        border: 'none', borderRadius: 3,
                      }}
                    >
                      {copied ? '✓' : t('conn.adviceCopy')}
                    </button>
                  </div>
                  <div style={{ fontSize: 10, opacity: 0.6 }}>{t('conn.adviceExtra')}</div>
                </div>
              )}
              {connError.kind === 'exhausted' && (
                <button
                  onClick={() => sse.reconnectNow()}
                  style={{
                    alignSelf: 'flex-start',
                    padding: '2px 10px', fontSize: 10, cursor: 'pointer',
                    background: 'var(--accent, #007acc)', color: '#fff',
                    border: 'none', borderRadius: 3,
                  }}
                >
                  {t('conn.retry')}
                </button>
              )}
            </div>
          )}
        </div>
      )
    }
    return t('panel.disconnected')
  })()

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', height: '100%',
      background: 'var(--panel-bg, #1a1a2e)', color: 'var(--text, #e0e0e0)',
      fontFamily: 'system-ui, -apple-system, sans-serif', fontSize: 13,
    }}>
      {/* Tab bar */}
      <div style={{
        display: 'flex', borderBottom: '1px solid var(--border, #333)',
        background: 'var(--tab-bg, #16162a)',
      }}>
        {TABS.map(tab => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            style={{
              flex: 1, padding: '8px 4px', fontSize: 12, cursor: 'pointer',
              border: 'none', borderBottom: activeTab === tab.key ? '2px solid var(--accent, #007acc)' : '2px solid transparent',
              background: activeTab === tab.key ? 'var(--tab-active, #1e1e3a)' : 'transparent',
              color: activeTab === tab.key ? 'var(--accent, #4fc3f7)' : 'var(--text-secondary, #888)',
              transition: 'all 0.15s',
            }}
          >
            {t(tab.labelKey)}
          </button>
        ))}
      </div>

      {/* Connection status bar */}
      {statusContent && (
        <div style={{
          padding: '4px 12px', fontSize: 11,
          background: idle ? 'var(--info-bg, #0a2540)' : connected ? 'transparent' : 'var(--warning-bg, #332200)',
          color: idle ? 'var(--info-text, #4fc3f7)' : connected ? 'var(--text)' : 'var(--warning-text, #ffa726)',
          textAlign: 'center',
        }}>
          {typeof statusContent === 'string' ? statusContent : statusContent}
        </div>
      )}

      {/* Action error feedback */}
      {actionError && (
        <div style={{
          padding: '4px 12px', fontSize: 11,
          background: '#331111', color: '#ff6666', textAlign: 'center',
        }}>
          {actionError}
          <button
            onClick={() => setActionError(null)}
            style={{ marginLeft: 8, background: 'none', border: 'none', color: '#888', cursor: 'pointer', fontSize: 10 }}
          >✕</button>
        </div>
      )}

      {/* Tab content */}
      <div style={{ flex: 1, overflow: 'auto' }}>
        {activeTab === 'overview' && (
          <MetricsTab
            metrics={metrics}
            metricsHistory={metricsHistory}
            t={t}
            onClearHistory={() => {
              setMetricsHistory([])
              clearMetricsHistory(effectiveKey)
            }}
          />
        )}
        {activeTab === 'gpu' && (
          <GpuTab gpuStatus={gpuStatus} t={t} />
        )}
        {activeTab === 'anomalies' && (
          <AnomaliesTab anomalies={anomalies} t={t} />
        )}
        {activeTab === 'decisions' && (
          <DecisionsTab
            pending={pendingActions}
            t={t}
            onApprove={safeApprove}
            onReject={safeReject}
          />
        )}
        {activeTab === 'arch' && (
          <ArchTab
            serverUrl={serverUrl}
            authToken={authToken}
            sessionId={sessionId}
            modelEntry={modelEntry}
            projectDir={projectDir}
            t={t}
            archNarration={archNarration}
          />
        )}
        {activeTab === 'history' && (
          <HistoryTab
            serverUrl={serverUrl}
            authToken={authToken}
            t={t}
            currentSessionKey={effectiveKey}
          />
        )}
      </div>
    </div>
  )
}
