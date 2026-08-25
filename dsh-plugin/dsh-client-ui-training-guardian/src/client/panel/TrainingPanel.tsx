import { useState, useEffect, useCallback, useRef } from 'react'
import type { SseClient, SseError } from '../sse/client'
import type { TgKey } from '../locales'
import { MetricsTab } from './MetricsTab'
import { GpuTab } from './GpuTab'
import { AnomaliesTab } from './AnomaliesTab'
import { DecisionsTab } from './DecisionsTab'
import { ArchTab, type ArchNarration } from './ArchTab'

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
}

type Tab = 'overview' | 'gpu' | 'anomalies' | 'decisions' | 'arch'

const TABS: { key: Tab; labelKey: TgKey }[] = [
  { key: 'overview', labelKey: 'tab.overview' },
  { key: 'gpu', labelKey: 'tab.gpu' },
  { key: 'anomalies', labelKey: 'tab.anomalies' },
  { key: 'decisions', labelKey: 'tab.decisions' },
  { key: 'arch', labelKey: 'tab.arch' },
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

export function TrainingPanel({ sse, sessionId, t, onApprove, onReject, serverUrl = '', authToken, modelEntry, projectDir }: TrainingPanelProps) {
  const [activeTab, setActiveTab] = useState<Tab>('overview')
  const [metrics, setMetrics] = useState<Record<string, unknown>>({})
  const [gpuStatus, setGpuStatus] = useState<Record<string, unknown> | null>(null)
  const [anomalies, setAnomalies] = useState<Array<Record<string, unknown>>>([])
  const [pendingActions, setPendingActions] = useState<Array<Record<string, unknown>>>([])
  const [connectionStatus, setConnectionStatus] = useState<string>('disconnected')
  const [connError, setConnError] = useState<SseError | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [archNarration, setArchNarration] = useState<ArchNarration | null>(null)

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
    unsubs.push(sse.on('metrics', (data) => {
      setMetrics(prev => ({ ...prev, ...(data as Record<string, unknown>) }))
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

    sse.connect()
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

  // Build the connection status bar content.
  const statusContent = (() => {
    if (connected) return null
    if (connectionStatus === 'connecting') return t('panel.connecting')
    // Disconnected or error — show diagnosis if available.
    if (connError) {
      const kindKey = ERROR_KIND_KEY[connError.kind] || 'conn.unknown'
      const hintKey = ERROR_HINT_KEY[connError.kind]
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
            {connError.kind === 'exhausted' && (
              <button
                onClick={() => sse.reconnectNow()}
                style={{
                  padding: '1px 8px', fontSize: 10, cursor: 'pointer',
                  background: 'var(--accent, #007acc)', color: '#fff',
                  border: 'none', borderRadius: 3,
                }}
              >
                {t('conn.retry')}
              </button>
            )}
          </div>
          {hintKey && (
            <div style={{ fontSize: 10, opacity: 0.7 }}>{t(hintKey)}</div>
          )}
          {connError.detail && (
            <div style={{ fontSize: 10, opacity: 0.6 }}>{connError.detail}</div>
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
          background: connected ? 'transparent' : 'var(--warning-bg, #332200)',
          color: connected ? 'var(--text)' : 'var(--warning-text, #ffa726)',
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
          <MetricsTab metrics={metrics} t={t} />
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
      </div>
    </div>
  )
}
