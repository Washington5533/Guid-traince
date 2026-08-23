import { useState, useEffect, useCallback, useRef } from 'react'
import type { SseClient } from '../sse/client'
import type { TgKey } from '../locales'
import { MetricsTab } from './MetricsTab'
import { GpuTab } from './GpuTab'
import { AnomaliesTab } from './AnomaliesTab'
import { DecisionsTab } from './DecisionsTab'
import { ArchTab } from './ArchTab'

export interface TrainingPanelProps {
  sse: SseClient
  sessionId: string | null
  t: (key: TgKey) => string
  onApprove: (actionId: string) => void
  onReject: (actionId: string, reason: string) => void
  serverUrl?: string
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

export function TrainingPanel({ sse, sessionId, t, onApprove, onReject, serverUrl = '', modelEntry, projectDir }: TrainingPanelProps) {
  const [activeTab, setActiveTab] = useState<Tab>('overview')
  const [metrics, setMetrics] = useState<Record<string, unknown>>({})
  const [gpuStatus, setGpuStatus] = useState<Record<string, unknown> | null>(null)
  const [anomalies, setAnomalies] = useState<Array<Record<string, unknown>>>([])
  const [pendingActions, setPendingActions] = useState<Array<Record<string, unknown>>>([])
  const [connectionStatus, setConnectionStatus] = useState<string>('disconnected')
  const unsubscribers = useRef<Array<() => void>>([])

  useEffect(() => {
    const unsubs: Array<() => void> = []

    const unsubStatus = sse.onStatusChange((status) => {
      setConnectionStatus(status)
    })
    unsubs.push(unsubStatus)

    const unsubMetrics = sse.on('metrics', (data) => {
      setMetrics(data as Record<string, unknown>)
    })
    unsubs.push(unsubMetrics)

    const unsubGpu = sse.on('gpu_status', (data) => {
      setGpuStatus(data as Record<string, unknown>)
    })
    unsubs.push(unsubGpu)

    const unsubAnomaly = sse.on('anomaly', (data) => {
      setAnomalies(prev => [data as Record<string, unknown>, ...prev].slice(0, 50))
    })
    unsubs.push(unsubAnomaly)

    const unsubDecision = sse.on('decision', (data) => {
      setPendingActions(prev => [data as Record<string, unknown>, ...prev].slice(0, 20))
    })
    unsubs.push(unsubDecision)

    unsubscribers.current = unsubs

    sse.connect()
    return () => {
      unsubs.forEach(fn => { try { fn() } catch { /* swallow */ } })
    }
  }, [sse])

  // Refresh pending actions periodically
  useEffect(() => {
    if (!sessionId || activeTab !== 'decisions') return
    const timer = window.setInterval(() => {
      // The SSE 'decision' event handles real-time updates.
      // This timer exists as a fallback for decisions that arrive before the tab opens.
    }, 5000)
    return () => window.clearInterval(timer)
  }, [sessionId, activeTab])

  const connected = connectionStatus === 'connected'
  const statusText = connected ? '' : connectionStatus === 'connecting' ? t('panel.connecting') : t('panel.disconnected')

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
      {statusText && (
        <div style={{
          padding: '4px 12px', fontSize: 11,
          background: connected ? 'transparent' : 'var(--warning-bg, #332200)',
          color: connected ? 'var(--text)' : 'var(--warning-text, #ffa726)',
          textAlign: 'center',
        }}>
          {statusText}
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
            onApprove={onApprove}
            onReject={onReject}
          />
        )}
        {activeTab === 'arch' && (
          <ArchTab
            serverUrl={serverUrl}
            modelEntry={modelEntry}
            projectDir={projectDir}
            t={t}
          />
        )}
      </div>
    </div>
  )
}
