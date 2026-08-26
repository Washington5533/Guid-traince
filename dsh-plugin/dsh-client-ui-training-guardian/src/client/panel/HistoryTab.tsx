import { useState, useCallback, useEffect } from 'react'
import type { TgKey } from '../locales'
import { MetricsTab } from './MetricsTab'
import { AnomaliesTab } from './AnomaliesTab'
import { DecisionsTab } from './DecisionsTab'
import { MetricsChart } from './MetricsChart'
import {
  listLocalSessions,
  loadMetricsHistory,
  removeLocalSession,
} from '../storage'

interface HistoryTabProps {
  serverUrl: string
  authToken?: string
  t: (key: TgKey) => string
  currentSessionKey?: string
}

interface SessionItem {
  session_id: string
  event_count: number
  first_ts: number
  last_ts: number
  duration_seconds: number
  type_counts: Record<string, number>
  last_metrics: Record<string, unknown>
  /** Whether this session comes from localStorage rather than the server. */
  isLocal?: boolean
}

type ReplayTab = 'overview' | 'anomalies' | 'decisions' | 'crashes' | 'ai'
type ViewMode = 'list' | 'replay' | 'compare'

function fmtDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`
  const h = Math.floor(seconds / 3600)
  const m = Math.round((seconds % 3600) / 60)
  return `${h}h ${m}m`
}

function fmtTs(ts: number): string {
  if (!ts) return '—'
  return new Date(ts * 1000).toLocaleString()
}

/**
 * History tab: lists past training sessions from the server AND from
 * localStorage. When the server is offline, shows a friendly banner and
 * falls back to locally cached sessions so the user can still review
 * training curves and metrics.
 */
export function HistoryTab({ serverUrl, authToken, t, currentSessionKey }: HistoryTabProps) {
  const [sessions, setSessions] = useState<SessionItem[]>([])
  const [localSessions, setLocalSessions] = useState<SessionItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [offline, setOffline] = useState(false)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [viewMode, setViewMode] = useState<ViewMode>('list')

  // ── Replay state ──
  const [replayId, setReplayId] = useState<string | null>(null)
  const [replayTab, setReplayTab] = useState<ReplayTab>('overview')
  const [replayMetricsList, setReplayMetricsList] = useState<Array<Record<string, unknown>>>([])
  const [replayMetrics, setReplayMetrics] = useState<Record<string, unknown>>({})
  const [replayAnomalies, setReplayAnomalies] = useState<Array<Record<string, unknown>>>([])
  const [replayDecisions, setReplayDecisions] = useState<Array<Record<string, unknown>>>([])
  const [replayCrashes, setReplayCrashes] = useState<Array<Record<string, unknown>>>([])
  const [replayLoading, setReplayLoading] = useState(false)
  const [replayIsLocal, setReplayIsLocal] = useState(false)

  // ── AI analysis state ──
  const [aiText, setAiText] = useState<string | null>(null)
  const [aiLoading, setAiLoading] = useState(false)
  const [aiSource, setAiSource] = useState<string>('')
  const [chatMessages, setChatMessages] = useState<Array<{ role: 'user' | 'ai'; text: string }>>([])
  const [chatInput, setChatInput] = useState('')
  const [chatLoading, setChatLoading] = useState(false)

  // ── Compare state ──
  const [compareData, setCompareData] = useState<Array<Record<string, unknown>> | null>(null)
  const [compareLoading, setCompareLoading] = useState(false)

  const base = serverUrl.replace(/\/$/, '')
  const headers = useCallback((): Record<string, string> => {
    const h: Record<string, string> = { 'Content-Type': 'application/json' }
    if (authToken) h['X-Auth-Token'] = authToken
    return h
  }, [authToken])

  // Load session list on mount — try server first, fall back to local.
  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)

    // Always load local sessions.
    const locals = loadLocalSessions()
    if (!cancelled) setLocalSessions(locals)

    // Try server.
    fetch(`${base}/api/history/sessions`, { headers: headers() })
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then(data => {
        if (!cancelled) {
          setSessions(data.sessions || [])
          setOffline(false)
        }
      })
      .catch(e => {
        if (!cancelled) {
          setOffline(true)
          setError(String(e))
          setSessions([])
        }
      })
      .finally(() => { if (!cancelled) setLoading(false) })

    return () => { cancelled = true }
  }, [base, headers])

  /** Convert localStorage registry to SessionItem[] for display. */
  function loadLocalSessions(): SessionItem[] {
    return listLocalSessions().map(s => ({
      session_id: s.id,
      event_count: s.pointCount,
      first_ts: 0,
      last_ts: new Date(s.updatedAt).getTime() / 1000,
      duration_seconds: 0,
      type_counts: { metrics: s.pointCount },
      last_metrics: s.lastMetrics,
      isLocal: true,
    }))
  }

  /** Merge remote + local sessions, dedup by session_id. */
  const allSessions: SessionItem[] = (() => {
    const map = new Map<string, SessionItem>()
    for (const s of sessions) map.set(s.session_id, s)
    for (const s of localSessions) {
      if (!map.has(s.session_id)) map.set(s.session_id, s)
    }
    return [...map.values()]
  })()

  // Load replay data for selected session (remote or local).
  const loadReplay = useCallback(async (sessionId: string, isLocal = false) => {
    setReplayId(sessionId)
    setViewMode('replay')
    setReplayTab('overview')
    setReplayLoading(true)
    setReplayIsLocal(isLocal)
    setAiText(null)
    setChatMessages([])

    if (isLocal) {
      // Load from localStorage.
      const metricsList = loadMetricsHistory(sessionId)
      setReplayMetricsList(metricsList)
      setReplayMetrics(metricsList.length > 0 ? metricsList[metricsList.length - 1] : {})
      setReplayAnomalies([])
      setReplayDecisions([])
      setReplayCrashes([])
      setReplayLoading(false)
      return
    }

    try {
      const [mRes, aRes, dRes, cRes] = await Promise.all([
        fetch(`${base}/api/history/${encodeURIComponent(sessionId)}/metrics`, { headers: headers() }),
        fetch(`${base}/api/history/${encodeURIComponent(sessionId)}/anomalies`, { headers: headers() }),
        fetch(`${base}/api/history/${encodeURIComponent(sessionId)}/decisions`, { headers: headers() }),
        fetch(`${base}/api/history/${encodeURIComponent(sessionId)}/crashes`, { headers: headers() }),
      ])
      const [mData, aData, dData, cData] = await Promise.all([mRes.json(), aRes.json(), dRes.json(), cRes.json()])
      const metricsList = mData.metrics || []
      setReplayMetricsList(metricsList)
      setReplayMetrics(metricsList.length > 0 ? metricsList[metricsList.length - 1] : {})
      setReplayAnomalies((aData.anomalies || []).slice(-50))
      setReplayDecisions((dData.decisions || []).slice(-50))
      setReplayCrashes(cData.crashes || [])
    } catch (e) {
      setError(String(e))
    } finally {
      setReplayLoading(false)
    }
  }, [base, headers])

  // AI Analysis
  const runAiAnalyze = useCallback(async () => {
    if (!replayId || replayIsLocal) return
    setAiLoading(true)
    setAiText(null)
    try {
      const res = await fetch(`${base}/api/history/${encodeURIComponent(replayId)}/ai/analyze`, {
        method: 'POST', headers: headers(),
      })
      const data = await res.json()
      setAiText(data.analysis || t('history.aiFailed'))
      setAiSource(data.source || '')
    } catch {
      setAiText(t('history.aiFailed'))
    } finally {
      setAiLoading(false)
    }
  }, [replayId, replayIsLocal, base, headers, t])

  // AI Chat
  const sendChat = useCallback(async () => {
    if (!replayId || !chatInput.trim() || replayIsLocal) return
    const question = chatInput.trim()
    setChatInput('')
    setChatMessages(prev => [...prev, { role: 'user', text: question }])
    setChatLoading(true)
    try {
      const res = await fetch(`${base}/api/history/${encodeURIComponent(replayId)}/ai/chat`, {
        method: 'POST', headers: headers(), body: JSON.stringify({ question }),
      })
      const data = await res.json()
      setChatMessages(prev => [...prev, { role: 'ai', text: data.answer || '?' }])
    } catch {
      setChatMessages(prev => [...prev, { role: 'ai', text: t('history.aiFailed') }])
    } finally {
      setChatLoading(false)
    }
  }, [replayId, replayIsLocal, chatInput, base, headers, t])

  // Multi-session compare
  const runCompare = useCallback(async () => {
    if (selectedIds.size < 2) return
    setViewMode('compare')
    setCompareLoading(true)
    setCompareData(null)
    try {
      const res = await fetch(`${base}/api/history/compare`, {
        method: 'POST', headers: headers(),
        body: JSON.stringify({ session_ids: [...selectedIds] }),
      })
      const data = await res.json()
      setCompareData(data.sessions || [])
    } catch (e) {
      setError(String(e))
    } finally {
      setCompareLoading(false)
    }
  }, [selectedIds, base, headers])

  const toggleSelect = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id); else next.add(id)
      return next
    })
  }

  const goBack = () => {
    setViewMode('list')
    setReplayId(null)
    setCompareData(null)
    setSelectedIds(new Set())
  }

  const handleClearLocal = (id: string) => {
    removeLocalSession(id)
    setLocalSessions(loadLocalSessions())
  }

  // ──────────────────── COMPARE VIEW ────────────────────
  if (viewMode === 'compare') {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '6px 12px', borderBottom: '1px solid var(--border, #333)',
          background: 'var(--tab-bg, #16162a)',
        }}>
          <button onClick={goBack} style={BTN_BACK}>← {t('history.back')}</button>
          <span style={{ fontSize: 12, fontWeight: 600 }}>{t('history.compareResult')}</span>
        </div>
        <div style={{ flex: 1, overflow: 'auto', padding: '12px 16px' }}>
          {compareLoading && <div style={{ textAlign: 'center', padding: 24, color: '#888' }}>{t('history.loading')}</div>}
          {compareData && (
            <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border, #333)' }}>
                  <th style={TH}>Session</th>
                  <th style={TH}>{t('history.duration')}</th>
                  <th style={TH}>{t('history.metrics')}</th>
                  <th style={TH}>{t('history.anomalies')}</th>
                  <th style={TH}>Loss</th>
                  <th style={TH}>Min Loss</th>
                  <th style={TH}>Acc</th>
                  <th style={TH}>Best Acc</th>
                </tr>
              </thead>
              <tbody>
                {compareData.map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid var(--border, #222)' }}>
                    <td style={TD}>{String(s.session_id)}</td>
                    <td style={TD}>{s.duration_seconds != null ? fmtDuration(Number(s.duration_seconds)) : '—'}</td>
                    <td style={TD}>{String(s.metrics_count ?? '—')}</td>
                    <td style={TD}>{String(s.anomaly_count ?? '—')}</td>
                    <td style={TD}>{s.loss_final != null ? Number(s.loss_final).toFixed(4) : '—'}</td>
                    <td style={TD}>{s.loss_min != null ? Number(s.loss_min).toFixed(4) : '—'}</td>
                    <td style={TD}>{s.acc_final != null ? Number(s.acc_final).toFixed(4) : '—'}</td>
                    <td style={TD}>{s.acc_best != null ? Number(s.acc_best).toFixed(4) : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    )
  }

  // ──────────────────── REPLAY VIEW ────────────────────
  if (viewMode === 'replay' && replayId) {
    const REPLAY_TABS: { key: ReplayTab; label: string }[] = [
      { key: 'overview', label: t('tab.overview') },
      { key: 'anomalies', label: t('tab.anomalies') },
      { key: 'decisions', label: t('tab.decisions') },
      { key: 'crashes', label: t('history.crashData') },
      ...(!replayIsLocal ? [{ key: 'ai' as ReplayTab, label: t('history.aiAnalyze') }] : []),
    ]
    return (
      <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap',
          padding: '6px 12px', borderBottom: '1px solid var(--border, #333)',
          background: 'var(--tab-bg, #16162a)',
        }}>
          <button onClick={goBack} style={BTN_BACK}>← {t('history.back')}</button>
          <span style={{ fontSize: 11, color: 'var(--text-secondary, #666)' }}>{replayId}</span>
          {replayIsLocal && (
            <span style={{
              fontSize: 9, padding: '1px 6px', borderRadius: 3,
              background: 'var(--info-bg, #0a2540)', color: 'var(--info-text, #4fc3f7)',
            }}>{t('history.localBadge')}</span>
          )}
          <div style={{ flex: 1 }} />
          {REPLAY_TABS.map(tab => (
            <button key={tab.key} onClick={() => setReplayTab(tab.key)}
              style={{
                padding: '3px 10px', fontSize: 11, cursor: 'pointer', borderRadius: 4,
                border: '1px solid var(--border, #333)',
                background: replayTab === tab.key ? 'var(--accent, #007acc)' : 'transparent',
                color: replayTab === tab.key ? '#fff' : 'var(--text-secondary, #888)',
              }}>
              {tab.label}
            </button>
          ))}
        </div>

        {replayLoading && (
          <div style={{ textAlign: 'center', padding: 24, color: 'var(--text-secondary, #888)' }}>
            {t('history.loading')}
          </div>
        )}

        {!replayLoading && (
          <div style={{ flex: 1, overflow: 'auto' }}>
            {replayTab === 'overview' && (
              <div style={{ padding: '8px 12px' }}>
                {/* Trend chart */}
                {replayMetricsList.length > 1 && (
                  <div style={{ marginBottom: 12 }}>
                    <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 4, color: 'var(--text-secondary, #aaa)' }}>
                      {t('history.trend')}
                    </div>
                    <MetricsChart data={replayMetricsList} />
                  </div>
                )}
                <MetricsTab metrics={replayMetrics} t={t} />
              </div>
            )}
            {replayTab === 'anomalies' && (
              replayIsLocal ? (
                <div style={{ textAlign: 'center', padding: 24, color: 'var(--text-secondary, #666)', fontSize: 12 }}>
                  {t('history.localBadge')} — {t('anomalies.none')}
                </div>
              ) : <AnomaliesTab anomalies={replayAnomalies} t={t} />
            )}
            {replayTab === 'decisions' && (
              replayIsLocal ? (
                <div style={{ textAlign: 'center', padding: 24, color: 'var(--text-secondary, #666)', fontSize: 12 }}>
                  {t('history.localBadge')} — {t('decisions.none')}
                </div>
              ) : (
                <DecisionsTab pending={replayDecisions} t={t}
                  onApprove={() => {}} onReject={() => {}} />
              )
            )}
            {replayTab === 'crashes' && (
              replayIsLocal ? (
                <div style={{ textAlign: 'center', padding: 24, color: 'var(--text-secondary, #666)', fontSize: 12 }}>
                  {t('history.localBadge')} — {t('history.noCrashes')}
                </div>
              ) : (
                <div style={{ padding: '12px 16px' }}>
                  {replayCrashes.length === 0 ? (
                    <div style={{ textAlign: 'center', padding: 24, color: 'var(--text-secondary, #888)', fontSize: 13 }}>
                      {t('history.noCrashes')}
                    </div>
                  ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {replayCrashes.map((c, i) => (
                        <div key={i} style={{
                          padding: '8px 12px', borderRadius: 4,
                          background: 'var(--card-bg, #222240)', border: '1px solid #ef5350',
                          fontSize: 12,
                        }}>
                          <div style={{ fontWeight: 600, color: '#ef5350', marginBottom: 4 }}>
                            {String(c.type || 'crash')} — {fmtTs(Number(c.timestamp || 0))}
                          </div>
                          <div style={{ color: 'var(--text-secondary, #aaa)' }}>
                            {String(c.detail || c.error || JSON.stringify(c))}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )
            )}
            {replayTab === 'ai' && !replayIsLocal && (
              <div style={{ padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: 12 }}>
                {/* AI Analysis */}
                <div>
                  <button onClick={runAiAnalyze} disabled={aiLoading}
                    style={{
                      padding: '6px 16px', fontSize: 12, cursor: aiLoading ? 'wait' : 'pointer',
                      borderRadius: 4, border: '1px solid var(--accent, #007acc)',
                      background: aiLoading ? 'transparent' : 'var(--accent, #007acc)',
                      color: aiLoading ? 'var(--text-secondary, #888)' : '#fff',
                    }}>
                    {aiLoading ? t('history.aiAnalyzing') : t('history.aiAnalyze')}
                  </button>
                  {aiText && (
                    <div style={{
                      marginTop: 8, padding: 12, borderRadius: 6, fontSize: 12, lineHeight: 1.6,
                      background: 'var(--card-bg, #222240)', border: '1px solid var(--border, #333)',
                      color: 'var(--text, #eee)', whiteSpace: 'pre-wrap',
                    }}>
                      {aiSource === 'agent' && (
                        <div style={{ fontSize: 10, color: 'var(--accent, #4fc3f7)', marginBottom: 4 }}>
                          🤖 {t('history.aiResult')}
                        </div>
                      )}
                      {aiText}
                    </div>
                  )}
                </div>

                {/* AI Chat */}
                <div style={{ borderTop: '1px solid var(--border, #333)', paddingTop: 12 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>{t('history.aiChat')}</div>
                  {chatMessages.map((msg, i) => (
                    <div key={i} style={{
                      marginBottom: 6, padding: '6px 10px', borderRadius: 6, fontSize: 12,
                      background: msg.role === 'user' ? 'var(--accent, #007acc)' : 'var(--card-bg, #222240)',
                      color: msg.role === 'user' ? '#fff' : 'var(--text, #eee)',
                      maxWidth: '85%', marginLeft: msg.role === 'user' ? 'auto' : 0,
                      whiteSpace: 'pre-wrap',
                    }}>
                      {msg.text}
                    </div>
                  ))}
                  {chatLoading && (
                    <div style={{ fontSize: 11, color: 'var(--text-secondary, #888)', padding: '4px 0' }}>
                      {t('history.aiAnalyzing')}
                    </div>
                  )}
                  <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
                    <input
                      value={chatInput}
                      onChange={e => setChatInput(e.target.value)}
                      onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat() } }}
                      placeholder={t('history.aiChatPlaceholder')}
                      style={{
                        flex: 1, padding: '6px 8px', fontSize: 12, borderRadius: 4,
                        border: '1px solid var(--border, #333)',
                        background: 'var(--bg, #1a1a2e)', color: 'var(--text, #eee)',
                      }}
                    />
                    <button onClick={sendChat} disabled={chatLoading || !chatInput.trim()}
                      style={{
                        padding: '6px 12px', fontSize: 12, cursor: 'pointer', borderRadius: 4,
                        border: '1px solid var(--accent, #007acc)',
                        background: 'var(--accent, #007acc)', color: '#fff',
                        opacity: chatLoading || !chatInput.trim() ? 0.5 : 1,
                      }}>
                      {t('history.aiChatSend')}
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    )
  }

  // ──────────────────── LIST VIEW ────────────────────
  return (
    <div style={{ padding: '12px 16px' }}>
      {/* Offline banner */}
      {offline && (
        <div style={{
          padding: '10px 14px', marginBottom: 12, borderRadius: 6,
          background: 'var(--info-bg, #0a2540)', border: '1px solid var(--info-text, #1a4a6a)',
          textAlign: 'center',
        }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--info-text, #4fc3f7)', marginBottom: 2 }}>
            {t('history.offlineBanner')}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-secondary, #6aa)', opacity: 0.8 }}>
            {t('history.offlineHint')}
          </div>
        </div>
      )}

      {loading && (
        <div style={{ textAlign: 'center', padding: 24, color: 'var(--text-secondary, #888)' }}>
          {t('history.loading')}
        </div>
      )}
      {error && !offline && (
        <div style={{ padding: 12, color: '#cf222e', fontSize: 12, textAlign: 'center' }}>{error}</div>
      )}
      {!loading && allSessions.length === 0 && (
        <div style={{ textAlign: 'center', padding: 24, color: 'var(--text-secondary, #888)', fontSize: 13 }}>
          {offline ? t('history.localNoData') : t('history.noData')}
        </div>
      )}
      {!loading && allSessions.length > 0 && (
        <>
          {/* Compare button */}
          {selectedIds.size >= 2 && !offline && (
            <div style={{ marginBottom: 8, display: 'flex', justifyContent: 'flex-end' }}>
              <button onClick={runCompare} style={{
                padding: '4px 14px', fontSize: 11, cursor: 'pointer', borderRadius: 4,
                border: '1px solid var(--accent, #007acc)',
                background: 'var(--accent, #007acc)', color: '#fff',
              }}>
                {t('history.compareRun')} ({selectedIds.size})
              </button>
            </div>
          )}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {allSessions.map((s) => (
              <div key={s.session_id} style={{
                padding: '10px 14px', borderRadius: 6,
                background: 'var(--card-bg, #222240)', border: '1px solid var(--border, #333)',
                transition: 'border-color .15s',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                  {!s.isLocal && !offline && (
                    <input type="checkbox" checked={selectedIds.has(s.session_id)}
                      onChange={() => toggleSelect(s.session_id)}
                      style={{ cursor: 'pointer' }} />
                  )}
                  <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--accent, #4fc3f7)', flex: 1, cursor: 'pointer' }}
                    onClick={() => loadReplay(s.session_id, !!s.isLocal)}>
                    {s.session_id}
                  </span>
                  {s.isLocal && (
                    <span style={{
                      fontSize: 9, padding: '1px 6px', borderRadius: 3,
                      background: 'var(--info-bg, #0a2540)', color: 'var(--info-text, #4fc3f7)',
                    }}>{t('history.localBadge')}</span>
                  )}
                  <span style={{ fontSize: 10, color: 'var(--text-secondary, #666)' }}>
                    {s.isLocal
                      ? new Date(s.last_ts * 1000).toLocaleString()
                      : fmtTs(s.last_ts)}
                  </span>
                  {s.isLocal ? (
                    <button onClick={() => handleClearLocal(s.session_id)} style={{
                      padding: '2px 10px', fontSize: 10, cursor: 'pointer', borderRadius: 4,
                      border: '1px solid var(--border, #333)', background: 'transparent',
                      color: 'var(--text-secondary, #888)',
                    }}>
                      {t('history.localClear')}
                    </button>
                  ) : (
                    <button onClick={() => loadReplay(s.session_id, false)} style={{
                      padding: '2px 10px', fontSize: 10, cursor: 'pointer', borderRadius: 4,
                      border: '1px solid var(--border, #333)', background: 'transparent',
                      color: 'var(--accent, #4fc3f7)',
                    }}>
                      {t('history.loadBtn')}
                    </button>
                  )}
                </div>
                <div style={{ display: 'flex', gap: 12, fontSize: 11, color: 'var(--text-secondary, #888)', flexWrap: 'wrap' }}>
                  <span>{s.event_count} {s.isLocal ? t('history.dataPoints') : t('history.events')}</span>
                  {!s.isLocal && (
                    <>
                      <span>{t('history.duration')}: {fmtDuration(s.duration_seconds)}</span>
                      {s.type_counts.metrics != null && <span>{t('history.metrics')}: {s.type_counts.metrics}</span>}
                      {s.type_counts.anomaly != null && (
                        <span style={{ color: s.type_counts.anomaly > 0 ? '#ffa726' : undefined }}>
                          {t('history.anomalies')}: {s.type_counts.anomaly}
                        </span>
                      )}
                      {s.type_counts.decision != null && <span>{t('history.decisions')}: {s.type_counts.decision}</span>}
                      {s.type_counts.crash != null && s.type_counts.crash > 0 && (
                        <span style={{ color: '#ef5350' }}>{t('history.crashes')}: {s.type_counts.crash}</span>
                      )}
                    </>
                  )}
                </div>
                {Object.keys(s.last_metrics).length > 0 && (
                  <div style={{
                    display: 'flex', gap: 10, marginTop: 6, fontSize: 11,
                    color: 'var(--text-secondary, #aaa)', fontFamily: 'ui-monospace, monospace',
                  }}>
                    {s.last_metrics.loss != null && <span>loss: {Number(s.last_metrics.loss).toFixed(4)}</span>}
                    {(s.last_metrics.accuracy ?? s.last_metrics.val_acc) != null && (
                      <span>acc: {Number(s.last_metrics.accuracy ?? s.last_metrics.val_acc).toFixed(4)}</span>
                    )}
                    {s.last_metrics.epoch != null && <span>ep: {String(s.last_metrics.epoch)}</span>}
                    {s.last_metrics.step != null && <span>step: {String(s.last_metrics.step)}</span>}
                  </div>
                )}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

// Shared styles
const BTN_BACK: React.CSSProperties = {
  padding: '3px 10px', fontSize: 11, cursor: 'pointer', borderRadius: 4,
  border: '1px solid var(--border, #333)', background: 'transparent',
  color: 'var(--text-secondary, #888)',
}
const TH: React.CSSProperties = {
  textAlign: 'left', padding: '6px 8px', fontSize: 10, fontWeight: 600,
  color: 'var(--text-secondary, #aaa)', borderBottom: '1px solid var(--border, #333)',
}
const TD: React.CSSProperties = {
  padding: '6px 8px', fontSize: 11, color: 'var(--text, #eee)',
  borderBottom: '1px solid var(--border, #222)',
}
