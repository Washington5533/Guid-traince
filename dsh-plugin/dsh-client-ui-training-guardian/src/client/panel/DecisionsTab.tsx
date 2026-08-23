import { useState, useEffect, useCallback } from 'react'
import type { TgKey } from '../locales'

interface DecisionsTabProps {
  pending: Array<Record<string, unknown>>
  t: (key: TgKey) => string
  onApprove: (actionId: string) => void
  onReject: (actionId: string, reason: string) => void
}

export function DecisionsTab({ pending, t, onApprove, onReject }: DecisionsTabProps) {
  const [rejectingId, setRejectingId] = useState<string | null>(null)
  const [rejectReason, setRejectReason] = useState('')

  useEffect(() => {
    setRejectingId(null)
    setRejectReason('')
  }, [pending])

  const handleReject = useCallback((actionId: string) => {
    if (!rejectReason.trim()) return
    onReject(actionId, rejectReason)
    setRejectingId(null)
    setRejectReason('')
  }, [onReject, rejectReason])

  if (pending.length === 0) {
    return (
      <div style={{
        textAlign: 'center', padding: '24px 0', color: 'var(--text-secondary, #888)', fontSize: 13,
      } as Record<string, unknown>}>
        {t('decisions.none')}
      </div>
    )
  }

  return (
    <div style={{ padding: 0 }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ borderBottom: '1px solid var(--border, #333)' }}>
            <Th>{t('decisions.tool')}</Th>
            <Th>{t('decisions.action')}</Th>
            <Th>{t('decisions.source')}</Th>
            <Th style={{ textAlign: 'right' }}>{t('decisions.time')}</Th>
            <Th style={{ textAlign: 'center', width: 180 }}>{''}</Th>
          </tr>
        </thead>
        <tbody>
          {pending.map((d, i) => {
            const id = String(d.id ?? d.action_id ?? `d-${i}`)
            const tool = String(d.tool ?? d.source ?? '—')
            const action = typeof d.action === 'string' ? d.action : (typeof d.action === 'object' ? JSON.stringify(d.action) : '—')
            const source = String(d.agent_id ?? d.phase ?? d.phase_inferred ?? '—')
            const time = String(d.timestamp ?? d.time ?? '')
            const shortTime = time.length > 19 ? time.slice(11, 19) : time
            const isRejecting = rejectingId === id

            return (
              <tr key={i} style={{ borderBottom: '1px solid var(--border, #333)' }}>
                <td><code style={{ color: 'var(--accent, #4fc3f7)' }}>{tool}</code></td>
                <td style={{ maxWidth: 260, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{action}</td>
                <td>{source}</td>
                <td style={{ textAlign: 'right', color: 'var(--text-secondary, #888)', whiteSpace: 'nowrap' }}>{shortTime}</td>
                <td style={{ textAlign: 'center' }}>
                  {!isRejecting ? (
                    <div style={{ display: 'flex', gap: 6, justifyContent: 'center' }}>
                      <button
                        onClick={() => onApprove(id)}
                        style={{
                          padding: '3px 10px', fontSize: 11, cursor: 'pointer',
                          background: 'var(--accent, #007acc)', color: '#fff',
                          border: 'none', borderRadius: 3,
                        }}
                      >
                        {t('decisions.approve')}
                      </button>
                      <button
                        onClick={() => setRejectingId(id)}
                        style={{
                          padding: '3px 10px', fontSize: 11, cursor: 'pointer',
                          background: 'transparent', color: 'var(--warning-text, #ffa726)',
                          border: '1px solid var(--warning-text, #ffa726)', borderRadius: 3,
                        }}
                      >
                        {t('decisions.reject')}
                      </button>
                    </div>
                  ) : (
                    <div style={{ display: 'flex', gap: 4, justifyContent: 'center', alignItems: 'center' }}>
                      <input
                        value={rejectReason}
                        onChange={e => setRejectReason(e.target.value)}
                        placeholder={t('misc.warning')}
                        style={{
                          width: 100, padding: '3px 6px', fontSize: 11,
                          border: '1px solid var(--border, #ddd)', borderRadius: 3,
                          background: 'var(--bg, #fff)', color: 'var(--text, #333)',
                        }}
                      />
                      <button
                        onClick={() => handleReject(id)}
                        style={{
                          padding: '3px 8px', fontSize: 11, cursor: 'pointer',
                          background: 'var(--warning-text, #ffa726)', color: '#000',
                          border: 'none', borderRadius: 3,
                        }}
                      >
                        OK
                      </button>
                    </div>
                  )}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function Th({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return <th style={{ padding: '6px 8px', fontWeight: 500, textAlign: 'left', color: 'var(--text-secondary, #888)', fontSize: 11, ...style }}>{children}</th>
}
