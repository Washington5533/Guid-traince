import type { TgKey } from '../locales'

interface AnomaliesTabProps {
  anomalies: Array<Record<string, unknown>>
  t: (key: TgKey) => string
}

export function AnomaliesTab({ anomalies, t }: AnomaliesTabProps) {
  if (anomalies.length === 0) {
    return (
      <div style={{
        textAlign: 'center', padding: '24px 0', color: 'var(--text-secondary, #888)', fontSize: 13,
      }}>
        {t('anomalies.none')}
      </div>
    )
  }

  return (
    <div style={{ padding: 0 }}>
      <table style={{
        width: '100%', borderCollapse: 'collapse', fontSize: 12,
      }}>
        <thead>
          <tr style={{ borderBottom: '1px solid var(--border, #333)' }}>
            <Th>{t('anomalies.type')}</Th>
            <Th>{t('anomalies.severity')}</Th>
            <Th style={{ flex: 1, textAlign: 'left' }}>{t('anomalies.description')}</Th>
            <Th style={{ textAlign: 'right' }}>{t('anomalies.time')}</Th>
          </tr>
        </thead>
        <tbody>
          {anomalies.map((a, i) => {
            const severity = String(a.severity ?? 'info')
            const desc = String(a.message ?? a.description ?? '—')
            const time = String(a.timestamp ?? a.time ?? '')
            const shortTime = time.length > 19 ? time.slice(11, 19) : time
            return (
              <tr key={i} style={{ borderBottom: '1px solid var(--border, #333)' }}>
                <td><code style={{ color: 'var(--accent, #4fc3f7)' }}>{String(a.type ?? a.event_type ?? '?')}</code></td>
                <td><SeverityBadge severity={severity} /></td>
                <td style={{ color: 'var(--text-secondary, #ccc)', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{desc}</td>
                <td style={{ textAlign: 'right', color: 'var(--text-secondary, #888)', whiteSpace: 'nowrap' }}>{shortTime}</td>
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

function SeverityBadge({ severity }: { severity: string }) {
  const colorMap: Record<string, string> = {
    low: '#66bb6a',
    medium: '#ffa726',
    high: '#ef5350',
    critical: '#ff1744',
    info: '#4fc3f7',
  }
  const c = colorMap[severity.toLowerCase()] || colorMap.info
  return (
    <span style={{
      display: 'inline-block', padding: '1px 8px', borderRadius: 10, fontSize: 11, fontWeight: 500,
      background: `${c}22`, color: c, border: `1px solid ${c}44`,
    }}>
      {severity}
    </span>
  )
}
