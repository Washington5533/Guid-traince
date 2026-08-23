import type { TgKey } from '../locales'

interface MetricsTabProps {
  metrics: Record<string, unknown>
  t: (key: TgKey) => string
}

function getValue(metrics: Record<string, unknown>, key: string): string {
  const v = metrics[key]
  if (v === null || v === undefined) return '—'
  if (typeof v === 'number') return v.toFixed(v < 10 ? 4 : 2)
  return String(v)
}

export function MetricsTab({ metrics, t }: MetricsTabProps) {
  const hasData = Object.keys(metrics).length > 0

  return (
    <div style={{ padding: '12px 16px' }}>
      {!hasData && (
        <div style={{
          textAlign: 'center', padding: '24px 0', color: 'var(--text-secondary, #888)',
          fontSize: 13,
        }}>
          {t('overview.noMetrics')}
        </div>
      )}
      {hasData && (
        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))',
          gap: 10,
        }}>
          <MetricCard label={t('overview.epoch')} value={getValue(metrics, 'epoch')} />
          <MetricCard label={t('overview.step')} value={getValue(metrics, 'step')} />
          <MetricCard label={t('overview.loss')} value={getValue(metrics, 'loss')} />
          <MetricCard label={t('overview.accuracy')} value={getValue(metrics, 'accuracy')} />
          <MetricCard label={t('overview.lr')} value={getValue(metrics, 'learning_rate')} />
          <MetricCard label={t('overview.status')} value={getValue(metrics, 'status')} />
        </div>
      )}
    </div>
  )
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div style={{
      background: 'var(--card-bg, #222240)', borderRadius: 6, padding: '10px 12px',
      border: '1px solid var(--border, #333)',
    }}>
      <div style={{ fontSize: 11, color: 'var(--text-secondary, #888)', marginBottom: 2 }}>{label}</div>
      <div style={{ fontSize: 16, fontWeight: 600, color: 'var(--accent, #4fc3f7)', fontFamily: 'ui-monospace, SFMono-Regular, monospace' }}>
        {value}
      </div>
    </div>
  )
}
