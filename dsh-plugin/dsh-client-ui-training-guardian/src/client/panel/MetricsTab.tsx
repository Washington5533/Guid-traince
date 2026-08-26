import { useState, useEffect, useRef } from 'react'
import type { TgKey } from '../locales'
import { MetricsChart } from './MetricsChart'

interface MetricsTabProps {
  metrics: Record<string, unknown>
  metricsHistory?: Array<Record<string, unknown>>
  t: (key: TgKey) => string
  onClearHistory?: () => void
}

/**
 * Read a metric value trying multiple backend field names.
 * Backend may emit step/loss/val_acc/lr while the UI uses epoch/loss/accuracy/lr.
 */
function getValue(metrics: Record<string, unknown>, ...keys: string[]): string {
  for (const key of keys) {
    const v = metrics[key]
    if (v !== null && v !== undefined) {
      if (typeof v === 'number') return v.toFixed(v < 10 ? 4 : 2)
      return String(v)
    }
  }
  return '—'
}

/** Detect which field name the backend uses for a given concept. */
function detectField(data: Array<Record<string, unknown>>, candidates: string[]): string | null {
  for (const c of candidates) {
    if (data.some(d => typeof d[c] === 'number')) return c
  }
  return null
}

export function MetricsTab({ metrics, metricsHistory = [], t, onClearHistory }: MetricsTabProps) {
  const hasData = Object.keys(metrics).length > 0

  // Auto-detect chart fields from available history data.
  const leftField = detectField(metricsHistory, ['loss', 'train_loss', 'training_loss']) || 'loss'
  const rightField = detectField(metricsHistory, ['accuracy', 'val_acc', 'acc', 'val_accuracy']) || 'accuracy'
  const xField = detectField(metricsHistory, ['step', 'epoch', 'global_step', 'batch']) || 'step'

  // Responsive width measurement for the chart.
  const containerRef = useRef<HTMLDivElement>(null)
  const [chartWidth, setChartWidth] = useState(460)

  useEffect(() => {
    if (!containerRef.current) return
    const el = containerRef.current
    const measure = () => {
      const w = el.clientWidth
      if (w > 100) setChartWidth(w)
    }
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  return (
    <div style={{ padding: '12px 16px' }}>
      {/* Metric cards row */}
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
          gap: 10, marginBottom: 16,
        }}>
          <MetricCard label={t('overview.epoch')} value={getValue(metrics, 'epoch', 'step')} />
          <MetricCard label={t('overview.step')} value={getValue(metrics, 'step', 'global_step')} />
          <MetricCard label={t('overview.loss')} value={getValue(metrics, 'loss')} />
          <MetricCard label={t('overview.accuracy')} value={getValue(metrics, 'accuracy', 'val_acc', 'acc')} />
          <MetricCard label={t('overview.lr')} value={getValue(metrics, 'lr', 'learning_rate')} />
          <MetricCard label={t('overview.status')} value={getValue(metrics, 'status')} />
        </div>
      )}

      {/* Training curves chart */}
      <div ref={containerRef} style={{
        background: 'var(--card-bg, #222240)', borderRadius: 6,
        border: '1px solid var(--border, #333)', padding: '8px 12px 12px',
      }}>
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          marginBottom: 8,
        }}>
          <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-secondary, #888)', textTransform: 'uppercase', letterSpacing: 0.5 }}>
            {t('overview.chartTitle')}
          </span>
          {metricsHistory.length > 0 && onClearHistory && (
            <button
              onClick={onClearHistory}
              style={{
                padding: '1px 8px', fontSize: 9, cursor: 'pointer',
                background: 'transparent', color: 'var(--text-secondary, #666)',
                border: '1px solid var(--border, #444)', borderRadius: 3,
              }}
            >
              {t('overview.chartClear')}
            </button>
          )}
        </div>
        {metricsHistory.length >= 2 ? (
          <MetricsChart
            data={metricsHistory}
            leftField={leftField}
            rightField={rightField}
            xField={xField}
            width={chartWidth - 24}
            height={200}
          />
        ) : (
          <div style={{
            textAlign: 'center', padding: '24px 0', color: 'var(--text-secondary, #666)',
            fontSize: 11, fontStyle: 'italic',
          }}>
            {t('overview.chartNoData')}
          </div>
        )}
      </div>
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
