import type { TgKey } from '../locales'

interface GpuTabProps {
  gpuStatus: Record<string, unknown> | null
  t: (key: TgKey) => string
}

export function GpuTab({ gpuStatus, t }: GpuTabProps) {
  if (!gpuStatus) {
    return (
      <div style={{
        textAlign: 'center', padding: '24px 0', color: 'var(--text-secondary, #888)', fontSize: 13,
      }}>
        {t('gpu.noData')}
      </div>
    )
  }

  const devices = Array.isArray(gpuStatus.devices) ? gpuStatus.devices : []
  const gpus = devices.length > 0 ? devices : [gpuStatus]

  return (
    <div style={{ padding: '12px 16px' }}>
      <div style={{ fontWeight: 500, marginBottom: 10, fontSize: 13, color: 'var(--text)' }}>
        {t('gpu.title')}
      </div>
      <div style={{
        display: 'flex', flexDirection: 'column', gap: 10,
      }}>
        {gpus.map((gpu: Record<string, unknown>, idx: number) => (
          <GpuCard key={idx} gpu={gpu} idx={idx} t={t} />
        ))}
      </div>
    </div>
  )
}

function GpuCard({ gpu, idx, t }: { gpu: Record<string, unknown>; idx: number; t: (key: TgKey) => string }) {
  const num = (v: unknown): number | undefined => (typeof v === 'number' && isFinite(v) ? v : undefined)
  const utilization = num(gpu.utilization) ?? (typeof gpu.utilization === 'object' && gpu.utilization !== null ? num((gpu.utilization as Record<string, unknown>).gpu) : undefined)
  const temperature = num(gpu.temperature)
  const memoryUsed = num(gpu.memory_used) ?? (typeof gpu.memory === 'object' && gpu.memory !== null ? num((gpu.memory as Record<string, unknown>).used) : undefined)
  const memoryTotal = num(gpu.memory_total) ?? (typeof gpu.memory === 'object' && gpu.memory !== null ? num((gpu.memory as Record<string, unknown>).total) : undefined)
  const power = num(gpu.power) ?? (typeof gpu.power === 'object' && gpu.power !== null ? num((gpu.power as Record<string, unknown>).current) : undefined)
  const name = String(gpu.name ?? gpu.device_name ?? `GPU ${idx}`)

  const memoryPct = memoryUsed !== undefined && memoryTotal !== undefined && memoryTotal > 0
    ? Math.round((memoryUsed / memoryTotal) * 100)
    : undefined

  return (
    <div style={{
      background: 'var(--card-bg, #222240)', borderRadius: 6,
      border: '1px solid var(--border, #333)', padding: '10px 12px',
    }}>
      <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 13 }}>{name}</div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px 16px' }}>
        <GpuMetric label={t('gpu.utilization')} value={`${utilization ?? '—'}${utilization !== undefined ? '%' : ''}`} pct={utilization} />
        <GpuMetric label={t('gpu.temperature')} value={temperature !== undefined ? `${temperature}°C` : '—'} pct={temperature ? Math.min(temperature / 100, 1) * 100 : undefined} warn={temperature !== undefined && temperature > 85} />
        <GpuMetric label={t('gpu.memory')} value={memoryPct !== undefined ? `${memoryPct}%` : '—'} pct={memoryPct} />
        <GpuMetric label={t('gpu.power')} value={power !== undefined ? `${power}W` : '—'} pct={power !== undefined ? (power / 500) * 100 : undefined} />
      </div>
    </div>
  )
}

function GpuMetric({ label, value, pct, warn }: { label: string; value: string; pct?: number | undefined; warn?: boolean }) {
  return (
    <div>
      <div style={{ fontSize: 11, color: 'var(--text-secondary, #888)', marginBottom: 2 }}>{label}</div>
      <div style={{ fontSize: 13, fontWeight: 500, color: warn ? 'var(--warning-text, #ffa726)' : 'var(--accent, #4fc3f7)' }}>
        {value}
      </div>
      {pct !== undefined && (
        <div style={{
          height: 3, borderRadius: 2, marginTop: 3,
          background: 'var(--border, #333)', overflow: 'hidden',
        }}>
          <div style={{
            height: '100%', width: `${Math.min(Math.max(pct, 0), 100)}%`,
            background: warn ? 'var(--warning-text, #ffa726)' : 'var(--accent, #4fc3f7)',
            borderRadius: 2, transition: 'width 0.3s',
          }} />
        </div>
      )}
    </div>
  )
}
