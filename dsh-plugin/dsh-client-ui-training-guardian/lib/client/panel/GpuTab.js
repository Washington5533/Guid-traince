import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
export function GpuTab({ gpuStatus, t }) {
    if (!gpuStatus) {
        return (_jsx("div", { style: {
                textAlign: 'center', padding: '24px 0', color: 'var(--text-secondary, #888)', fontSize: 13,
            }, children: t('gpu.noData') }));
    }
    const devices = Array.isArray(gpuStatus.devices) ? gpuStatus.devices : [];
    const gpus = devices.length > 0 ? devices : [gpuStatus];
    return (_jsxs("div", { style: { padding: '12px 16px' }, children: [_jsx("div", { style: { fontWeight: 500, marginBottom: 10, fontSize: 13, color: 'var(--text)' }, children: t('gpu.title') }), _jsx("div", { style: {
                    display: 'flex', flexDirection: 'column', gap: 10,
                }, children: gpus.map((gpu, idx) => (_jsx(GpuCard, { gpu: gpu, idx: idx, t: t }, idx))) })] }));
}
function GpuCard({ gpu, idx, t }) {
    const utilization = typeof gpu.utilization === 'number' ? gpu.utilization : (typeof gpu.utilization === 'object' ? gpu.utilization.gpu : undefined);
    const temperature = typeof gpu.temperature === 'number' ? gpu.temperature : undefined;
    const memoryUsed = typeof gpu.memory_used === 'number' ? gpu.memory_used : (typeof gpu.memory === 'object' ? gpu.memory.used : undefined);
    const memoryTotal = typeof gpu.memory_total === 'number' ? gpu.memory_total : (typeof gpu.memory === 'object' ? gpu.memory.total : undefined);
    const power = typeof gpu.power === 'number' ? gpu.power : (typeof gpu.power === 'object' ? gpu.power.current : undefined);
    const name = String(gpu.name ?? gpu.device_name ?? `GPU ${idx}`);
    const memoryPct = memoryUsed !== undefined && memoryTotal && memoryTotal > 0
        ? Math.round((memoryUsed / memoryTotal) * 100)
        : undefined;
    return (_jsxs("div", { style: {
            background: 'var(--card-bg, #222240)', borderRadius: 6,
            border: '1px solid var(--border, #333)', padding: '10px 12px',
        }, children: [_jsx("div", { style: { fontWeight: 600, marginBottom: 8, fontSize: 13 }, children: name }), _jsxs("div", { style: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px 16px' }, children: [_jsx(GpuMetric, { label: t('gpu.utilization'), value: `${utilization ?? '—'}${utilization !== undefined ? '%' : ''}`, pct: utilization }), _jsx(GpuMetric, { label: t('gpu.temperature'), value: temperature !== undefined ? `${temperature}°C` : '—', pct: temperature ? Math.min(temperature / 100, 1) * 100 : undefined, warn: temperature && temperature > 85 }), _jsx(GpuMetric, { label: t('gpu.memory'), value: memoryPct !== undefined ? `${memoryPct}%` : '—', pct: memoryPct }), _jsx(GpuMetric, { label: t('gpu.power'), value: power !== undefined ? `${power}W` : '—', pct: power ? (power / 500) * 100 : undefined })] })] }));
}
function GpuMetric({ label, value, pct, warn }) {
    return (_jsxs("div", { children: [_jsx("div", { style: { fontSize: 11, color: 'var(--text-secondary, #888)', marginBottom: 2 }, children: label }), _jsx("div", { style: { fontSize: 13, fontWeight: 500, color: warn ? 'var(--warning-text, #ffa726)' : 'var(--accent, #4fc3f7)' }, children: value }), pct !== undefined && (_jsx("div", { style: {
                    height: 3, borderRadius: 2, marginTop: 3,
                    background: 'var(--border, #333)', overflow: 'hidden',
                }, children: _jsx("div", { style: {
                        height: '100%', width: `${Math.min(Math.max(pct, 0), 100)}%`,
                        background: warn ? 'var(--warning-text, #ffa726)' : 'var(--accent, #4fc3f7)',
                        borderRadius: 2, transition: 'width 0.3s',
                    } }) }))] }));
}
//# sourceMappingURL=GpuTab.js.map