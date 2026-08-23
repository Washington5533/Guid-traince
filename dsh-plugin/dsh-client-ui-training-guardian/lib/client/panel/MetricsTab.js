import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
function getValue(metrics, key) {
    const v = metrics[key];
    if (v === null || v === undefined)
        return '—';
    if (typeof v === 'number')
        return v.toFixed(v < 10 ? 4 : 2);
    return String(v);
}
export function MetricsTab({ metrics, t }) {
    const hasData = Object.keys(metrics).length > 0;
    return (_jsxs("div", { style: { padding: '12px 16px' }, children: [!hasData && (_jsx("div", { style: {
                    textAlign: 'center', padding: '24px 0', color: 'var(--text-secondary, #888)',
                    fontSize: 13,
                }, children: t('overview.noMetrics') })), hasData && (_jsxs("div", { style: {
                    display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))',
                    gap: 10,
                }, children: [_jsx(MetricCard, { label: t('overview.epoch'), value: getValue(metrics, 'epoch') }), _jsx(MetricCard, { label: t('overview.step'), value: getValue(metrics, 'step') }), _jsx(MetricCard, { label: t('overview.loss'), value: getValue(metrics, 'loss') }), _jsx(MetricCard, { label: t('overview.accuracy'), value: getValue(metrics, 'accuracy') }), _jsx(MetricCard, { label: t('overview.lr'), value: getValue(metrics, 'learning_rate') }), _jsx(MetricCard, { label: t('overview.status'), value: getValue(metrics, 'status') })] }))] }));
}
function MetricCard({ label, value }) {
    return (_jsxs("div", { style: {
            background: 'var(--card-bg, #222240)', borderRadius: 6, padding: '10px 12px',
            border: '1px solid var(--border, #333)',
        }, children: [_jsx("div", { style: { fontSize: 11, color: 'var(--text-secondary, #888)', marginBottom: 2 }, children: label }), _jsx("div", { style: { fontSize: 16, fontWeight: 600, color: 'var(--accent, #4fc3f7)', fontFamily: 'ui-monospace, SFMono-Regular, monospace' }, children: value })] }));
}
//# sourceMappingURL=MetricsTab.js.map