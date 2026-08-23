import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
export function AnomaliesTab({ anomalies, t }) {
    if (anomalies.length === 0) {
        return (_jsx("div", { style: {
                textAlign: 'center', padding: '24px 0', color: 'var(--text-secondary, #888)', fontSize: 13,
            }, children: t('anomalies.none') }));
    }
    return (_jsx("div", { style: { padding: 0 }, children: _jsxs("table", { style: {
                width: '100%', borderCollapse: 'collapse', fontSize: 12,
            }, children: [_jsx("thead", { children: _jsxs("tr", { style: { borderBottom: '1px solid var(--border, #333)' }, children: [_jsx(Th, { children: t('anomalies.type') }), _jsx(Th, { children: t('anomalies.severity') }), _jsx(Th, { style: { flex: 1, textAlign: 'left' }, children: t('anomalies.description') }), _jsx(Th, { style: { textAlign: 'right' }, children: t('anomalies.time') })] }) }), _jsx("tbody", { children: anomalies.map((a, i) => {
                        const severity = String(a.severity ?? 'info');
                        const desc = String(a.message ?? a.description ?? '—');
                        const time = String(a.timestamp ?? a.time ?? '');
                        const shortTime = time.length > 19 ? time.slice(11, 19) : time;
                        return (_jsxs("tr", { style: { borderBottom: '1px solid var(--border, #333)' }, children: [_jsx("td", { children: _jsx("code", { style: { color: 'var(--accent, #4fc3f7)' }, children: String(a.type ?? a.event_type ?? '?') }) }), _jsx("td", { children: _jsx(SeverityBadge, { severity: severity }) }), _jsx("td", { style: { color: 'var(--text-secondary, #ccc)', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }, children: desc }), _jsx("td", { style: { textAlign: 'right', color: 'var(--text-secondary, #888)', whiteSpace: 'nowrap' }, children: shortTime })] }, i));
                    }) })] }) }));
}
function Th({ children }) {
    return _jsx("th", { style: { padding: '6px 8px', fontWeight: 500, textAlign: 'left', color: 'var(--text-secondary, #888)', fontSize: 11 }, children: children });
}
function SeverityBadge({ severity }) {
    const colorMap = {
        low: '#66bb6a',
        medium: '#ffa726',
        high: '#ef5350',
        critical: '#ff1744',
        info: '#4fc3f7',
    };
    const c = colorMap[severity.toLowerCase()] || colorMap.info;
    return (_jsx("span", { style: {
            display: 'inline-block', padding: '1px 8px', borderRadius: 10, fontSize: 11, fontWeight: 500,
            background: `${c}22`, color: c, border: `1px solid ${c}44`,
        }, children: severity }));
}
//# sourceMappingURL=AnomaliesTab.js.map