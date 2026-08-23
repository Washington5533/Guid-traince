import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useEffect, useCallback } from 'react';
export function DecisionsTab({ pending, t, onApprove, onReject }) {
    const [rejectingId, setRejectingId] = useState(null);
    const [rejectReason, setRejectReason] = useState('');
    useEffect(() => {
        setRejectingId(null);
        setRejectReason('');
    }, [pending]);
    const handleReject = useCallback((actionId) => {
        if (!rejectReason.trim())
            return;
        onReject(actionId, rejectReason);
        setRejectingId(null);
        setRejectReason('');
    }, [onReject, rejectReason]);
    if (pending.length === 0) {
        return (_jsx("div", { style: {
                textAlign: 'center', padding: '24px 0', color: 'var(--text-secondary, #888)', fontSize: 13,
            }, children: t('decisions.none') }));
    }
    return (_jsx("div", { style: { padding: 0 }, children: _jsxs("table", { style: { width: '100%', borderCollapse: 'collapse', fontSize: 12 }, children: [_jsx("thead", { children: _jsxs("tr", { style: { borderBottom: '1px solid var(--border, #333)' }, children: [_jsx(Th, { children: t('decisions.tool') }), _jsx(Th, { children: t('decisions.action') }), _jsx(Th, { children: t('decisions.source') }), _jsx(Th, { style: { textAlign: 'right' }, children: t('decisions.time') }), _jsx(Th, { style: { textAlign: 'center', width: 180 } })] }) }), _jsx("tbody", { children: pending.map((d, i) => {
                        const id = String(d.id ?? d.action_id ?? `d-${i}`);
                        const tool = String(d.tool ?? d.source ?? '—');
                        const action = typeof d.action === 'string' ? d.action : (typeof d.action === 'object' ? JSON.stringify(d.action) : '—');
                        const source = String(d.agent_id ?? d.phase ?? d.phase_inferred ?? '—');
                        const time = String(d.timestamp ?? d.time ?? '');
                        const shortTime = time.length > 19 ? time.slice(11, 19) : time;
                        const isRejecting = rejectingId === id;
                        return (_jsxs("tr", { style: { borderBottom: '1px solid var(--border, #333)' }, children: [_jsx("td", { children: _jsx("code", { style: { color: 'var(--accent, #4fc3f7)' }, children: tool }) }), _jsx("td", { style: { maxWidth: 260, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }, children: action }), _jsx("td", { children: source }), _jsx("td", { style: { textAlign: 'right', color: 'var(--text-secondary, #888)', whiteSpace: 'nowrap' }, children: shortTime }), _jsx("td", { style: { textAlign: 'center' }, children: !isRejecting ? (_jsxs("div", { style: { display: 'flex', gap: 6, justifyContent: 'center' }, children: [_jsx("button", { onClick: () => onApprove(id), style: {
                                                    padding: '3px 10px', fontSize: 11, cursor: 'pointer',
                                                    background: 'var(--accent, #007acc)', color: '#fff',
                                                    border: 'none', borderRadius: 3,
                                                }, children: t('decisions.approve') }), _jsx("button", { onClick: () => setRejectingId(id), style: {
                                                    padding: '3px 10px', fontSize: 11, cursor: 'pointer',
                                                    background: 'transparent', color: 'var(--warning-text, #ffa726)',
                                                    border: '1px solid var(--warning-text, #ffa726)', borderRadius: 3,
                                                }, children: t('decisions.reject') })] })) : (_jsxs("div", { style: { display: 'flex', gap: 4, justifyContent: 'center', alignItems: 'center' }, children: [_jsx("input", { value: rejectReason, onChange: e => setRejectReason(e.target.value), placeholder: t('misc.warning'), style: {
                                                    width: 100, padding: '3px 6px', fontSize: 11,
                                                    border: '1px solid var(--border, #ddd)', borderRadius: 3,
                                                    background: 'var(--bg, #fff)', color: 'var(--text, #333)',
                                                } }), _jsx("button", { onClick: () => handleReject(id), style: {
                                                    padding: '3px 8px', fontSize: 11, cursor: 'pointer',
                                                    background: 'var(--warning-text, #ffa726)', color: '#000',
                                                    border: 'none', borderRadius: 3,
                                                }, children: "OK" })] })) })] }, i));
                    }) })] }) }));
}
function Th({ children }) {
    return _jsx("th", { style: { padding: '6px 8px', fontWeight: 500, textAlign: 'left', color: 'var(--text-secondary, #888)', fontSize: 11 }, children: children });
}
//# sourceMappingURL=DecisionsTab.js.map