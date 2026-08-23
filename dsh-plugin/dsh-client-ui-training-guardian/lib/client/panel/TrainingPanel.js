import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useEffect, useCallback, useRef } from 'react';
import { MetricsTab } from './MetricsTab';
import { GpuTab } from './GpuTab';
import { AnomaliesTab } from './AnomaliesTab';
import { DecisionsTab } from './DecisionsTab';
const TABS = [
    { key: 'overview', labelKey: 'tab.overview' },
    { key: 'gpu', labelKey: 'tab.gpu' },
    { key: 'anomalies', labelKey: 'tab.anomalies' },
    { key: 'decisions', labelKey: 'tab.decisions' },
];
export function TrainingPanel({ sse, sessionId, t, onApprove, onReject }) {
    const [activeTab, setActiveTab] = useState('overview');
    const [metrics, setMetrics] = useState({});
    const [gpuStatus, setGpuStatus] = useState(null);
    const [anomalies, setAnomalies] = useState([]);
    const [pendingActions, setPendingActions] = useState([]);
    const [connectionStatus, setConnectionStatus] = useState('disconnected');
    const unsubscribers = useRef([]);
    useEffect(() => {
        const unsubs = [];
        const unsubStatus = sse.onStatusChange((status) => {
            setConnectionStatus(status);
        });
        unsubs.push(unsubStatus);
        const unsubMetrics = sse.on('metrics', (data) => {
            setMetrics(data);
        });
        unsubs.push(unsubMetrics);
        const unsubGpu = sse.on('gpu_status', (data) => {
            setGpuStatus(data);
        });
        unsubs.push(unsubGpu);
        const unsubAnomaly = sse.on('anomaly', (data) => {
            setAnomalies(prev => [data, ...prev].slice(0, 50));
        });
        unsubs.push(unsubAnomaly);
        const unsubDecision = sse.on('decision', (data) => {
            setPendingActions(prev => [data, ...prev].slice(0, 20));
        });
        unsubs.push(unsubDecision);
        unsubscribers.current = unsubs;
        sse.connect();
        return () => {
            unsubs.forEach(fn => { try {
                fn();
            }
            catch { /* swallow */ } });
        };
    }, [sse]);
    // Refresh pending actions periodically
    useEffect(() => {
        if (!sessionId || activeTab !== 'decisions')
            return;
        const timer = window.setInterval(() => {
            // The SSE 'decision' event handles real-time updates.
            // This timer exists as a fallback for decisions that arrive before the tab opens.
        }, 5000);
        return () => window.clearInterval(timer);
    }, [sessionId, activeTab]);
    const connected = connectionStatus === 'connected';
    const statusText = connected ? '' : connectionStatus === 'connecting' ? t('panel.connecting') : t('panel.disconnected');
    return (_jsxs("div", { style: {
            display: 'flex', flexDirection: 'column', height: '100%',
            background: 'var(--panel-bg, #1a1a2e)', color: 'var(--text, #e0e0e0)',
            fontFamily: 'system-ui, -apple-system, sans-serif', fontSize: 13,
        }, children: [_jsx("div", { style: {
                    display: 'flex', borderBottom: '1px solid var(--border, #333)',
                    background: 'var(--tab-bg, #16162a)',
                }, children: TABS.map(tab => (_jsx("button", { onClick: () => setActiveTab(tab.key), style: {
                        flex: 1, padding: '8px 4px', fontSize: 12, cursor: 'pointer',
                        border: 'none', borderBottom: activeTab === tab.key ? '2px solid var(--accent, #007acc)' : '2px solid transparent',
                        background: activeTab === tab.key ? 'var(--tab-active, #1e1e3a)' : 'transparent',
                        color: activeTab === tab.key ? 'var(--accent, #4fc3f7)' : 'var(--text-secondary, #888)',
                        transition: 'all 0.15s',
                    }, children: t(tab.labelKey) }, tab.key))) }), statusText && (_jsx("div", { style: {
                    padding: '4px 12px', fontSize: 11,
                    background: connected ? 'transparent' : 'var(--warning-bg, #332200)',
                    color: connected ? 'var(--text)' : 'var(--warning-text, #ffa726)',
                    textAlign: 'center',
                }, children: statusText })), _jsxs("div", { style: { flex: 1, overflow: 'auto' }, children: [activeTab === 'overview' && (_jsx(MetricsTab, { metrics: metrics, t: t })), activeTab === 'gpu' && (_jsx(GpuTab, { gpuStatus: gpuStatus, t: t })), activeTab === 'anomalies' && (_jsx(AnomaliesTab, { anomalies: anomalies, t: t })), activeTab === 'decisions' && (_jsx(DecisionsTab, { pending: pendingActions, t: t, onApprove: onApprove, onReject: onReject }))] })] }));
}
//# sourceMappingURL=TrainingPanel.js.map