import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useEffect, useCallback } from 'react';
export const DEFAULT_SETTINGS = {
    serverUrl: 'http://localhost:8765',
    authToken: '',
    sessionId: '',
    autoConnect: true,
};
export class TgSettingsCardController {
    scope;
    listeners = new Set();
    constructor(scope) {
        this.scope = scope;
    }
    getSnapshot() {
        const raw = this.scope.getSnapshot();
        if (raw.status === 'ready' && raw.value) {
            return { ...DEFAULT_SETTINGS, ...raw.value };
        }
        return { ...DEFAULT_SETTINGS };
    }
    subscribe(fn) {
        this.listeners.add(fn);
        return () => { this.listeners.delete(fn); };
    }
    update(partial) {
        const current = this.getSnapshot();
        const next = { ...current, ...partial };
        this.scope.setValue(next);
        this.listeners.forEach(fn => { try {
            fn();
        }
        catch { /* swallow */ } });
    }
    dispose() {
        this.listeners.clear();
    }
    inject() {
        // The actual JSX is rendered by the React component registered via slots.
        // This method exists for slot injection compatibility.
        return null;
    }
}
export function SettingsCard({ controller, t }) {
    const [settings, setSettings] = useState(controller.getSnapshot());
    const [saved, setSaved] = useState(false);
    useEffect(() => {
        const unsub = controller.subscribe(() => {
            setSettings(controller.getSnapshot());
        });
        return unsub;
    }, [controller]);
    const handleSubmit = useCallback((e) => {
        e.preventDefault();
        controller.update(settings);
        setSaved(true);
        setTimeout(() => setSaved(false), 2000);
    }, [controller, settings]);
    const handleChange = useCallback((field, value) => {
        setSettings(prev => ({ ...prev, [field]: value }));
    }, []);
    return (_jsxs("form", { onSubmit: handleSubmit, style: { padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: 12 }, children: [_jsxs("div", { children: [_jsx("label", { style: { display: 'block', fontSize: 13, fontWeight: 500, marginBottom: 4 }, children: t('settings.serverUrl') }), _jsx("input", { type: "text", value: settings.serverUrl, onChange: e => handleChange('serverUrl', e.target.value), placeholder: t('settings.serverUrlHint'), style: {
                            width: '100%', padding: '6px 8px', fontSize: 13,
                            border: '1px solid var(--border, #ddd)', borderRadius: 4,
                            background: 'var(--bg, #fff)', color: 'var(--text, #333)',
                        } })] }), _jsxs("div", { children: [_jsx("label", { style: { display: 'block', fontSize: 13, fontWeight: 500, marginBottom: 4 }, children: t('settings.authToken') }), _jsx("input", { type: "password", value: settings.authToken, onChange: e => handleChange('authToken', e.target.value), placeholder: t('settings.authTokenHint'), style: {
                            width: '100%', padding: '6px 8px', fontSize: 13,
                            border: '1px solid var(--border, #ddd)', borderRadius: 4,
                            background: 'var(--bg, #fff)', color: 'var(--text, #333)',
                        } })] }), _jsxs("div", { children: [_jsx("label", { style: { display: 'block', fontSize: 13, fontWeight: 500, marginBottom: 4 }, children: t('settings.sessionId') }), _jsx("input", { type: "text", value: settings.sessionId, onChange: e => handleChange('sessionId', e.target.value), placeholder: t('settings.sessionIdHint'), style: {
                            width: '100%', padding: '6px 8px', fontSize: 13,
                            border: '1px solid var(--border, #ddd)', borderRadius: 4,
                            background: 'var(--bg, #fff)', color: 'var(--text, #333)',
                        } })] }), _jsxs("div", { style: { display: 'flex', alignItems: 'center', gap: 8 }, children: [_jsx("input", { type: "checkbox", id: "tg-auto-connect", checked: settings.autoConnect, onChange: e => handleChange('autoConnect', e.target.checked) }), _jsx("label", { htmlFor: "tg-auto-connect", style: { fontSize: 13 }, children: t('settings.autoConnect') })] }), _jsx("div", { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' }, children: _jsx("button", { type: "submit", style: {
                        padding: '6px 16px', fontSize: 13, cursor: 'pointer',
                        background: 'var(--accent, #007acc)', color: '#fff',
                        border: 'none', borderRadius: 4,
                    }, children: saved ? t('settings.saved') : 'Save' }) })] }));
}
//# sourceMappingURL=SettingsCard.js.map