/**
 * DSH browser-side entry for Training Guardian.
 *
 * Registered by the host half (src/index.ts) as a `dsh.client` entry.
 * Reads settings from the DSH settings scope, constructs the SSE client,
 * and injects the sidebar panel + settings card into DSH slots.
 */
import { TrainingPanel } from './panel/TrainingPanel';
import { SettingsCard, DEFAULT_SETTINGS, TgSettingsCardController } from './settings/SettingsCard';
import { SseClient } from './sse/client';
import { zh, en } from './locales';
export function apply({ ctx, localeNs = 'training-guardian' }) {
    // ---------- i18n ----------
    ctx.locale.register(localeNs, { zh, en });
    const t = (key) => ctx.locale.t(`${localeNs}:${key}`);
    // ---------- settings ----------
    // dsh-web-ui rc.6 exposes bind() on ctx.webUiSettings when the runtime
    // provides a settings scope. We also fall back to the runtime-level bind
    // if available.
    const settingsBinder = (ctx.webUiSettings ?? ctx);
    let settingsScope = null;
    try {
        settingsScope = settingsBinder.bind({
            default: DEFAULT_SETTINGS,
            namespace: 'training-guardian',
        });
    }
    catch {
        // Settings scope not available in this runtime version.
    }
    const controller = new TgSettingsCardController(settingsScope ?? {
        getSnapshot: () => ({ ...DEFAULT_SETTINGS }),
        setValue: () => { },
    });
    // ---------- SSE client ----------
    const sse = new SseClient({
        url: controller.getSnapshot().serverUrl,
        authToken: controller.getSnapshot().authToken || undefined,
    });
    // ---------- react to settings changes ----------
    controller.subscribe(() => {
        const s = controller.getSnapshot();
        sse.setSessionId(s.sessionId || null);
        if (sse.getStatus() !== 'connected') {
            sse.disconnect();
            sse.connect();
        }
    });
    // ---------- approve / reject -> REST ----------
    async function post(path, body) {
        const base = controller.getSnapshot().serverUrl.replace(/\/$/, '');
        const url = `${base}${path}`;
        const headers = { 'Content-Type': 'application/json' };
        const token = controller.getSnapshot().authToken;
        if (token)
            headers['Authorization'] = `Bearer ${token}`;
        const res = await fetch(url, {
            method: 'POST',
            headers,
            body: JSON.stringify(body),
        });
        if (!res.ok) {
            throw new Error(`REST ${path} failed: ${res.status}`);
        }
    }
    const onApprove = (actionId) => post('/api/decisions/approve', { action_id: actionId });
    const onReject = (actionId, reason) => post('/api/decisions/reject', { action_id: actionId, reason });
    // ---------- inject slots ----------
    try {
        ctx.slots.inject('sidebar.training-guardian', {
            inject: () => ({
                type: 'react',
                component: () => import('react').then(React => React.createElement(TrainingPanel, {
                    sse,
                    sessionId: controller.getSnapshot().sessionId || null,
                    t,
                    onApprove,
                    onReject,
                })),
            }),
        });
    }
    catch {
        // Slot not declared in this runtime; ignore gracefully.
    }
    // Settings card in Web UI plugin group.
    try {
        ctx.slots.inject('web-ui.plugin.item', {
            inject: () => ({
                type: 'react',
                component: () => import('react').then(React => React.createElement(SettingsCard, { controller, t })),
            }),
        });
    }
    catch {
        // Optional slot — ignore when unavailable.
    }
}
//# sourceMappingURL=index.js.map