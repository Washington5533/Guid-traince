/**
 * i18n dictionary keys for Training Guardian plugin.
 *
 * The browser half registers this under the 'training-guardian' namespace
 * via ctx.locale.register(NS, { zh, en }).
 */
export type TgKey = 'panel.title' | 'panel.noData' | 'panel.connecting' | 'panel.disconnected' | 'tab.overview' | 'tab.gpu' | 'tab.anomalies' | 'tab.decisions' | 'overview.epoch' | 'overview.loss' | 'overview.accuracy' | 'overview.lr' | 'overview.step' | 'overview.status' | 'overview.noMetrics' | 'gpu.title' | 'gpu.utilization' | 'gpu.temperature' | 'gpu.memory' | 'gpu.power' | 'gpu.noData' | 'anomalies.title' | 'anomalies.none' | 'anomalies.type' | 'anomalies.description' | 'anomalies.time' | 'anomalies.severity' | 'decisions.title' | 'decisions.none' | 'decisions.tool' | 'decisions.action' | 'decisions.source' | 'decisions.time' | 'decisions.approve' | 'decisions.reject' | 'settings.title' | 'settings.serverUrl' | 'settings.serverUrlHint' | 'settings.authToken' | 'settings.authTokenHint' | 'settings.sessionId' | 'settings.sessionIdHint' | 'settings.autoConnect' | 'settings.saved' | 'severity.low' | 'severity.medium' | 'severity.high' | 'severity.critical' | 'misc.unknown' | 'misc.never' | 'misc.error' | 'misc.warning' | 'misc.info';
declare const zh: Record<TgKey, string>;
declare const en: Record<TgKey, string>;
export { zh, en };
//# sourceMappingURL=locales.d.ts.map