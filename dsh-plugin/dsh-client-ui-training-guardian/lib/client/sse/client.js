/**
 * SSE client for connecting to a guardian RemoteServer.
 *
 * Maintains a persistent EventSource connection, handles reconnection,
 * and dispatches events to registered listeners.
 */
const DEFAULT_RECONNECT_INTERVAL = 3000;
const DEFAULT_MAX_ATTEMPTS = 10;
export class SseClient {
    url;
    authToken;
    reconnectInterval;
    maxAttempts;
    eventSource = null;
    status = 'disconnected';
    handlers = new Map();
    statusHandlers = new Set();
    reconnectAttempts = 0;
    reconnectTimer = null;
    intentionalClose = false;
    sessionId = null;
    constructor(opts) {
        this.url = opts.url.replace(/\/$/, '');
        this.authToken = opts.authToken;
        this.reconnectInterval = opts.reconnectInterval ?? DEFAULT_RECONNECT_INTERVAL;
        this.maxAttempts = opts.maxAttempts ?? DEFAULT_MAX_ATTEMPTS;
    }
    setSessionId(id) {
        this.sessionId = id;
    }
    getStatus() {
        return this.status;
    }
    on(event, handler) {
        if (!this.handlers.has(event)) {
            this.handlers.set(event, new Set());
        }
        this.handlers.get(event).add(handler);
        return () => { this.handlers.get(event)?.delete(handler); };
    }
    onStatusChange(handler) {
        this.statusHandlers.add(handler);
        return () => { this.statusHandlers.delete(handler); };
    }
    connect() {
        if (this.eventSource && this.eventSource.readyState === EventSource.OPEN) {
            return;
        }
        this.intentionalClose = false;
        this.reconnectAttempts = 0;
        this._open();
    }
    disconnect() {
        this.intentionalClose = true;
        this._clearReconnectTimer();
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
        this._setStatus('disconnected');
    }
    _open() {
        const path = this.sessionId ? `/sse/${this.sessionId}` : '/sse';
        const url = `${this.url}${path}`;
        this._setStatus('connecting');
        try {
            this.eventSource = new EventSource(url);
        }
        catch {
            this._scheduleReconnect();
            return;
        }
        this.eventSource.onopen = () => {
            this.reconnectAttempts = 0;
            this._setStatus('connected');
        };
        this.eventSource.onerror = () => {
            if (this.intentionalClose)
                return;
            this._setStatus('error');
            this._scheduleReconnect();
        };
        this.eventSource.addEventListener('heartbeat', () => {
            // Heartbeat received — connection is alive
        });
        this.eventSource.onmessage = (evt) => {
            try {
                const payload = JSON.parse(evt.data);
                this._dispatch(payload);
            }
            catch {
                // Ignore unparseable messages
            }
        };
    }
    _dispatch(payload) {
        const eventType = typeof payload.type === 'string' ? payload.type : 'message';
        const data = (payload.data ?? payload);
        const handlers = this.handlers.get(eventType);
        if (handlers) {
            for (const fn of handlers) {
                try {
                    fn(data);
                }
                catch { /* swallow handler errors */ }
            }
        }
        const wildcard = this.handlers.get('*');
        if (wildcard) {
            for (const fn of wildcard) {
                try {
                    fn(eventType);
                }
                catch { /* swallow */ }
            }
        }
    }
    _scheduleReconnect() {
        if (this.intentionalClose)
            return;
        if (this.reconnectAttempts >= this.maxAttempts) {
            this._setStatus('disconnected');
            return;
        }
        this._clearReconnectTimer();
        this.reconnectAttempts++;
        this._setStatus('connecting');
        this.reconnectTimer = window.setTimeout(() => {
            this._open();
        }, this.reconnectInterval);
    }
    _clearReconnectTimer() {
        if (this.reconnectTimer !== null) {
            window.clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
    }
    _setStatus(s) {
        if (this.status === s)
            return;
        this.status = s;
        for (const fn of this.statusHandlers) {
            try {
                fn(s);
            }
            catch { /* swallow */ }
        }
    }
}
//# sourceMappingURL=client.js.map