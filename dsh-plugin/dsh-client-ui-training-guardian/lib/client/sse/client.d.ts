/**
 * SSE client for connecting to a guardian RemoteServer.
 *
 * Maintains a persistent EventSource connection, handles reconnection,
 * and dispatches events to registered listeners.
 */
export type EventHandler = (data: unknown) => void;
export type StatusHandler = (status: ConnectionStatus) => void;
export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';
export interface SseClientOptions {
    url: string;
    authToken?: string;
    reconnectInterval?: number;
    maxReconnectAttempts?: number;
}
export declare class SseClient {
    private url;
    private authToken?;
    private reconnectInterval;
    private maxAttempts;
    private eventSource;
    private status;
    private handlers;
    private statusHandlers;
    private reconnectAttempts;
    private reconnectTimer;
    private intentionalClose;
    private sessionId;
    constructor(opts: SseClientOptions);
    setSessionId(id: string | null): void;
    getStatus(): ConnectionStatus;
    on(event: string, handler: EventHandler): () => void;
    onStatusChange(handler: StatusHandler): () => void;
    connect(): void;
    disconnect(): void;
    private _open;
    private _dispatch;
    private _scheduleReconnect;
    private _clearReconnectTimer;
    private _setStatus;
}
//# sourceMappingURL=client.d.ts.map