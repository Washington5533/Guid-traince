/**
 * SSE client for connecting to a guardian RemoteServer.
 *
 * Maintains a persistent EventSource connection, handles reconnection,
 * and dispatches events to registered listeners.
 */

export type EventHandler = (data: unknown) => void
export type StatusHandler = (status: ConnectionStatus) => void

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error'

export interface SseClientOptions {
  url: string
  authToken?: string
  reconnectInterval?: number
  maxReconnectAttempts?: number
}

const DEFAULT_RECONNECT_INTERVAL = 3000
const DEFAULT_MAX_ATTEMPTS = 10

export class SseClient {
  private url: string
  private authToken?: string
  private reconnectInterval: number
  private maxAttempts: number

  private eventSource: EventSource | null = null
  private status: ConnectionStatus = 'disconnected'
  private handlers: Map<string, Set<EventHandler>> = new Map()
  private statusHandlers: Set<StatusHandler> = new Set()
  private reconnectAttempts = 0
  private reconnectTimer: number | null = null
  private intentionalClose = false
  private sessionId: string | null = null

  constructor(opts: SseClientOptions) {
    this.url = opts.url.replace(/\/$/, '')
    this.authToken = opts.authToken
    this.reconnectInterval = opts.reconnectInterval ?? DEFAULT_RECONNECT_INTERVAL
    this.maxAttempts = opts.maxReconnectAttempts ?? DEFAULT_MAX_ATTEMPTS
  }

  setSessionId(id: string | null): void {
    this.sessionId = id
  }

  getStatus(): ConnectionStatus {
    return this.status
  }

  on(event: string, handler: EventHandler): () => void {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, new Set())
    }
    this.handlers.get(event)!.add(handler)
    return () => { this.handlers.get(event)?.delete(handler) }
  }

  onStatusChange(handler: StatusHandler): () => void {
    this.statusHandlers.add(handler)
    return () => { this.statusHandlers.delete(handler) }
  }

  connect(): void {
    if (this.eventSource && this.eventSource.readyState === EventSource.OPEN) {
      return
    }
    this.intentionalClose = false
    this.reconnectAttempts = 0
    this._open()
  }

  disconnect(): void {
    this.intentionalClose = true
    this._clearReconnectTimer()
    if (this.eventSource) {
      this.eventSource.close()
      this.eventSource = null
    }
    this._setStatus('disconnected')
  }

  private _open(): void {
    const path = this.sessionId ? `/sse/${this.sessionId}` : '/sse'
    const url = `${this.url}${path}`
    this._setStatus('connecting')

    try {
      this.eventSource = new EventSource(url)
    } catch {
      this._scheduleReconnect()
      return
    }

    this.eventSource.onopen = () => {
      this.reconnectAttempts = 0
      this._setStatus('connected')
    }

    this.eventSource.onerror = () => {
      if (this.intentionalClose) return
      this._setStatus('error')
      this._scheduleReconnect()
    }

    this.eventSource.addEventListener('heartbeat', () => {
      // Heartbeat received — connection is alive
    })

    this.eventSource.onmessage = (evt) => {
      try {
        const payload = JSON.parse(evt.data)
        this._dispatch(payload)
      } catch {
        // Ignore unparseable messages
      }
    }
  }

  private _dispatch(payload: Record<string, unknown>): void {
    const eventType = typeof payload.type === 'string' ? payload.type : 'message'
    const data = (payload.data ?? payload) as Record<string, unknown>

    const handlers = this.handlers.get(eventType)
    if (handlers) {
      for (const fn of handlers) {
        try { fn(data) } catch { /* swallow handler errors */ }
      }
    }

    const wildcard = this.handlers.get('*')
    if (wildcard) {
      for (const fn of wildcard) {
        try { fn(eventType) } catch { /* swallow */ }
      }
    }
  }

  private _scheduleReconnect(): void {
    if (this.intentionalClose) return
    if (this.reconnectAttempts >= this.maxAttempts) {
      this._setStatus('disconnected')
      return
    }

    this._clearReconnectTimer()
    this.reconnectAttempts++
    this._setStatus('connecting')

    this.reconnectTimer = window.setTimeout(() => {
      this._open()
    }, this.reconnectInterval)
  }

  private _clearReconnectTimer(): void {
    if (this.reconnectTimer !== null) {
      window.clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
  }

  private _setStatus(s: ConnectionStatus): void {
    if (this.status === s) return
    this.status = s
    for (const fn of this.statusHandlers) {
      try { fn(s) } catch { /* swallow */ }
    }
  }
}
