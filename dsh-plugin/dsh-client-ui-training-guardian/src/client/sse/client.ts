/**
 * SSE client for connecting to a guardian RemoteServer.
 *
 * Maintains a persistent EventSource connection, reconnects with exponential
 * backoff, and dispatches events to registered listeners.
 *
 * Failures are diagnosed rather than swallowed. The browser's EventSource
 * `error` event carries no status code, so on failure the client re-requests
 * the same URL with fetch() to recover the real reason (401 vs unreachable vs
 * wrong path) and publishes it as a structured {@link SseError}.
 */

export type EventHandler = (data: unknown) => void
export type StatusHandler = (status: ConnectionStatus) => void
export type ErrorHandler = (error: SseError | null) => void

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error'

/** Coarse failure cause; drives the localized headline and the advice prompt. */
export type SseErrorKind =
  | 'unauthorized'
  | 'not_found'
  | 'server_error'
  | 'unreachable'
  | 'exhausted'
  | 'unknown'

/** Why a connection attempt failed, in enough detail to act on. */
export interface SseError {
  kind: SseErrorKind
  /** The URL that failed, auth token stripped. */
  url: string
  /** HTTP status when the probe recovered one, else null. */
  status: number | null
  /** EventSource.readyState at failure (0 CONNECTING, 2 CLOSED). */
  readyState: number
  /** 1-based attempt count within the current connect() cycle. */
  attempt: number
  /** Attempts allowed before the client gives up. */
  maxAttempts: number
  /** Probe error message when fetch itself threw; '' otherwise. */
  detail: string
  /** ms epoch of the failure. */
  at: number
}

export interface SseClientOptions {
  url: string
  authToken?: string
  reconnectInterval?: number
  maxReconnectAttempts?: number
}

const DEFAULT_RECONNECT_INTERVAL = 3000
const DEFAULT_MAX_ATTEMPTS = 10
/** Backoff ceiling — a parked reconnect should still recover within a minute. */
const MAX_RECONNECT_INTERVAL = 30000

export class SseClient {
  private url: string
  private authToken: string | undefined
  private reconnectInterval: number
  private maxAttempts: number

  private eventSource: EventSource | null = null
  private status: ConnectionStatus = 'disconnected'
  private handlers: Map<string, Set<EventHandler>> = new Map()
  private statusHandlers: Set<StatusHandler> = new Set()
  private errorHandlers: Set<ErrorHandler> = new Set()
  private lastError: SseError | null = null
  private reconnectAttempts = 0
  private reconnectTimer: number | null = null
  private intentionalClose = false
  private sessionId: string | null = null
  /** Guards against a probe result landing after a newer attempt superseded it. */
  private generation = 0

  constructor(opts: SseClientOptions) {
    this.url = opts.url.replace(/\/$/, '')
    this.authToken = opts.authToken
    this.reconnectInterval = opts.reconnectInterval ?? DEFAULT_RECONNECT_INTERVAL
    this.maxAttempts = opts.maxReconnectAttempts ?? DEFAULT_MAX_ATTEMPTS
  }

  setSessionId(id: string | null): void {
    this.sessionId = id
  }

  /**
   * Re-target the connection. Settings edits must take effect without a page
   * reload, so url/token live behind a setter rather than being frozen at
   * construction.
   * @param opts - new base URL and/or auth token.
   * @returns true when something actually changed (caller should reconnect).
   */
  setEndpoint(opts: { url?: string; authToken?: string | undefined }): boolean {
    let changed = false
    if (opts.url !== undefined) {
      const next = opts.url.replace(/\/$/, '')
      if (next !== this.url) {
        this.url = next
        changed = true
      }
    }
    if ('authToken' in opts) {
      const next = opts.authToken || undefined
      if (next !== this.authToken) {
        this.authToken = next
        changed = true
      }
    }
    return changed
  }

  getStatus(): ConnectionStatus {
    return this.status
  }

  /** The most recent failure, or null while healthy. */
  getLastError(): SseError | null {
    return this.lastError
  }

  /**
   * Observe connection failures with their diagnosed cause. Fires with null
   * when a connection succeeds, so subscribers can clear a stale banner.
   * @param handler - called on each diagnosis.
   * @returns unsubscribe.
   */
  onError(handler: ErrorHandler): () => void {
    this.errorHandlers.add(handler)
    return () => { this.errorHandlers.delete(handler) }
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

  /**
   * Restart the connection from a clean slate — the retry affordance behind
   * the panel's error banner, and the path a settings change takes.
   */
  reconnectNow(): void {
    this.disconnect()
    this.lastError = null
    this._emitError(null)
    this.connect()
  }

  disconnect(): void {
    this.intentionalClose = true
    this.generation += 1
    this._clearReconnectTimer()
    if (this.eventSource) {
      this.eventSource.close()
      this.eventSource = null
    }
    this._setStatus('disconnected')
  }

  /** The stream URL for the current session, without the auth token. */
  private _streamUrl(): string {
    return `${this.url}${this.sessionId ? `/sse/${this.sessionId}` : '/sse'}`
  }

  private _open(): void {
    // EventSource cannot set custom headers, so pass the auth token as a
    // query parameter instead. The RemoteServer validates it server-side.
    const safeUrl = this._streamUrl()
    const token = this.authToken ? `?token=${encodeURIComponent(this.authToken)}` : ''
    const url = `${safeUrl}${token}`
    const generation = ++this.generation
    this._setStatus('connecting')

    try {
      this.eventSource = new EventSource(url)
    } catch (e) {
      this._diagnose(safeUrl, 0, generation, e)
      return
    }

    this.eventSource.onopen = () => {
      this.reconnectAttempts = 0
      this.lastError = null
      this._setStatus('connected')
      this._emitError(null)
    }

    this.eventSource.onerror = () => {
      if (this.intentionalClose || generation !== this.generation) return
      const readyState = this.eventSource?.readyState ?? 2
      // A CLOSED readyState means the handshake was rejected outright and the
      // browser will not retry on its own; CONNECTING is a transient drop that
      // it retries internally, but we still own the user-visible status.
      this._setStatus('error')
      this._diagnose(safeUrl, readyState, generation, null)
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

  /**
   * Recover the real failure reason. The EventSource error event carries no
   * status, so re-request the same URL with fetch(): a 401 from the auth check,
   * a 404 from a wrong path, and a refused/DNS failure are indistinguishable
   * at the EventSource layer but not here.
   */
  private _diagnose(safeUrl: string, readyState: number, generation: number, thrown: unknown): void {
    const attempt = this.reconnectAttempts + 1
    const base: Omit<SseError, 'kind' | 'status' | 'detail'> = {
      url: safeUrl,
      readyState,
      attempt,
      maxAttempts: this.maxAttempts,
      at: Date.now(),
    }

    const settle = (kind: SseErrorKind, status: number | null, detail: string): void => {
      if (this.intentionalClose || generation !== this.generation) return
      this.lastError = { ...base, kind, status, detail }
      this._emitError(this.lastError)
      this._scheduleReconnect()
    }

    if (thrown !== null) {
      settle('unknown', null, thrown instanceof Error ? thrown.message : String(thrown))
      return
    }

    const headers: Record<string, string> = {}
    if (this.authToken) headers['X-Auth-Token'] = this.authToken
    // The probe hits the same auth gate as the stream. Its body is a live SSE
    // stream, so never read it — the status line is the whole point.
    fetch(`${safeUrl}${this.authToken ? `?token=${encodeURIComponent(this.authToken)}` : ''}`, {
      method: 'GET',
      headers,
    }).then((res) => {
      const kind: SseErrorKind = res.status === 401 || res.status === 403
        ? 'unauthorized'
        : res.status === 404
          ? 'not_found'
          : res.status >= 500
            ? 'server_error'
            : 'unknown'
      settle(kind, res.status, '')
    }).catch((e: unknown) => {
      // fetch() rejects for DNS failure, connection refused, and CORS denial.
      settle('unreachable', null, e instanceof Error ? e.message : String(e))
    })
  }

  private _emitError(error: SseError | null): void {
    for (const fn of [...this.errorHandlers]) {
      try { fn(error) } catch { /* swallow handler errors */ }
    }
  }

  private _scheduleReconnect(): void {
    if (this.intentionalClose) return
    if (this.reconnectAttempts >= this.maxAttempts) {
      // Budget spent. Park in a terminal state, but keep the diagnosis on
      // screen and re-label it so the banner can offer a retry.
      if (this.lastError) {
        this.lastError = { ...this.lastError, kind: 'exhausted' }
        this._emitError(this.lastError)
      }
      this._setStatus('disconnected')
      return
    }

    this._clearReconnectTimer()
    this.reconnectAttempts++
    this._setStatus('connecting')

    // Exponential backoff with a ceiling: a server that is down for a minute
    // should not be hammered every 3s for the whole retry budget.
    const delay = Math.min(
      this.reconnectInterval * 2 ** (this.reconnectAttempts - 1),
      MAX_RECONNECT_INTERVAL,
    )
    this.reconnectTimer = window.setTimeout(() => {
      this._open()
    }, delay)
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
