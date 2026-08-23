/// <reference types="vitest/globals" />

// Polyfill EventSource for jsdom test environment
class EventSourceMock {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSED = 2

  readyState = EventSourceMock.CONNECTING
  onopen: (() => void) | null = null
  onerror: (() => void) | null = null
  onmessage: ((evt: MessageEvent) => void) | null = null
  private listeners: Record<string, Set<(evt: MessageEvent) => void>> = {}

  constructor(public url: string) {}

  addEventListener(type: string, handler: (evt: MessageEvent) => void) {
    if (!this.listeners[type]) this.listeners[type] = new Set()
    this.listeners[type].add(handler)
  }
  removeEventListener(type: string, handler: (evt: MessageEvent) => void) {
    this.listeners[type]?.delete(handler)
  }
  close() {
    this.readyState = EventSourceMock.CLOSED
  }

  // Test helpers
  _emitOpen() {
    this.readyState = EventSourceMock.OPEN
    this.onopen?.()
  }
  _emitMessage(data: unknown) {
    const evt = new MessageEvent('message', { data: JSON.stringify(data) })
    this.onmessage?.(evt)
    for (const fn of this.listeners['message'] ?? []) fn(evt)
  }
  _emitError() {
    this.onerror?.()
  }
}

// @ts-ignore
globalThis.EventSource = EventSourceMock as unknown as typeof EventSource
