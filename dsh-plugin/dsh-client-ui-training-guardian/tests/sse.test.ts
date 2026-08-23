import { describe, it, expect, vi } from 'vitest'
import { SseClient } from '../src/client/sse/client'

describe('SseClient', () => {
  it('subscribes to named events and dispatches payloads', () => {
    const client = new SseClient({ url: 'http://localhost:8765' })
    const handler = vi.fn()
    client.on('metrics', handler)

    // Open the connection
    const es = client as unknown as { eventSource: InstanceType<typeof EventSource> }
    // @ts-ignore — exercise internals
    client._open()
    es.eventSource._emitOpen()
    es.eventSource._emitMessage({ type: 'metrics', data: { loss: 0.5 } })

    expect(handler).toHaveBeenCalledTimes(1)
    expect(handler).toHaveBeenCalledWith({ loss: 0.5 })
  })

  it('calls status handlers on connect', () => {
    const client = new SseClient({ url: 'http://localhost:8765' })
    const statusHandler = vi.fn()
    client.onStatusChange(statusHandler)

    const es = client as unknown as { eventSource: InstanceType<typeof EventSource> }
    // @ts-ignore
    client._open()
    es.eventSource._emitOpen()

    expect(statusHandler).toHaveBeenCalledWith('connected')
  })

  it('unsubscribes correctly', () => {
    const client = new SseClient({ url: 'http://localhost:8765' })
    const handler = vi.fn()
    const unsub = client.on('metrics', handler)
    unsub()

    const es = client as unknown as { eventSource: InstanceType<typeof EventSource> }
    // @ts-ignore
    client._open()
    es.eventSource._emitOpen()
    es.eventSource._emitMessage({ type: 'metrics', data: {} })

    expect(handler).not.toHaveBeenCalled()
  })
})
