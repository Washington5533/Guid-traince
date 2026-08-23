/**
 * Session-header action that opens the Training Guardian monitoring panel in
 * a popover. Mirrors the ui-jobs header-action pattern (button + popover with
 * outside-pointer dismissal) but stays fully self-contained: inline styles
 * only, so the tsdown client bundle needs no CSS-module processing and no
 * extra ui-primitives dependency.
 */
import { useEffect, useRef, useState, type CSSProperties } from 'react'
import type { SseClient } from '../sse/client'
import type { TgKey } from '../locales'
import { TrainingPanel } from './TrainingPanel'

export interface TrainingGuardianActionProps {
  sse: SseClient
  sessionId: string | null
  serverUrl: string
  modelEntry?: string
  projectDir?: string
  t: (key: TgKey) => string
  onApprove: (actionId: string) => void
  onReject: (actionId: string, reason: string) => void
}

const POPOVER: CSSProperties = {
  position: 'absolute',
  right: 0,
  top: 'calc(100% + 6px)',
  width: 520,
  maxWidth: '82vw',
  maxHeight: '70vh',
  overflow: 'auto',
  background: 'var(--surface, #1e1e2e)',
  border: '1px solid var(--border, #333)',
  borderRadius: 8,
  boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
  zIndex: 1000,
  padding: 12,
}

/**
 * Header trigger with the guardian panel popover.
 * @param props - the guardian panel's needs, captured from the apply closure.
 */
export function TrainingGuardianAction(props: TrainingGuardianActionProps) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onPointerDown = (event: MouseEvent): void => {
      if (rootRef.current !== null && !rootRef.current.contains(event.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', onPointerDown)
    return () => { document.removeEventListener('mousedown', onPointerDown) }
  }, [open])

  return (
    <div ref={rootRef} style={{ position: 'relative', display: 'inline-block' }}>
      <button
        type="button"
        onClick={() => setOpen(value => !value)}
        aria-expanded={open}
        style={{
          background: 'transparent',
          border: 'none',
          cursor: 'pointer',
          color: 'var(--text, #eee)',
          padding: '4px 8px',
          fontSize: 13,
        }}
      >
        {props.t('panel.title')}
      </button>
      {open && (
        <div style={POPOVER}>
          <TrainingPanel
            sse={props.sse}
            sessionId={props.sessionId}
            serverUrl={props.serverUrl}
            modelEntry={props.modelEntry}
            projectDir={props.projectDir}
            t={props.t}
            onApprove={props.onApprove}
            onReject={props.onReject}
          />
        </div>
      )}
    </div>
  )
}
