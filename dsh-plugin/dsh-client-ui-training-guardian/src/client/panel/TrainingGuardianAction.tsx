/**
 * Session-header action that opens the Training Guardian monitoring panel as
 * a free-floating, draggable window.
 *
 * - Rendered through a portal to document.body, so no header/overflow/
 *   transform container can clip or block it.
 * - Drag the title bar to move it anywhere; release near a screen edge to
 *   dock (snap) to that edge/corner.
 * - The docked position persists per browser (localStorage).
 * - Outside-pointer click closes the window.
 */
import { useEffect, useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from 'react'
import { createPortal } from 'react-dom'
import type { SseClient } from '../sse/client'
import type { TgKey } from '../locales'
import { TrainingPanel } from './TrainingPanel'

export interface TrainingGuardianActionProps {
  sse: SseClient
  sessionId: string | null
  serverUrl: string
  authToken?: string
  modelEntry?: string
  projectDir?: string
  onApprove: (actionId: string) => void | Promise<void>
  onReject: (actionId: string, reason: string) => void | Promise<void>
  /** Translate function, injected by the slot framework via `locale: 'training-guardian'`. */
  t: (key: TgKey) => string
}

const WIDTH = 520
const MAX_HEIGHT = 0.7 // of viewport height
const EDGE_MARGIN = 8
const SNAP_THRESHOLD = 24 // px — release within this distance of an edge to dock
const POS_KEY = 'training-guardian.panel-pos'

interface PanelPos {
  x: number
  y: number
}

const WINDOW_STYLE: CSSProperties = {
  position: 'fixed',
  width: WIDTH,
  maxWidth: '92vw',
  maxHeight: `${MAX_HEIGHT * 100}vh`,
  display: 'flex',
  flexDirection: 'column',
  background: 'var(--surface, #1e1e2e)',
  border: '1px solid var(--border, #333)',
  borderRadius: 8,
  boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
  zIndex: 10000,
  overflow: 'hidden',
}

const HANDLE_STYLE: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  padding: '6px 10px',
  cursor: 'move',
  userSelect: 'none',
  WebkitUserSelect: 'none',
  background: 'var(--tab-bg, #16162a)',
  borderBottom: '1px solid var(--border, #333)',
  fontSize: 12,
  color: 'var(--text-secondary, #888)',
}

const BODY_STYLE: CSSProperties = {
  overflow: 'auto',
  padding: 12,
  flex: 1,
  minHeight: 0,
}

function loadPos(): PanelPos | null {
  try {
    const raw = localStorage.getItem(POS_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as Partial<PanelPos>
    if (typeof parsed.x === 'number' && typeof parsed.y === 'number') {
      return { x: parsed.x, y: parsed.y }
    }
  } catch { /* corrupted storage — fall through to default */ }
  return null
}

function savePos(pos: PanelPos): void {
  try {
    localStorage.setItem(POS_KEY, JSON.stringify(pos))
  } catch { /* storage unavailable — position just won't persist */ }
}

/** Clamp a position into the viewport, keeping the window fully visible. */
function clampPos(x: number, y: number, winW: number, winH: number): PanelPos {
  const vw = window.innerWidth
  const vh = window.innerHeight
  return {
    x: Math.min(Math.max(x, EDGE_MARGIN), Math.max(EDGE_MARGIN, vw - winW - EDGE_MARGIN)),
    y: Math.min(Math.max(y, EDGE_MARGIN), Math.max(EDGE_MARGIN, vh - winH - EDGE_MARGIN)),
  }
}

/** Snap (dock) near screen edges/corners on release. */
function snapPos(pos: PanelPos, winW: number, winH: number): PanelPos {
  const vw = window.innerWidth
  const vh = window.innerHeight
  const toLeft = pos.x - EDGE_MARGIN
  const toRight = vw - EDGE_MARGIN - winW - pos.x
  const toTop = pos.y - EDGE_MARGIN
  const toBottom = vh - EDGE_MARGIN - winH - pos.y

  let { x, y } = pos
  if (toLeft <= SNAP_THRESHOLD && toLeft <= toRight) x = EDGE_MARGIN
  else if (toRight <= SNAP_THRESHOLD) x = vw - winW - EDGE_MARGIN
  if (toTop <= SNAP_THRESHOLD && toTop <= toBottom) y = EDGE_MARGIN
  else if (toBottom <= SNAP_THRESHOLD) y = vh - winH - EDGE_MARGIN
  return { x, y }
}

/**
 * Header trigger with the guardian panel floating window.
 * @param props - the guardian panel's needs, captured from the apply closure.
 */
export function TrainingGuardianAction(props: TrainingGuardianActionProps) {
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState<PanelPos | null>(null)
  const rootRef = useRef<HTMLDivElement>(null)
  const winRef = useRef<HTMLDivElement>(null)
  const dragRef = useRef<{ startX: number; startY: number; baseX: number; baseY: number } | null>(null)

  // Outside-pointer dismissal: the window lives in a portal, so both the
  // trigger and the window must be treated as "inside".
  useEffect(() => {
    if (!open) return
    const onPointerDown = (event: MouseEvent): void => {
      const target = event.target as Node
      const inside = (rootRef.current?.contains(target) ?? false)
        || (winRef.current?.contains(target) ?? false)
      if (!inside) setOpen(false)
    }
    document.addEventListener('mousedown', onPointerDown)
    return () => { document.removeEventListener('mousedown', onPointerDown) }
  }, [open])

  // Initialize (or restore) the window position on first open.
  useEffect(() => {
    if (!open || pos !== null) return
    const stored = loadPos()
    if (stored) {
      setPos(clampPos(stored.x, stored.y, WIDTH, window.innerHeight * MAX_HEIGHT))
      return
    }
    const rect = rootRef.current?.getBoundingClientRect()
    const x = rect ? Math.min(rect.right - WIDTH, window.innerWidth - WIDTH - EDGE_MARGIN) : window.innerWidth - WIDTH - EDGE_MARGIN
    const y = rect ? rect.bottom + 6 : 80
    setPos(clampPos(x, y, WIDTH, window.innerHeight * MAX_HEIGHT))
  }, [open, pos])

  const onHandlePointerDown = (e: ReactPointerEvent<HTMLDivElement>): void => {
    if (pos === null) return
    // Let buttons (e.g. the ✕ close) receive their own pointer/click events.
    if ((e.target as HTMLElement).closest('button')) return
    e.preventDefault()
    dragRef.current = { startX: e.clientX, startY: e.clientY, baseX: pos.x, baseY: pos.y }
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
  }

  const onHandlePointerMove = (e: ReactPointerEvent<HTMLDivElement>): void => {
    const drag = dragRef.current
    if (drag === null || pos === null) return
    const winW = winRef.current?.offsetWidth ?? WIDTH
    const winH = winRef.current?.offsetHeight ?? window.innerHeight * MAX_HEIGHT
    setPos(clampPos(drag.baseX + (e.clientX - drag.startX), drag.baseY + (e.clientY - drag.startY), winW, winH))
  }

  const onHandlePointerUp = (): void => {
    const drag = dragRef.current
    dragRef.current = null
    if (drag === null || pos === null) return
    const winW = winRef.current?.offsetWidth ?? WIDTH
    const winH = winRef.current?.offsetHeight ?? window.innerHeight * MAX_HEIGHT
    const docked = snapPos(clampPos(pos.x, pos.y, winW, winH), winW, winH)
    setPos(docked)
    savePos(docked)
  }

  const windowNode = open && pos !== null ? (
    <div ref={winRef} style={{ ...WINDOW_STYLE, left: pos.x, top: pos.y }} data-plugin="training-guardian-panel">
      <div
        style={HANDLE_STYLE}
        onPointerDown={onHandlePointerDown}
        onPointerMove={onHandlePointerMove}
        onPointerUp={onHandlePointerUp}
        title="Drag to move — release near a screen edge to dock"
      >
        <span aria-hidden="true" style={{ letterSpacing: -2 }}>⠿</span>
        <span style={{ flex: 1 }}>{props.t ? props.t('panel.title') : 'Training Guardian'}</span>
        <button
          type="button"
          aria-label="Close"
          onClick={() => setOpen(false)}
          onPointerDown={e => e.stopPropagation()}
          style={{
            background: 'transparent', border: 'none', cursor: 'pointer',
            color: 'var(--text-secondary, #888)', fontSize: 13, padding: '0 2px',
          }}
        >
          ✕
        </button>
      </div>
      <div style={BODY_STYLE}>
        <TrainingPanel
          sse={props.sse}
          sessionId={props.sessionId}
          serverUrl={props.serverUrl}
          authToken={props.authToken}
          modelEntry={props.modelEntry}
          projectDir={props.projectDir}
          t={props.t}
          onApprove={props.onApprove}
          onReject={props.onReject}
        />
      </div>
    </div>
  ) : null

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
        {props.t ? props.t('panel.title') : 'TG'}
      </button>
      {windowNode !== null && createPortal(windowNode, document.body)}
    </div>
  )
}
