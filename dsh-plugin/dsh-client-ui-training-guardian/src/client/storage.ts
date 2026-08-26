/**
 * Shared localStorage utilities for Training Guardian.
 *
 * Provides:
 * - Per-session metrics history persistence (survives page refresh)
 * - A session registry so HistoryTab can enumerate locally cached sessions
 * - Works independently of the user-configured sessionId setting
 */

/** localStorage key: metrics data for a session. */
const METRICS_KEY = (sid: string) => `tg:metrics-history:${sid}`

/** localStorage key: registry of all known local session IDs. */
const REGISTRY_KEY = 'tg:local-sessions'

/** Max data points kept per session. */
export const MAX_HISTORY = 2000

// ── Metrics history ────────────────────────────────────────────────────

export function loadMetricsHistory(sessionId: string): Array<Record<string, unknown>> {
  try {
    const stored = localStorage.getItem(METRICS_KEY(sessionId))
    return stored ? JSON.parse(stored) : []
  } catch { return [] }
}

export function saveMetricsHistory(
  sessionId: string,
  history: Array<Record<string, unknown>>,
): void {
  try {
    localStorage.setItem(METRICS_KEY(sessionId), JSON.stringify(history))
  } catch { /* quota exceeded — ignore */ }
}

export function clearMetricsHistory(sessionId: string): void {
  try { localStorage.removeItem(METRICS_KEY(sessionId)) } catch { /* ignore */ }
}

// ── Session registry ───────────────────────────────────────────────────

interface LocalSession {
  id: string
  /** ISO timestamp of last metrics update. */
  updatedAt: string
  /** Number of metrics data points stored. */
  pointCount: number
  /** Last known metrics snapshot. */
  lastMetrics: Record<string, unknown>
}

function readRegistry(): LocalSession[] {
  try {
    return JSON.parse(localStorage.getItem(REGISTRY_KEY) || '[]')
  } catch { return [] }
}

function writeRegistry(sessions: LocalSession[]): void {
  try {
    localStorage.setItem(REGISTRY_KEY, JSON.stringify(sessions))
  } catch { /* ignore */ }
}

/**
 * Register / update a local session in the registry.
 * Called whenever metrics history is persisted.
 */
export function registerLocalSession(
  id: string,
  pointCount: number,
  lastMetrics: Record<string, unknown>,
): void {
  const sessions = readRegistry()
  const idx = sessions.findIndex(s => s.id === id)
  const entry: LocalSession = {
    id,
    updatedAt: new Date().toISOString(),
    pointCount,
    lastMetrics,
  }
  if (idx >= 0) {
    sessions[idx] = entry
  } else {
    sessions.unshift(entry)
  }
  // Keep at most 50 entries to avoid unbounded growth.
  writeRegistry(sessions.slice(0, 50))
}

/**
 * List all locally cached training sessions.
 * Used by HistoryTab to show data when the server is offline.
 */
export function listLocalSessions(): LocalSession[] {
  return readRegistry()
}

/**
 * Remove a local session from both the registry and metrics storage.
 */
export function removeLocalSession(id: string): void {
  const sessions = readRegistry().filter(s => s.id !== id)
  writeRegistry(sessions)
  clearMetricsHistory(id)
}
