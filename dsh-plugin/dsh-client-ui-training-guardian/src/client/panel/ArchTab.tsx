import { useState, useEffect, useRef, useCallback } from 'react'
import * as d3 from 'd3'
import type { TgKey } from '../locales'

export interface ArchNarration {
  narration: string | null
  error: string | null
  model_name: string
  total_params: number
  bottleneck_count: number
}

interface ArchTabProps {
  serverUrl: string
  authToken?: string
  sessionId?: string | null
  modelEntry?: string
  projectDir?: string
  t: (key: TgKey) => string
  archNarration?: ArchNarration | null
}

type ViewMode = 'treemap' | 'backbone'
type ColorBy = 'params' | 'flops'

interface TreeNode {
  name: string
  type: string
  params: number
  flops: number
  repeat: number
  depth: number
  params_pct: number
  children: TreeNode[]
}

interface Bottleneck {
  layer: string
  type: string
  params_pct: number
  flops_pct: number
  severity: string
  params: number
  flops: number
}

interface AnalysisResult {
  ok: boolean
  model_name: string
  total_params: number
  total_flops: number
  total_flops_m: number
  module_count: number
  layer_count: number
  bottleneck_count: number
  bottlenecks: Bottleneck[]
  tree: { name: string; children: TreeNode[] }
  elapsed_ms: number
  agent_pending?: boolean
}

function fmtNum(n: number): string {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M'
  if (n >= 1e3) return (n / 1e3).toFixed(0) + 'K'
  return String(n)
}

export function ArchTab({ serverUrl, authToken, sessionId, modelEntry, projectDir, t, archNarration }: ArchTabProps) {
  const [view, setView] = useState<ViewMode>('treemap')
  const [colorBy, setColorBy] = useState<ColorBy>('params')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const vizRef = useRef<HTMLDivElement>(null)

  const runAnalysis = useCallback(async () => {
    if (!modelEntry) return
    setLoading(true)
    setError(null)
    try {
      const base = serverUrl.replace(/\/$/, '')
      const headers: Record<string, string> = { 'Content-Type': 'application/json' }
      if (authToken) headers['X-Auth-Token'] = authToken
      const res = await fetch(`${base}/api/arch/analyze`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          model_entry: modelEntry,
          project_dir: projectDir,
          session_id: sessionId || '',
          agent: true,
        }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }))
        throw new Error(err.error || `HTTP ${res.status}`)
      }
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      setResult(data)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [serverUrl, authToken, sessionId, modelEntry, projectDir])

  // D3 rendering — re-runs when result, view, or colorBy changes.
  // The onSelect callback bridges d3 click events back to React state.
  useEffect(() => {
    if (!result?.tree?.children?.length || !vizRef.current) return
    const container = vizRef.current
    container.innerHTML = ''

    const w = container.clientWidth || 600
    const h = container.clientHeight || 500
    const svg = d3.select(container).append('svg').attr('viewBox', [0, 0, w, h])
    const nodes = result.tree.children

    if (view === 'treemap') {
      renderTreemap(svg, nodes, w, h, colorBy, setSelectedNode)
    } else {
      renderBackbone(svg, nodes, w, h, colorBy, setSelectedNode)
    }
  }, [result, view, colorBy])

  // Derive the selected node's data for the detail panel.
  const detailNode = selectedNode && result?.tree?.children
    ? result.tree.children.find(n => n.name === selectedNode) ?? null
    : null

  const detailMetrics = detailNode && result?.tree?.children
    ? (() => {
        const mp = Math.max(1, ...result.tree.children.map(n => n.params || 0))
        const mf = Math.max(1, ...result.tree.children.map(n => n.flops || 0))
        return { paramsPct: (detailNode.params / mp) * 100, flopsPct: (detailNode.flops / mf) * 100 }
      })()
    : null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: 'var(--panel-bg, #1a1a2e)', color: 'var(--text, #e0e0e0)' }}>
      {/* Toolbar */}
      <div style={{ display: 'flex', gap: 8, padding: '8px 12px', borderBottom: '1px solid var(--border, #333)', alignItems: 'center', flexWrap: 'wrap' }}>
        <button onClick={runAnalysis} disabled={loading || !modelEntry} style={{
          padding: '4px 14px', fontSize: 12, cursor: loading ? 'wait' : 'pointer',
          background: 'var(--accent, #007acc)', color: '#fff', border: 'none', borderRadius: 4,
          opacity: loading || !modelEntry ? 0.6 : 1,
        }}>
          {loading ? t('arch.loading') : t('arch.analyzeBtn')}
        </button>
        {!modelEntry && (
          <span style={{ fontSize: 11, color: 'var(--text-secondary, #888)', fontStyle: 'italic' }}>
            {t('arch.modelEntryMissing')}
          </span>
        )}
        <div style={{ width: 1, height: 20, background: 'var(--border, #333)', margin: '0 4px' }} />
        <div style={{ display: 'flex', gap: 3 }}>
          <button onClick={() => setView('treemap')} style={{
            padding: '3px 8px', fontSize: 11, cursor: 'pointer', borderRadius: 4,
            border: '1px solid var(--border, #333)', background: view === 'treemap' ? 'var(--accent, #007acc)' : 'transparent',
            color: view === 'treemap' ? '#fff' : 'var(--text-secondary, #888)',
          }}>{t('arch.viewTreemap')}</button>
          <button onClick={() => setView('backbone')} style={{
            padding: '3px 8px', fontSize: 11, cursor: 'pointer', borderRadius: 4,
            border: '1px solid var(--border, #333)', background: view === 'backbone' ? 'var(--accent, #007acc)' : 'transparent',
            color: view === 'backbone' ? '#fff' : 'var(--text-secondary, #888)',
          }}>{t('arch.viewBackbone')}</button>
        </div>
        <div style={{ display: 'flex', gap: 3 }}>
          <button onClick={() => setColorBy('params')} style={{
            padding: '3px 8px', fontSize: 11, cursor: 'pointer', borderRadius: 4,
            border: '1px solid var(--border, #333)', background: colorBy === 'params' ? 'var(--accent, #007acc)' : 'transparent',
            color: colorBy === 'params' ? '#fff' : 'var(--text-secondary, #888)',
          }}>{t('arch.colorParams')}</button>
          <button onClick={() => setColorBy('flops')} style={{
            padding: '3px 8px', fontSize: 11, cursor: 'pointer', borderRadius: 4,
            border: '1px solid var(--border, #333)', background: colorBy === 'flops' ? 'var(--accent, #007acc)' : 'transparent',
            color: colorBy === 'flops' ? '#fff' : 'var(--text-secondary, #888)',
          }}>{t('arch.colorFlops')}</button>
        </div>
        {result && (
          <span style={{ fontSize: 11, color: 'var(--text-secondary, #888)', marginLeft: 'auto' }}>
            {result.elapsed_ms}ms | {result.module_count} {t('arch.moduleCount').toLowerCase()}
            {result.agent_pending && !archNarration && (
              <span style={{ marginLeft: 8, color: 'var(--accent, #4fc3f7)', fontStyle: 'italic' }}>
                {t('arch.agentDispatched')}
              </span>
            )}
          </span>
        )}
      </div>

      {/* Body: viz + sidebar + detail panel */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Visualization */}
        <div ref={vizRef} style={{ flex: 1, overflow: 'hidden', position: 'relative', background: 'var(--bg, #1a1a2e)' }} />

        {/* Sidebar: bottlenecks */}
        <div style={{ width: 240, minWidth: 240, borderLeft: '1px solid var(--border, #333)', background: 'var(--tab-bg, #16162a)', padding: 12, overflowY: 'auto' }}>
          {result && (
            <>
              <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-secondary, #888)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 8 }}>
                {t('arch.bottlenecks')} {result.bottleneck_count > 0 ? `(${result.bottleneck_count} ${t('arch.bottleneckCount')})` : ''}
              </div>
              {result.bottlenecks.length === 0 ? (
                <div style={{ fontSize: 11, color: 'var(--text-secondary, #666)', padding: '8px 0' }}>{t('arch.noBottlenecks')}</div>
              ) : result.bottlenecks.map((bn, i) => (
                <div key={i} onClick={() => setSelectedNode(bn.layer)} style={{
                  padding: '6px 8px', marginBottom: 4, borderRadius: 4, cursor: 'pointer',
                  borderLeft: `3px solid ${bn.severity === 'critical' ? '#cf222e' : bn.severity === 'warning' ? '#bf8700' : '#0969da'}`,
                  background: 'var(--bg, #1a1a2e)', border: '1px solid var(--border, #333)', borderLeftWidth: 3,
                }}>
                  <div style={{ fontSize: 11, fontWeight: 600, wordBreak: 'break-all' }}>{bn.layer}</div>
                  <div style={{ fontSize: 10, color: 'var(--text-secondary, #888)', marginTop: 2 }}>
                    P: {(bn.params_pct || 0).toFixed(1)}% | F: {(bn.flops_pct || 0).toFixed(1)}%
                  </div>
                </div>
              ))}
            </>
          )}
        </div>

        {/* Detail panel — rendered via React, not innerHTML */}
        {detailNode && detailMetrics && (
          <div style={{ width: 240, minWidth: 240, borderLeft: '1px solid var(--border, #333)', background: 'var(--tab-bg, #16162a)', padding: 12, overflowY: 'auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
              <div style={{ fontSize: 13, fontWeight: 700 }}>{detailNode.name.split('.').pop()}</div>
              <button
                onClick={() => setSelectedNode(null)}
                style={{
                  width: 24, height: 24, border: '1px solid var(--border, #333)', background: 'none',
                  color: 'var(--text-secondary, #888)', cursor: 'pointer', borderRadius: 4, fontSize: 12,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}
              >✕</button>
            </div>
            <DetailRow label="Type" value={detailNode.type || '?'} />
            {detailNode.repeat > 1 && (
              <DetailRow label={t('arch.repeat')} value={`x${detailNode.repeat}`} />
            )}
            <DetailRow label={t('arch.params')} value={`${fmtNum(detailNode.params)} (${detailMetrics.paramsPct.toFixed(1)}%)`} />
            <ProgressBar pct={detailMetrics.paramsPct} color="#1a7f37" />
            <DetailRow label={t('arch.flops')} value={fmtNum(detailNode.flops)} />
            <ProgressBar pct={detailMetrics.flopsPct} color="#0969da" />
            {detailNode.children?.length > 0 && (
              <DetailRow label={t('arch.subModules')} value={String(detailNode.children.length)} />
            )}
          </div>
        )}
      </div>

      {/* AI narration panel */}
      {archNarration && (
        <div style={{
          margin: '0 12px 8px', padding: '10px 14px', borderRadius: 6,
          background: archNarration.error ? '#331111' : 'var(--tab-bg, #16162a)',
          border: '1px solid var(--border, #333)',
          fontSize: 12, lineHeight: 1.6,
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
            <span style={{ fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.5, color: archNarration.error ? '#ff6666' : 'var(--accent, #4fc3f7)' }}>
              {archNarration.error ? t('misc.error') : t('arch.sourceAgent')}
            </span>
            {archNarration.model_name && (
              <span style={{ fontSize: 10, color: 'var(--text-secondary, #888)' }}>{archNarration.model_name}</span>
            )}
          </div>
          {archNarration.narration ? (
            <div style={{ color: 'var(--text, #e0e0e0)', whiteSpace: 'pre-wrap' }}>{archNarration.narration}</div>
          ) : archNarration.error ? (
            <div style={{ color: '#ff6666' }}>{archNarration.error}</div>
          ) : (
            <div style={{ color: 'var(--text-secondary, #888)', fontStyle: 'italic' }}>{t('arch.agentRunning')}</div>
          )}
        </div>
      )}

      {/* Error / empty state */}
      {error && (
        <div style={{ padding: 16, textAlign: 'center', color: '#cf222e', fontSize: 12 }}>
          {t('arch.error')}: {error}
        </div>
      )}
      {!result && !error && !loading && (
        <div style={{ padding: 24, textAlign: 'center', color: 'var(--text-secondary, #666)', fontSize: 12 }}>
          {modelEntry ? t('arch.noData') : t('arch.modelEntryMissing')}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// React detail-panel helpers
// ---------------------------------------------------------------------------

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '5px 0', borderBottom: '1px solid #333', fontSize: 11 }}>
      <span style={{ color: 'var(--text-secondary, #888)' }}>{label}</span>
      <span style={{ color: 'var(--text, #e0e0e0)', fontWeight: 600 }}>{value}</span>
    </div>
  )
}

function ProgressBar({ pct, color }: { pct: number; color: string }) {
  return (
    <div style={{ height: 3, borderRadius: 2, background: '#333', margin: '3px 0 6px', overflow: 'hidden' }}>
      <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: 2, transition: 'width .3s' }} />
    </div>
  )
}

// ---------------------------------------------------------------------------
// D3 renderers
// ---------------------------------------------------------------------------

function renderTreemap(
  container: any,
  nodes: TreeNode[],
  w: number,
  h: number,
  colorBy: ColorBy,
  onSelect: (name: string) => void,
) {
  const ch = nodes.map((c, i) => ({
    id: i, name: c.name,
    value: colorBy === 'params' ? (c.params || 1) : (c.flops || 1),
    raw: c,
  }))
  const root = d3.hierarchy({ children: ch }).sum((d: any) => d.value)
  d3.treemap().size([w, h]).padding(3)(root as any)
  const mp = Math.max(1, ...nodes.map(n => colorBy === 'params' ? (n.params || 0) : (n.flops || 0)))

  const g = container.selectAll('g').data(root.leaves()).join('g')
    .attr('transform', (d: any) => `translate(${d.x0},${d.y0})`)

  g.append('rect')
    .attr('width', (d: any) => d.x1 - d.x0)
    .attr('height', (d: any) => d.y1 - d.y0)
    .attr('fill', (d: any) => getColorD3((d.data as any).raw[colorBy === 'params' ? 'params' : 'flops'] || 1, mp, colorBy))
    .attr('rx', 3).attr('opacity', 0.85)
    .attr('stroke', '#fff').attr('stroke-width', 1)
    .style('cursor', 'pointer')
    .on('click', (_ev: any, d: any) => {
      const nd = (d.data as any).raw
      onSelect(nd.name)
    })
    .on('mouseenter', function (this: any, _ev: any, d: any) {
      const nd = (d.data as any).raw
      d3.select(this).attr('stroke', '#4fc3f7').attr('stroke-width', 2)
      const el = d3.select('body').append('div').attr('class', 'arch-tooltip')
      el.style('position', 'fixed').style('padding', '8px 12px').style('background', '#1a1a2e')
        .style('border', '1px solid #333').style('border-radius', '6px').style('font-size', '11px')
        .style('color', '#e0e0e0').style('pointer-events', 'none').style('z-index', '9999')
        .style('box-shadow', '0 3px 12px rgba(0,0,0,.4)').style('line-height', '1.5')
        .html(`<b>${nd.name}</b><br>${nd.type || ''} ${nd.repeat > 1 ? 'x' + nd.repeat : ''}<br>Params: ${fmtNum(nd.params)}<br>FLOPs: ${fmtNum(nd.flops)}`)
    })
    .on('mouseleave', function (this: any) {
      d3.select(this).attr('stroke', '#fff').attr('stroke-width', 1)
      d3.selectAll('.arch-tooltip').remove()
    })

  g.filter((d: any) => (d.x1 - d.x0) > 40 && (d.y1 - d.y0) > 20)
    .append('text')
    .attr('x', 4).attr('y', 14)
    .attr('fill', '#fff').attr('font-size', 9).attr('font-weight', 600)
    .text((d: any) => {
      const n = (d.data as any).raw.name.split('.').pop()
      return n && n.length > 12 ? n.slice(0, 10) + '..' : n
    })
}

function renderBackbone(
  container: any,
  nodes: TreeNode[],
  w: number,
  h: number,
  colorBy: ColorBy,
  onSelect: (name: string) => void,
) {
  const nw = 140, nh = 72, gap = 60, mx = 60, my = 50
  const tw = nodes.length * (nw + gap) - gap + mx * 2
  const th = nh + my * 2 + 30
  container.attr('viewBox', [0, 0, Math.max(tw, w), Math.max(th, h)])
  const gg = container.append('g')
  const mp = Math.max(1, ...nodes.map(n => colorBy === 'params' ? (n.params || 0) : (n.flops || 0)))

  for (let i = 0; i < nodes.length - 1; i++) {
    const x1 = mx + i * (nw + gap) + nw, y = my + nh / 2, x2 = mx + (i + 1) * (nw + gap)
    const tk = Math.max(1, Math.min(5, (nodes[i].params || 0) / mp * 5))
    gg.append('line').attr('x1', x1).attr('y1', y).attr('x2', x2).attr('y2', y)
      .attr('stroke', '#d0d7de').attr('stroke-width', tk).attr('stroke-linecap', 'round')
    gg.append('polygon')
      .attr('points', `${x2},${y} ${x2 - 5},${y - 3} ${x2 - 5},${y + 3}`)
      .attr('fill', '#d0d7de')
  }

  nodes.forEach((nd, i) => {
    const x = mx + i * (nw + gap), y = my
    const pctP = (nd.params || 0) / mp, pctF = (nd.flops || 0) / mp
    const g = gg.append('g').attr('transform', `translate(${x},${y})`).style('cursor', 'pointer')
    g.on('click', () => onSelect(nd.name))
    g.on('mouseenter', function (this: any, _ev: any) {
      d3.select(this).select('rect:first-child').attr('stroke', '#4fc3f7').attr('stroke-width', 2)
      const el = d3.select('body').append('div').attr('class', 'arch-tooltip')
      el.style('position', 'fixed').style('padding', '8px 12px').style('background', '#1a1a2e')
        .style('border', '1px solid #333').style('border-radius', '6px').style('font-size', '11px')
        .style('color', '#e0e0e0').style('pointer-events', 'none').style('z-index', '9999')
        .style('box-shadow', '0 3px 12px rgba(0,0,0,.4)').style('line-height', '1.5')
        .html(`<b>${nd.name}</b><br>${nd.type || ''} ${nd.repeat > 1 ? 'x' + nd.repeat : ''}<br>Params: ${fmtNum(nd.params)}<br>FLOPs: ${fmtNum(nd.flops)}`)
    })
    g.on('mouseleave', function (this: any) {
      d3.select(this).select('rect:first-child').attr('stroke', '#d0d7de').attr('stroke-width', 1)
      d3.selectAll('.arch-tooltip').remove()
    })

    g.append('rect').attr('width', nw).attr('height', nh).attr('rx', 8)
      .attr('fill', '#fff').attr('stroke', '#d0d7de')
    g.append('rect').attr('width', nw).attr('height', 2).attr('rx', 1)
      .attr('fill', getColorD3(colorBy === 'params' ? nd.params : nd.flops || 0, mp, colorBy))
    const lbl = nd.name.split('.').pop() ?? ''
    g.append('text').attr('x', nw / 2).attr('y', 20).attr('text-anchor', 'middle')
      .attr('fill', '#1f2328').attr('font-size', 10).attr('font-weight', 600)
      .text(lbl.length > 16 ? lbl.slice(0, 14) + '..' : lbl)
    g.append('text').attr('x', nw / 2).attr('y', 32).attr('text-anchor', 'middle')
      .attr('fill', '#656d76').attr('font-size', 8)
      .text((nd.type || '') + (nd.repeat > 1 ? ' x' + nd.repeat : ''))
    g.append('rect').attr('x', 8).attr('y', 38).attr('width', nw - 16).attr('height', 3).attr('rx', 1.5).attr('fill', '#f3f4f6')
    g.append('rect').attr('x', 8).attr('y', 38).attr('width', Math.max(2, (nw - 16) * pctP)).attr('height', 3).attr('rx', 1.5).attr('fill', '#1a7f37')
    g.append('rect').attr('x', 8).attr('y', 44).attr('width', nw - 16).attr('height', 3).attr('rx', 1.5).attr('fill', '#f3f4f6')
    g.append('rect').attr('x', 8).attr('y', 44).attr('width', Math.max(2, (nw - 16) * pctF)).attr('height', 3).attr('rx', 1.5).attr('fill', '#0969da')
    g.append('text').attr('x', 6).attr('y', 58).attr('fill', '#656d76').attr('font-size', 8).text(fmtNum(nd.params) + ' p')
    g.append('text').attr('x', nw - 6).attr('y', 58).attr('text-anchor', 'end').attr('fill', '#bf8700').attr('font-size', 7).text(fmtNum(nd.flops))
  })

  container.call(d3.zoom().scaleExtent([0.3, 3]).on('zoom', (e: any) => gg.attr('transform', e.transform)))
}

function getColorD3(v: number, maxVal: number, colorBy: ColorBy): string {
  const r = v / maxVal
  if (colorBy === 'params') return r < 0.25 ? '#1a7f37' : r < 0.5 ? '#8250df' : '#cf222e'
  return r < 0.25 ? '#0969da' : r < 0.5 ? '#8250df' : '#bf8700'
}
