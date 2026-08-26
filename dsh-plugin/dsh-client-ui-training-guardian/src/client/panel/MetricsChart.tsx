import { useMemo } from 'react'
import * as d3 from 'd3'

interface MetricsChartProps {
  /** Array of metrics data points, each must have at least a timestamp or step. */
  data: Array<Record<string, unknown>>
  /** Which field to plot on the left Y-axis (default: 'loss'). */
  leftField?: string
  /** Which field to plot on the right Y-axis (default: 'accuracy'). */
  rightField?: string
  /** X-axis field (default: 'step', falls back to array index). */
  xField?: string
  width?: number
  height?: number
}

const MARGIN = { top: 12, right: 48, bottom: 28, left: 48 }

/**
 * Lightweight SVG line chart for training metrics trends.
 * Uses d3 for scale/path computation, renders plain SVG (no external DOM lib).
 * Left Y-axis = loss (lower is better, blue), Right Y-axis = accuracy (higher is better, green).
 */
export function MetricsChart({
  data,
  leftField = 'loss',
  rightField = 'accuracy',
  xField = 'step',
  width = 460,
  height = 200,
}: MetricsChartProps) {
  const chartW = width - MARGIN.left - MARGIN.right
  const chartH = height - MARGIN.top - MARGIN.bottom

  const { leftPath, rightPath, xTicks, yLeftTicks, yRightTicks, leftColor, rightColor } = useMemo(() => {
    if (data.length === 0) {
      return { leftPath: '', rightPath: '', xTicks: [] as { x: number; label: string }[], yLeftTicks: [] as { y: number; label: string }[], yRightTicks: [] as { y: number; label: string }[], leftColor: '#4fc3f7', rightColor: '#66bb6a' }
    }

    // Extract numeric series
    const leftSeries: { x: number; y: number }[] = []
    const rightSeries: { x: number; y: number }[] = []
    data.forEach((d, i) => {
      const xVal = d[xField]
      const x = typeof xVal === 'number' ? xVal : i
      const lVal = d[leftField]
      const rVal = d[rightField]
      if (typeof lVal === 'number' && isFinite(lVal)) leftSeries.push({ x, y: lVal })
      if (typeof rVal === 'number' && isFinite(rVal)) rightSeries.push({ x, y: rVal })
    })

    // X scale: union of both series domains
    const allX = [...leftSeries.map(p => p.x), ...rightSeries.map(p => p.x)]
    const xMin = Math.min(...allX)
    const xMax = Math.max(...allX)
    const xScale = d3.scaleLinear().domain([xMin, xMax || 1]).range([0, chartW])

    // Left Y scale (loss)
    const leftYVals = leftSeries.map(p => p.y)
    const leftYMin = leftYVals.length ? Math.min(...leftYVals) : 0
    const leftYMax = leftYVals.length ? Math.max(...leftYVals) : 1
    const leftPad = (leftYMax - leftYMin) * 0.1 || 0.1
    const yLeftScale = d3.scaleLinear().domain([leftYMin - leftPad, leftYMax + leftPad]).range([chartH, 0])

    // Right Y scale (accuracy)
    const rightYVals = rightSeries.map(p => p.y)
    const rightYMin = rightYVals.length ? Math.min(...rightYVals) : 0
    const rightYMax = rightYVals.length ? Math.max(...rightYVals) : 1
    const rightPad = (rightYMax - rightYMin) * 0.1 || 0.1
    const yRightScale = d3.scaleLinear().domain([rightYMin - rightPad, rightYMax + rightPad]).range([chartH, 0])

    // Line generators
    const leftLine = d3.line<{ x: number; y: number }>()
      .x(d => xScale(d.x))
      .y(d => yLeftScale(d.y))
    const rightLine = d3.line<{ x: number; y: number }>()
      .x(d => xScale(d.x))
      .y(d => yRightScale(d.y))

    const lp = leftSeries.length > 1 ? leftLine(leftSeries) || '' : ''
    const rp = rightSeries.length > 1 ? rightLine(rightSeries) || '' : ''

    // X ticks (5 evenly spaced)
    const xTickVals = d3.ticks(xMin, xMax, 5)
    const xTicks = xTickVals.map(v => ({ x: xScale(v), label: String(v) }))

    // Y ticks (left, 4 ticks)
    const yLeftTickVals = d3.ticks(leftYMin - leftPad, leftYMax + leftPad, 4)
    const yLeftTicks = yLeftTickVals.map(v => ({ y: yLeftScale(v), label: v.toFixed(3) }))

    // Y ticks (right, 4 ticks)
    const yRightTickVals = d3.ticks(rightYMin - rightPad, rightYMax + rightPad, 4)
    const yRightTicks = yRightTickVals.map(v => ({ y: yRightScale(v), label: v.toFixed(3) }))

    return { leftPath: lp, rightPath: rp, xTicks, yLeftTicks, yRightTicks, leftColor: '#4fc3f7', rightColor: '#66bb6a' }
  }, [data, leftField, rightField, xField, chartW, chartH])

  if (data.length === 0) {
    return (
      <div style={{ textAlign: 'center', padding: '16px 0', color: 'var(--text-secondary, #888)', fontSize: 12 }}>
        No metrics data
      </div>
    )
  }

  return (
    <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', maxWidth: width, display: 'block' }}>
      <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
        {/* Grid lines */}
        {yLeftTicks.map((t, i) => (
          <line key={`gl${i}`} x1={0} x2={chartW} y1={t.y} y2={t.y}
            stroke="var(--border, #333)" strokeWidth={0.5} strokeDasharray="2,3" />
        ))}

        {/* Left Y-axis labels (loss) */}
        {yLeftTicks.map((t, i) => (
          <text key={`yl${i}`} x={-6} y={t.y} textAnchor="end" dominantBaseline="middle"
            fontSize={9} fill={leftColor}>{t.label}</text>
        ))}

        {/* Right Y-axis labels (accuracy) */}
        {yRightTicks.map((t, i) => (
          <text key={`yr${i}`} x={chartW + 6} y={t.y} textAnchor="start" dominantBaseline="middle"
            fontSize={9} fill={rightColor}>{t.label}</text>
        ))}

        {/* X-axis labels */}
        {xTicks.map((t, i) => (
          <text key={`xt${i}`} x={t.x} y={chartH + 16} textAnchor="middle"
            fontSize={9} fill="var(--text-secondary, #888)">{t.label}</text>
        ))}

        {/* Loss line */}
        {leftPath && (
          <path d={leftPath} fill="none" stroke={leftColor} strokeWidth={1.5} />
        )}

        {/* Accuracy line */}
        {rightPath && (
          <path d={rightPath} fill="none" stroke={rightColor} strokeWidth={1.5} />
        )}

        {/* Legend */}
        <g transform={`translate(${chartW / 2 - 60}, -4)`}>
          <line x1={0} x2={14} y1={0} y2={0} stroke={leftColor} strokeWidth={2} />
          <text x={18} y={0} fontSize={9} fill={leftColor} dominantBaseline="middle">{leftField}</text>
          <line x1={70} x2={84} y1={0} y2={0} stroke={rightColor} strokeWidth={2} />
          <text x={88} y={0} fontSize={9} fill={rightColor} dominantBaseline="middle">{rightField}</text>
        </g>
      </g>
    </svg>
  )
}
