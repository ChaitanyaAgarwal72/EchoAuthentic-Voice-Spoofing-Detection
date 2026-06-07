import { useState, useEffect, useRef, useMemo, useCallback } from 'react'
import './App.css'
import LiveMicDetection from './LiveMicDetection'

// ─── ConfidenceGauge ──────────────────────────────────────────────────────────
function ConfidenceGauge({ score, confidenceBand }) {
  const R = 78
  const CX = 110
  const CY = 108
  const arcLength = Math.PI * R

  const pct = Math.min(100, Math.max(0, score ?? 0))
  const dashOffset = arcLength * (1 - pct / 100)
  const color = pct < 30 ? '#10b981' : pct < 90 ? '#f59e0b' : '#f43f5e'
  const bandClass = pct < 30 ? 'human' : pct < 90 ? 'uncertain' : 'ai'

  return (
    <div className="confidence-gauge">
      <svg viewBox="0 0 220 145" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <filter id="gaugeGlow" x="-40%" y="-40%" width="180%" height="180%">
            <feGaussianBlur stdDeviation="5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        <path d={`M ${CX - R},${CY} A ${R},${R} 0 0,1 ${CX + R},${CY}`} fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="14" strokeLinecap="round" />
        <path
          d={`M ${CX - R},${CY} A ${R},${R} 0 0,1 ${CX + R},${CY}`}
          fill="none" stroke={color} strokeWidth="14" strokeLinecap="round"
          strokeDasharray={arcLength} strokeDashoffset={dashOffset}
          style={{ transition: 'stroke-dashoffset 1.3s cubic-bezier(0.4,0,0.2,1), stroke 0.5s ease', filter: `drop-shadow(0 0 8px ${color}80)` }}
        />
        <text x="110" y="97" textAnchor="middle" fontSize="27" fontWeight="800" fill="#f8fafc" fontFamily="Inter, sans-serif">{pct.toFixed(1)}%</text>
        <text x="110" y="116" textAnchor="middle" fontSize="9" fill="#475569" fontFamily="Inter, sans-serif" letterSpacing="0.14em">AI PROBABILITY</text>
        <text x="26" y={CY + 20} textAnchor="middle" fontSize="8" fill="#334155" fontFamily="Inter, sans-serif">0%</text>
        <text x="194" y={CY + 20} textAnchor="middle" fontSize="8" fill="#334155" fontFamily="Inter, sans-serif">100%</text>
      </svg>
      <div className={`gauge-band gauge-band--${bandClass}`}>{confidenceBand}</div>
    </div>
  )
}

// ─── AudioPlayer ──────────────────────────────────────────────────────────────
function AudioPlayer({ file }) {
  const [audioUrl, setAudioUrl] = useState(null)
  useEffect(() => {
    if (!file) return
    const url = URL.createObjectURL(file)
    setAudioUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])
  if (!audioUrl) return null
  return (
    <div className="panel-block audio-player-wrap">
      <div className="panel-label">
        <span className="panel-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 18v-6a9 9 0 0 1 18 0v6"/><path d="M21 19a2 2 0 0 1-2 2h-1v-6h3v4zM3 19a2 2 0 0 0 2 2h1v-6H3v4z"/></svg></span>
        <span>Audio Preview</span>
        <span className="panel-note">uploaded file</span>
      </div>
      <audio className="audio-player" src={audioUrl} controls preload="metadata" />
    </div>
  )
}

// ─── SpectrogramPanel ─────────────────────────────────────────────────────────
function SpectrogramPanel({ spectrogramB64 }) {
  if (!spectrogramB64) return null
  return (
    <div className="panel-block spectrogram-panel">
      <div className="panel-label">
        <span className="panel-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 20V10M12 20V4M6 20v-6"/></svg></span>
        <span>Mel Spectrogram</span>
        <span className="panel-note">full audio · capped at 120 s</span>
      </div>
      <div className="spectrogram-img-wrap">
        <img src={`data:image/png;base64,${spectrogramB64}`} alt="Mel spectrogram" className="spectrogram-img" />
      </div>
    </div>
  )
}

// ─── DurationWarning ─────────────────────────────────────────────────────────
function DurationWarning({ message }) {
  if (!message) return null
  return (
    <div className="duration-warning" role="alert">
      <span className="duration-warning-icon" aria-hidden="true"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg></span>
      <div className="duration-warning-body">
        <strong>Short Audio Warning</strong>
        <p>{message}</p>
      </div>
    </div>
  )
}

// ─── FrequencyHeatmap ─────────────────────────────────────────────────────────
const MAGMA_STOPS = [
  [0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129],
  [181, 54, 122], [229, 80, 100], [251, 136, 97], [252, 253, 191],
]
function magmaColor(t) {
  const clamped = Math.max(0, Math.min(1, t))
  const idx = clamped * (MAGMA_STOPS.length - 1)
  const lo = Math.floor(idx), hi = Math.min(lo + 1, MAGMA_STOPS.length - 1)
  const f = idx - lo
  return `rgb(${Math.round(MAGMA_STOPS[lo][0] + f * (MAGMA_STOPS[hi][0] - MAGMA_STOPS[lo][0]))},${Math.round(MAGMA_STOPS[lo][1] + f * (MAGMA_STOPS[hi][1] - MAGMA_STOPS[lo][1]))},${Math.round(MAGMA_STOPS[lo][2] + f * (MAGMA_STOPS[hi][2] - MAGMA_STOPS[lo][2]))})`
}

const BAND_LABELS_TOP_TO_BOTTOM = ['Air', 'Brilliance', 'Presence', 'Upper-mid', 'Mid', 'Low-mid', 'Bass', 'Sub-bass']
const BAND_LABELS_LOOKUP = ['Sub-bass', 'Bass', 'Low-mid', 'Mid', 'Upper-mid', 'Presence', 'Brilliance', 'Air']

function FrequencyHeatmap({ profiles, chunkTimeline, selectedChunkIdx, onChunkClick }) {
  const canvasRef = useRef(null)
  const [tooltip, setTooltip] = useState(null)
  const N_BUCKETS = 16

  const drawCanvas = useCallback(() => {
    if (!profiles?.length || !canvasRef.current) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const N = profiles.length
    const W = canvas.width, H = canvas.height
    const cw = W / N, ch = H / N_BUCKETS
    ctx.clearRect(0, 0, W, H)
    for (let c = 0; c < N; c++) {
      for (let b = 0; b < N_BUCKETS; b++) {
        const val = profiles[c][N_BUCKETS - 1 - b]
        ctx.fillStyle = magmaColor(val)
        ctx.fillRect(c * cw, b * ch, Math.max(1, cw - 0.5), Math.max(1, ch - 0.5))
        const aiScore = (chunkTimeline?.[c]?.score ?? 0) / 100
        if (aiScore > 0.3) {
          ctx.fillStyle = `rgba(244,63,94,${Math.min(0.42, (aiScore - 0.3) * 0.6)})`
          ctx.fillRect(c * cw, b * ch, Math.max(1, cw - 0.5), Math.max(1, ch - 0.5))
        }
      }
      if (selectedChunkIdx === c) {
        ctx.strokeStyle = 'rgba(167,139,250,0.95)'
        ctx.lineWidth = 2.5
        ctx.strokeRect(c * cw + 1, 1, cw - 2, H - 2)
      }
    }
  }, [profiles, chunkTimeline, selectedChunkIdx, N_BUCKETS])

  useEffect(() => { drawCanvas() }, [drawCanvas])

  const handleMouseMove = useCallback((e) => {
    if (!profiles?.length || !canvasRef.current) return
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    const cx = (e.clientX - rect.left) * scaleX
    const cy = (e.clientY - rect.top) * scaleY
    const chunkIdx = Math.min(Math.floor(cx / (canvas.width / profiles.length)), profiles.length - 1)
    const bandFlipped = Math.floor(cy / (canvas.height / N_BUCKETS))
    const bandIdx = Math.max(0, Math.min(N_BUCKETS - 1, N_BUCKETS - 1 - bandFlipped))
    const chunk = chunkTimeline?.[chunkIdx]
    setTooltip({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      displayWidth: rect.width,
      chunkIdx,
      bandLabel: BAND_LABELS_LOOKUP[Math.floor(bandIdx / 2)] ?? '—',
      energy: profiles[chunkIdx]?.[bandIdx] ?? 0,
      score: chunk?.score ?? null,
      startSec: chunk?.start_sec ?? 0,
      endSec: chunk?.end_sec ?? 0,
    })
  }, [profiles, chunkTimeline, N_BUCKETS])

  const handleClick = useCallback(() => {
    if (!tooltip) return
    const chunk = chunkTimeline?.[tooltip.chunkIdx]
    if (chunk) onChunkClick?.(chunk, tooltip.chunkIdx)
  }, [tooltip, chunkTimeline, onChunkClick])

  if (!profiles?.length) return null

  return (
    <div className="panel-block heatmap-panel">
      <div className="panel-label">
        <span className="panel-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/></svg></span>
        <span>Frequency Band Heatmap</span>
        <span className="panel-note">spectral energy per chunk · red overlay = elevated AI score · click to play</span>
      </div>
      <div className="heatmap-layout">
        <div className="heatmap-y-axis">
          {BAND_LABELS_TOP_TO_BOTTOM.map(l => <span key={l}>{l}</span>)}
        </div>
        <div className="heatmap-canvas-wrap" onMouseLeave={() => setTooltip(null)}>
          <canvas
            ref={canvasRef}
            width={900} height={144}
            className="heatmap-canvas"
            onMouseMove={handleMouseMove}
            onClick={handleClick}
          />
          {tooltip && (() => {
            const TOOLTIP_W = 170
            const flipLeft = tooltip.displayWidth && tooltip.x > tooltip.displayWidth / 2
            return (
              <div className="heatmap-tooltip" style={{
                left: flipLeft ? tooltip.x - TOOLTIP_W - 8 : tooltip.x + 14,
                top: Math.max(4, tooltip.y - 82),
              }}>
                <span className="ht-time">{tooltip.startSec.toFixed(1)}s – {tooltip.endSec.toFixed(1)}s</span>
                <span className="ht-band">{tooltip.bandLabel}</span>
                <span className="ht-energy">Energy: {(tooltip.energy * 100).toFixed(1)}%</span>
                {tooltip.score != null && (
                  <span className="ht-score" style={{ color: tooltip.score >= 90 ? '#fb7185' : tooltip.score >= 30 ? '#fbbf24' : '#34d399' }}>
                    AI Score: {tooltip.score.toFixed(1)}%
                  </span>
                )}
                <span className="ht-hint">Click to play segment</span>
              </div>
            )
          })()}
        </div>
      </div>
      <div className="heatmap-legend">
        <span className="hl-item">Low</span>
        <div className="heatmap-colorbar" />
        <span className="hl-item">High energy</span>
        <span className="hl-sep">·</span>
        <span className="hl-item hl-item--ai">🔴 Red overlay = high AI score</span>
      </div>
    </div>
  )
}

// ─── ConfidenceWaterfall ──────────────────────────────────────────────────────
function ConfidenceWaterfall({ chunkTimeline, selectedChunkIdx, onChunkClick }) {
  const [hovered, setHovered] = useState(null)
  if (!chunkTimeline?.length) return null

  const SVG_W = 900, SVG_H = 200
  const PAD = { top: 20, right: 52, bottom: 40, left: 48 }
  const chartW = SVG_W - PAD.left - PAD.right
  const chartH = SVG_H - PAD.top - PAD.bottom
  const N = chunkTimeline.length
  const spacing = chartW / N
  const barW = Math.max(2, spacing * 0.72)

  const yScale = (pct) => PAD.top + chartH * (1 - pct / 100)
  const barColor = (s) => s < 30 ? '#10b981' : s < 90 ? '#f59e0b' : '#f43f5e'
  const glowColor = (s) => s < 30 ? 'rgba(16,185,129,0.45)' : s < 90 ? 'rgba(245,158,11,0.45)' : 'rgba(244,63,94,0.55)'

  const tickIndices = N <= 6
    ? chunkTimeline.map((_, i) => i)
    : [0, Math.floor(N * 0.25), Math.floor(N * 0.5), Math.floor(N * 0.75), N - 1]

  return (
    <div className="panel-block waterfall-panel">
      <div className="panel-label">
        <span className="panel-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg></span>
        <span>Confidence Waterfall</span>
        <span className="panel-note">AI probability per segment · {N} chunks · click to play</span>
      </div>
      <div className="waterfall-svg-wrap">
        <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} className="waterfall-svg" onMouseLeave={() => setHovered(null)}>
          {/* Grid lines */}
          {[0, 30, 60, 90, 100].map(pct => {
            const y = yScale(pct)
            const isThresh = pct === 30 || pct === 90
            return (
              <g key={pct}>
                <line x1={PAD.left} x2={SVG_W - PAD.right} y1={y} y2={y}
                  stroke={isThresh ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.04)'}
                  strokeDasharray={isThresh ? '5 5' : undefined}
                  strokeWidth={isThresh ? 1.5 : 1} />
                <text x={PAD.left - 8} y={y + 4} textAnchor="end" fontSize="10"
                  fontFamily="Inter, sans-serif"
                  fill={pct === 30 ? '#34d399' : pct === 90 ? '#fb7185' : '#334155'}>
                  {pct}%
                </text>
              </g>
            )
          })}
          {/* Right-side labels */}
          <text x={SVG_W - PAD.right + 6} y={yScale(15) + 4} fontSize="9" fontFamily="Inter, sans-serif" fill="#34d39988">Human</text>
          <text x={SVG_W - PAD.right + 6} y={yScale(95) + 4} fontSize="9" fontFamily="Inter, sans-serif" fill="#fb718588">AI</text>
          {/* Y label */}
          <text x={14} y={PAD.top + chartH / 2} textAnchor="middle" fontSize="9"
            fontFamily="Inter, sans-serif" fill="#475569"
            transform={`rotate(-90, 14, ${PAD.top + chartH / 2})`}>AI PROB %</text>

          {/* Bars */}
          {chunkTimeline.map((chunk, i) => {
            const x = PAD.left + i * spacing + (spacing - barW) / 2
            const barH = Math.max(2, chartH * (chunk.score / 100))
            const y = PAD.top + chartH - barH
            const isHov = hovered === i, isSel = selectedChunkIdx === i
            const color = barColor(chunk.score)
            return (
              <g key={i} style={{ cursor: 'pointer' }}
                onMouseEnter={() => setHovered(i)}
                onMouseLeave={() => setHovered(null)}
                onClick={() => onChunkClick?.(chunk, i)}>
                {/* Invisible full-height hit area — makes empty space above short bars hoverable */}
                <rect
                  x={PAD.left + i * spacing} y={PAD.top}
                  width={spacing} height={chartH}
                  fill="transparent"
                />
                {(isHov || isSel) && (
                  <rect x={x - 2} y={PAD.top} width={barW + 4} height={chartH}
                    fill="rgba(255,255,255,0.04)" rx="3" />
                )}
                <rect x={x} y={y} width={barW} height={barH} fill={color} rx="2"
                  className="waterfall-bar"
                  style={{
                    animationDelay: `${i * 0.015}s`,
                    filter: (isHov || isSel) ? `drop-shadow(0 0 6px ${glowColor(chunk.score)})` : undefined,
                  }} />
                {isSel && (
                  <rect x={x - 1} y={PAD.top} width={barW + 2} height={chartH}
                    fill="none" stroke="rgba(167,139,250,0.8)" strokeWidth="1.5" rx="3" />
                )}
              </g>
            )
          })}

          {/* Render Tooltip ON TOP of all bars */}
          {hovered !== null && chunkTimeline[hovered] && (() => {
            const i = hovered
            const chunk = chunkTimeline[i]
            const x = PAD.left + i * spacing + (spacing - barW) / 2
            const barH = Math.max(2, chartH * (chunk.score / 100))
            const y = PAD.top + chartH - barH
            const color = barColor(chunk.score)
            
            const ttX = Math.min(x - 20, SVG_W - PAD.right - 78)
            const TT_H = 44
            const rawTtY = y - 52
            const ttY = rawTtY < PAD.top + 2 ? PAD.top + 2 : rawTtY
            
            return (
              <g style={{ pointerEvents: 'none' }}>
                <rect x={ttX} y={ttY} width={76} height={TT_H} rx="8"
                  fill="rgba(10,13,22,0.96)" stroke="rgba(255,255,255,0.12)" strokeWidth="1" />
                <text x={ttX + 38} y={ttY + 19} textAnchor="middle" fontSize="12"
                  fontWeight="700" fontFamily="Inter, sans-serif" fill={color}>
                  {chunk.score.toFixed(1)}%
                </text>
                <text x={ttX + 38} y={ttY + 36} textAnchor="middle" fontSize="9"
                  fontFamily="Inter, sans-serif" fill="#64748b">
                  {chunk.start_sec.toFixed(1)}–{chunk.end_sec.toFixed(1)}s
                </text>
              </g>
            )
          })()}

          {/* X-axis time labels */}
          {tickIndices.map(i => {
            const chunk = chunkTimeline[i]
            if (!chunk) return null
            return (
              <text key={i} x={PAD.left + i * spacing + spacing / 2} y={SVG_H - 8}
                textAnchor="middle" fontSize="10" fontFamily="Inter, sans-serif" fill="#475569">
                {chunk.start_sec.toFixed(0)}s
              </text>
            )
          })}
        </svg>
      </div>
    </div>
  )
}

// ─── ChunkPlayer ──────────────────────────────────────────────────────────────
function ChunkPlayer({ chunk, audioFile, apiBaseUrl, mode, onClose }) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [error, setError] = useState('')
  const audioCtxRef = useRef(null)
  const sourceRef = useRef(null)
  const audioElRef = useRef(null)

  const stop = useCallback(() => {
    try { sourceRef.current?.stop() } catch { }
    sourceRef.current = null
    if (audioElRef.current) { audioElRef.current.pause(); audioElRef.current.src = ''; audioElRef.current = null }
    setIsPlaying(false)
  }, [])

  useEffect(() => () => stop(), [stop])
  useEffect(() => { stop(); setError('') }, [chunk?.start_sec, chunk?.end_sec, stop])

  const play = useCallback(async () => {
    stop(); setError(''); setIsPlaying(true)
    try {
      if (mode === 'upload' && audioFile) {
        const ctx = audioCtxRef.current || new AudioContext()
        if (ctx.state === 'suspended') await ctx.resume()
        audioCtxRef.current = ctx
        const audioBuf = await ctx.decodeAudioData(await audioFile.arrayBuffer())
        const src = ctx.createBufferSource()
        src.buffer = audioBuf
        src.connect(ctx.destination)
        sourceRef.current = src
        src.onended = () => setIsPlaying(false)
        src.start(0, chunk.start_sec, chunk.end_sec - chunk.start_sec)
      } else {
        const resp = await fetch(`${apiBaseUrl}/audio/segment/?start=${chunk.start_sec}&end=${chunk.end_sec}`)
        if (!resp.ok) throw new Error('Failed to fetch segment from server.')
        const objUrl = URL.createObjectURL(await resp.blob())
        const audio = new Audio(objUrl)
        audioElRef.current = audio
        audio.onended = () => { setIsPlaying(false); URL.revokeObjectURL(objUrl) }
        audio.onerror = () => { setIsPlaying(false); setError('Playback failed.') }
        audio.play()
      }
    } catch (err) { setError(err.message || 'Playback failed.'); setIsPlaying(false) }
  }, [chunk, audioFile, apiBaseUrl, mode, stop])

  const scoreColor = chunk.score >= 90 ? '#fb7185' : chunk.score >= 30 ? '#fbbf24' : '#34d399'

  return (
    <div className="chunk-player">
      <div className="cp-header">
        <span className="cp-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/></svg></span>
        <span className="cp-title">Segment Player</span>
        <div className="cp-badges">
          <span className="cp-badge cp-badge--time">{chunk.start_sec.toFixed(2)}s – {chunk.end_sec.toFixed(2)}s</span>
          <span className="cp-badge" style={{ color: scoreColor, borderColor: `${scoreColor}50`, background: `${scoreColor}15` }}>
            {chunk.score.toFixed(1)}% AI
          </span>
        </div>
        <button className="cp-close" onClick={onClose} aria-label="Close player">✕</button>
      </div>
      <div className="cp-controls">
        <button className={`cp-play-btn${isPlaying ? ' cp-play-btn--stop' : ''}`} onClick={isPlaying ? stop : play}>
          {isPlaying ? '⏹ Stop' : '▶ Play Segment'}
        </button>
        {isPlaying && (
          <span className="cp-playing-indicator" aria-label="Playing">
            <span className="cp-wave" /><span className="cp-wave" /><span className="cp-wave" /><span className="cp-wave" />
          </span>
        )}
        {error && <span className="cp-error">{error}</span>}
      </div>
    </div>
  )
}

// ─── SegmentTimeline ──────────────────────────────────────────────────────────
function SegmentTimeline({ timeline, onChunkClick, selectedChunkIdx }) {
  const [hovered, setHovered] = useState(null)
  if (!timeline || timeline.length === 0) return null

  const aiCount = timeline.filter(c => c.score >= 90).length
  const humanCount = timeline.filter(c => c.score < 30).length
  const uncCount = timeline.length - aiCount - humanCount
  const segColor = (score) => score < 30 ? 'human' : score < 90 ? 'uncertain' : 'ai'

  return (
    <div className="panel-block segment-timeline">
      <div className="panel-label">
        <span className="panel-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg></span>
        <span>Segment Strip</span>
        <span className="panel-note">VAD-stripped speech · {timeline.length} chunks · click any segment to play</span>
      </div>
      <div className="timeline-track-wrap">
        <div className="timeline-track">
          {timeline.map((chunk, i) => (
            <div
              key={i}
              className={`timeline-seg timeline-seg--${segColor(chunk.score)}${selectedChunkIdx === i ? ' timeline-seg--selected' : ''}`}
              style={{ flex: 1 }}
              onMouseEnter={() => setHovered(i)}
              onMouseLeave={() => setHovered(null)}
              onClick={() => onChunkClick?.(chunk, i)}
            >
              {hovered === i && (
                <div className="seg-tooltip">
                  <span className="seg-tt-time">{chunk.start_sec.toFixed(1)}s – {chunk.end_sec.toFixed(1)}s</span>
                  <span className="seg-tt-score" style={{ color: chunk.score >= 90 ? '#fb7185' : chunk.score >= 30 ? '#fbbf24' : '#34d399' }}>
                    {chunk.score.toFixed(1)}% AI
                  </span>
                  <span className="seg-tt-hint">Click to play</span>
                </div>
              )}
            </div>
          ))}
        </div>
        <div className="timeline-scale">
          <span>0 s</span>
          {timeline[Math.floor(timeline.length / 2)] && (
            <span>{((timeline[Math.floor(timeline.length / 2)].start_sec + timeline[Math.floor(timeline.length / 2)].end_sec) / 2).toFixed(1)} s</span>
          )}
          <span>{timeline[timeline.length - 1]?.end_sec.toFixed(1)} s</span>
        </div>
      </div>
      <div className="timeline-legend">
        <span className="leg leg--human">● Human <em>({humanCount})</em></span>
        <span className="leg leg--uncertain">● Uncertain <em>({uncCount})</em></span>
        <span className="leg leg--ai">● AI-Generated <em>({aiCount})</em></span>
        <span className="leg leg--ratio">{aiCount}/{timeline.length} chunks high-confidence AI ({Math.round((aiCount / timeline.length) * 100)}%)</span>
      </div>
    </div>
  )
}

// ─── DisclaimerCallout ────────────────────────────────────────────────────────
function DisclaimerCallout() {
  return (
    <div className="disclaimer-callout">
      <span className="disclaimer-icon" aria-hidden="true"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg></span>
      <div className="disclaimer-body">
        <strong>Model Reliability Notice</strong>
        <p>
          This model was trained on the ASVspoof 2019 dataset and may not detect the newest AI voice generators (e.g., ElevenLabs, Suno). Please use as one of many signals, not as conclusive proof.
        </p>
      </div>
    </div>
  )
}

// ─── ProgressStepper ─────────────────────────────────────────────────────────
const PIPELINE_STAGES = [
  { key: 'downloading', icon: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>, label: 'Download' },
  { key: 'loading', icon: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>, label: 'Load Audio' },
  { key: 'vad', icon: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg>, label: 'VAD' },
  { key: 'inference', icon: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="2" y="2" width="20" height="8" rx="2" ry="2"/><rect x="2" y="14" width="20" height="8" rx="2" ry="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg>, label: 'AI Analysis' },
  { key: 'spectrogram', icon: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 20V10M12 20V4M6 20v-6"/></svg>, label: 'Spectrogram' },
  { key: 'done', icon: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="20 6 9 17 4 12"/></svg>, label: 'Done' },
]

function ProgressStepper({ progress }) {
  if (!progress) return null
  const currentIdx = PIPELINE_STAGES.findIndex(s => s.key === progress.stage)
  return (
    <div className="pipeline-progress">
      <div className="pipeline-steps">
        {PIPELINE_STAGES.map((stage, idx) => {
          const status = idx < currentIdx ? 'done' : idx === currentIdx ? 'active' : 'pending'
          return (
            <div key={stage.key} className={`pipeline-step pipeline-step--${status}`}>
              <div className="pipeline-step-dot">
                {status === 'done' ? <span className="pipeline-step-check">✓</span> : <span className="pipeline-step-icon">{stage.icon}</span>}
                {status === 'active' && <span className="pipeline-step-pulse" />}
              </div>
              <span className="pipeline-step-label">{stage.label}</span>
              {idx < PIPELINE_STAGES.length - 1 && (
                <div className={`pipeline-connector pipeline-connector--${status === 'done' ? 'done' : 'pending'}`} />
              )}
            </div>
          )
        })}
      </div>
      <div className="pipeline-bar-wrap">
        <div className="pipeline-bar-fill" style={{ width: `${progress.pct ?? 0}%` }} />
      </div>
      <p className="pipeline-label">
        <span className="pipeline-spinner" aria-hidden="true" />
        {progress.label}
      </p>
    </div>
  )
}

// ─── SkeletonLoader ───────────────────────────────────────────────────────────
function SkeletonLoader() {
  return (
    <div className="skeleton-wrap" aria-label="Loading results…">
      <div className="skel skel--banner" />
      <div className="skel-gauge-row">
        <div className="skel skel--gauge" />
        <div className="skel-stats">
          <div className="skel skel--stat" /><div className="skel skel--stat" />
          <div className="skel skel--stat" /><div className="skel skel--stat" />
        </div>
      </div>
      <div className="skel skel--spec" />
      <div className="skel skel--timeline" />
    </div>
  )
}

// ─── Main App ─────────────────────────────────────────────────────────────────
function App() {
  const [mode, setMode] = useState('youtube')
  const [youtubeUrl, setYoutubeUrl] = useState('')
  const [audioFile, setAudioFile] = useState(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [progress, setProgress] = useState(null)
  const [selectedChunk, setSelectedChunk] = useState(null)
  const [selectedChunkIdx, setSelectedChunkIdx] = useState(null)
  const audioFileRef = useRef(null)

  const apiBaseUrl = useMemo(() => import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000', [])

  const predictionLabel = result?.prediction || result?.binary_prediction
  const isAi = predictionLabel === 'AI-Generated'
  const isHuman = predictionLabel === 'Human Voice'
  const bannerVariant = isAi ? 'ai' : isHuman ? 'human' : 'uncertain'

  const handleModeChange = (next) => {
    setMode(next); setResult(null); setError(''); setProgress(null)
    setSelectedChunk(null); setSelectedChunkIdx(null)
  }

  const handleFileChange = (e) => {
    const f = e.target.files?.[0] || null
    setAudioFile(f); audioFileRef.current = f
  }

  const handleChunkClick = useCallback((chunk, idx) => {
    if (selectedChunkIdx === idx) { setSelectedChunk(null); setSelectedChunkIdx(null) }
    else { setSelectedChunk(chunk); setSelectedChunkIdx(idx) }
  }, [selectedChunkIdx])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(''); setResult(null); setProgress(null)
    setSelectedChunk(null); setSelectedChunkIdx(null)

    try {
      setIsSubmitting(true)
      if (mode === 'youtube') {
        if (!youtubeUrl.trim()) throw new Error('Please paste a YouTube link first.')
        const response = await fetch(`${apiBaseUrl}/predict/youtube/stream/`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: youtubeUrl.trim() }),
        })
        if (!response.ok) { const d = await response.json().catch(() => ({})); throw new Error(d?.detail || 'Request failed.') }

        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = '', processingError = null

        outer: while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n'); buffer = lines.pop() ?? ''
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue
            let evt; try { evt = JSON.parse(line.slice(6)) } catch { continue }
            if (evt.stage === 'error') { processingError = evt.message || 'Processing failed.'; reader.cancel(); break outer }
            else if (evt.stage === 'done') { setResult(evt.result); setProgress(null); break outer }
            else { setProgress(evt) }
          }
        }
        if (processingError) throw new Error(processingError)

      } else {
        if (!audioFile) throw new Error('Please choose an audio file first.')
        const fd = new FormData(); fd.append('file', audioFile)
        const response = await fetch(`${apiBaseUrl}/predict/`, { method: 'POST', body: fd })
        const data = await response.json()
        if (!response.ok) throw new Error(data?.detail || 'Prediction failed.')
        setResult(data)
      }
    } catch (err) {
      setError(err.message || 'Something went wrong.')
    } finally {
      setIsSubmitting(false); setProgress(null)
    }
  }

  const avgScore = result?.average_ai_probability_score ?? 0
  const confidenceBand = result?.confidence_band ?? ''

  return (
    <main className="dashboard-shell">
      <section className="dashboard-card">

        {/* ── Masthead ── */}
        <div className="hero-section">
          <div className="hero-copy">
            <span className="eyebrow">EchoAuthentic</span>
            <h1>Audio Deepfake Detection</h1>
            <p>
              Upload any audio or YouTube link – or test live with your microphone. Get an instant AI probability score, an interactive spectrogram heatmap, and a per‑segment playback timeline.
            </p>
          </div>
          
          <div className="hero-visual">
            <div className="hv-waveform">
              {[20, 40, 75, 45, 30, 65, 90, 70, 40, 35, 60, 85, 100, 75, 45, 30, 55, 80, 40, 25].map((h, i) => (
                <div key={i} className="hv-bar" style={{ height: `${h}%`, animationDelay: `${i * 0.06}s` }} />
              ))}
            </div>
          </div>
        </div>

        {/* ── Input form ── */}
        <form className="predict-form" onSubmit={handleSubmit}>
          <div className="mode-switch" role="tablist" aria-label="Input source">
            <button id="tab-youtube" type="button" role="tab" aria-selected={mode === 'youtube'}
              className={mode === 'youtube' ? 'mode-button active' : 'mode-button'}
              onClick={() => handleModeChange('youtube')}>YouTube</button>
            <button id="tab-upload" type="button" role="tab" aria-selected={mode === 'upload'}
              className={mode === 'upload' ? 'mode-button active' : 'mode-button'}
              onClick={() => handleModeChange('upload')}>Upload</button>
            <button id="tab-live" type="button" role="tab" aria-selected={mode === 'live'}
              className={mode === 'live' ? 'mode-button active' : 'mode-button'}
              onClick={() => handleModeChange('live')}>Live Mic</button>
          </div>

          {mode === 'youtube' ? (
            <label className="field" htmlFor="youtube-url">
              <span>YouTube URL</span>
              <input id="youtube-url" type="url" placeholder="https://www.youtube.com/watch?v=..."
                value={youtubeUrl} onChange={(e) => setYoutubeUrl(e.target.value)} />
            </label>
          ) : mode === 'upload' ? (
            <label className="field" htmlFor="audio-file">
              <span>Supported formats: .wav, .mp3, .flac (max 50MB)</span>
              <input id="audio-file" type="file" accept=".wav,.mp3,.flac,audio/*" onChange={handleFileChange} />
            </label>
          ) : null}

          {mode !== 'live' && (
            <button id="predict-btn" type="submit" className="predict-button" disabled={isSubmitting}>
              {isSubmitting ? <span className="btn-inner"><span className="spinner" aria-hidden="true" />Analysing…</span> : 'Run Detection'}
            </button>
          )}
        </form>

        {/* ── Live Mic mode ── */}
        {mode === 'live' && <LiveMicDetection />}

        {/* ── Error ── */}
        <div className={`error-box ${mode !== 'live' && error ? 'show' : ''}`} role="alert" aria-hidden={!(mode !== 'live' && error)}>
          {error}
        </div>

        {/* ── Loading states ── */}
        {isSubmitting && mode === 'youtube' && <ProgressStepper progress={progress} />}
        {isSubmitting && mode !== 'youtube' && <SkeletonLoader />}

        {/* ── Results ── */}
        {result && !isSubmitting && (
          <section className="results-area" aria-live="polite" aria-label="Detection results">

            {/* Prediction banner */}
            <div className={`prediction-banner prediction-banner--${bannerVariant}`}>
              <span className="pred-icon" aria-hidden="true">
                {isAi ? <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="11" width="18" height="10" rx="2" /><circle cx="12" cy="5" r="2" /><path d="M12 7v4" /><line x1="8" y1="16" x2="8.01" y2="16"/><line x1="16" y1="16" x2="16.01" y2="16"/></svg> : isHuman ? <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 6L9 17l-5-5"/></svg> : <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>}
              </span>
              <div className="pred-text">
                <div className="pred-label">Prediction</div>
                <div className="pred-value">{predictionLabel}</div>
                {result.filename && (
                  <div className="pred-file" title={result.filename}>
                    <span className="pred-file-icon"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg></span>
                    <span className="pred-file-name">{result.filename}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Duration warning (soft alert for very short audio) */}
            <DurationWarning message={result.duration_warning} />

            {/* Gauge + stats */}
            <div className="gauge-stats-row">
              <ConfidenceGauge score={avgScore} confidenceBand={confidenceBand} />
              <div className="stats-grid">
                <div className="stat-card">
                  <span>AI Probability Score</span>
                  <strong style={{ color: avgScore < 30 ? '#34d399' : avgScore < 90 ? '#fbbf24' : '#fb7185' }}>
                    {avgScore.toFixed(2)}%
                  </strong>
                </div>
                <div className="stat-card">
                  <span>AI Chunk Ratio</span>
                  <strong style={{ color: (result.chunk_high_score_ratio ?? 0) < 0.3 ? '#34d399' : (result.chunk_high_score_ratio ?? 0) < 0.66 ? '#fbbf24' : '#fb7185' }}>
                    {result.chunk_high_score_ratio != null ? `${Math.round(result.chunk_high_score_ratio * 100)}%` : '—'}
                  </strong>
                </div>
                <div className="stat-card">
                  <span>Chunks Analysed</span>
                  <strong>{result.chunk_count}</strong>
                </div>
                <div className="stat-card">
                  <span>High-Conf AI Chunks</span>
                  <strong>
                    {result.chunk_timeline ? `${result.chunk_timeline.filter(c => c.score >= 90).length} / ${result.chunk_count}` : '—'}
                  </strong>
                </div>
                {result.vad_summary?.vad_note && (
                  <div className="stat-card stat-card--wide">
                    <span>Voice Activity Detection</span>
                    <strong>{result.vad_summary.vad_note}</strong>
                  </div>
                )}
              </div>
            </div>

            {/* ── Explainability ── */}
            <ConfidenceWaterfall
              chunkTimeline={result.chunk_timeline}
              selectedChunkIdx={selectedChunkIdx}
              onChunkClick={handleChunkClick}
            />

            <FrequencyHeatmap
              profiles={result.chunk_mel_profiles}
              chunkTimeline={result.chunk_timeline}
              selectedChunkIdx={selectedChunkIdx}
              onChunkClick={handleChunkClick}
            />

            <SegmentTimeline
              timeline={result.chunk_timeline}
              onChunkClick={handleChunkClick}
              selectedChunkIdx={selectedChunkIdx}
            />

            {/* Chunk player — appears when a segment is clicked */}
            {selectedChunk && (
              <ChunkPlayer
                chunk={selectedChunk}
                audioFile={audioFileRef.current}
                apiBaseUrl={apiBaseUrl}
                mode={mode}
                onClose={() => { setSelectedChunk(null); setSelectedChunkIdx(null) }}
              />
            )}

            {/* Audio player (upload mode only) */}
            {mode === 'upload' && <AudioPlayer file={audioFileRef.current} />}

            {/* Mel spectrogram */}
            <SpectrogramPanel spectrogramB64={result.spectrogram_b64} />

            {/* Disclaimer */}
            <DisclaimerCallout />
          </section>
        )}
      </section>
    </main>
  )
}

export default App
