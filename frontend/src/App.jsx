import { useState, useEffect, useRef, useMemo } from 'react'
import './App.css'

// ─── ConfidenceGauge ──────────────────────────────────────────────────────────
// SVG semicircle gauge using stroke-dasharray/dashoffset for smooth animation.
// Arc goes left→right through the top (0% = left / 9-o'clock, 100% = right / 3-o'clock).
function ConfidenceGauge({ score, confidenceBand }) {
  const R = 78
  const CX = 110
  const CY = 108
  const arcLength = Math.PI * R // ≈ 245

  const pct = Math.min(100, Math.max(0, score ?? 0))
  // dashOffset shrinks the visible portion: 0 = full arc shown, arcLength = nothing shown
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
        {/* Track */}
        <path
          d={`M ${CX - R},${CY} A ${R},${R} 0 0,1 ${CX + R},${CY}`}
          fill="none"
          stroke="rgba(255,255,255,0.07)"
          strokeWidth="14"
          strokeLinecap="round"
        />
        {/* Score arc — sweep=1 (clockwise in SVG = upper arc left→right) */}
        <path
          d={`M ${CX - R},${CY} A ${R},${R} 0 0,1 ${CX + R},${CY}`}
          fill="none"
          stroke={color}
          strokeWidth="14"
          strokeLinecap="round"
          strokeDasharray={arcLength}
          strokeDashoffset={dashOffset}
          style={{
            transition: 'stroke-dashoffset 1.3s cubic-bezier(0.4,0,0.2,1), stroke 0.5s ease',
            filter: `drop-shadow(0 0 8px ${color}80)`,
          }}
        />
        {/* Score text */}
        <text
          x="110" y="97"
          textAnchor="middle"
          fontSize="27"
          fontWeight="800"
          fill="#f8fafc"
          fontFamily="Inter, sans-serif"
        >
          {pct.toFixed(1)}%
        </text>
        <text
          x="110" y="116"
          textAnchor="middle"
          fontSize="9"
          fill="#475569"
          fontFamily="Inter, sans-serif"
          letterSpacing="0.14em"
        >
          AI PROBABILITY
        </text>
        {/* Scale labels */}
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
        <span className="panel-icon">🎧</span>
        <span>Audio Preview</span>
        <span className="panel-note">uploaded file</span>
      </div>
      <audio
        className="audio-player"
        src={audioUrl}
        controls
        preload="metadata"
      />
    </div>
  )
}

// ─── SpectrogramPanel ─────────────────────────────────────────────────────────
function SpectrogramPanel({ spectrogramB64 }) {
  if (!spectrogramB64) return null

  return (
    <div className="panel-block spectrogram-panel">
      <div className="panel-label">
        <span className="panel-icon">📊</span>
        <span>Mel Spectrogram</span>
        <span className="panel-note">full audio · capped at 120 s</span>
      </div>
      <div className="spectrogram-img-wrap">
        <img
          src={`data:image/png;base64,${spectrogramB64}`}
          alt="Mel spectrogram of the analysed audio"
          className="spectrogram-img"
        />
      </div>
    </div>
  )
}

// ─── SegmentTimeline ──────────────────────────────────────────────────────────
function SegmentTimeline({ timeline }) {
  const [hovered, setHovered] = useState(null)

  if (!timeline || timeline.length === 0) return null

  const aiCount    = timeline.filter(c => c.score >= 90).length
  const humanCount = timeline.filter(c => c.score < 30).length
  const uncCount   = timeline.length - aiCount - humanCount

  const segColor = (score) =>
    score < 30 ? 'human' : score < 90 ? 'uncertain' : 'ai'

  return (
    <div className="panel-block segment-timeline">
      <div className="panel-label">
        <span className="panel-icon">⏱</span>
        <span>Segment Analysis</span>
        <span className="panel-note">VAD-stripped speech audio · {timeline.length} chunks · chunks may overlap</span>
      </div>

      <div className="timeline-track-wrap">
        <div className="timeline-track">
          {timeline.map((chunk, i) => (
            <div
              key={i}
              className={`timeline-seg timeline-seg--${segColor(chunk.score)}`}
              style={{ flex: 1 }}
              onMouseEnter={() => setHovered(i)}
              onMouseLeave={() => setHovered(null)}
            >
              {hovered === i && (
                <div className="seg-tooltip">
                  <span className="seg-tt-time">
                    {chunk.start_sec.toFixed(1)}s – {chunk.end_sec.toFixed(1)}s
                  </span>
                  <span className="seg-tt-score"
                    style={{ color: chunk.score >= 90 ? '#fb7185' : chunk.score >= 30 ? '#fbbf24' : '#34d399' }}
                  >
                    {chunk.score.toFixed(1)}% AI
                  </span>
                </div>
              )}
            </div>
          ))}
        </div>
        {/* Time scale */}
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
        <span className="leg leg--ratio">
          {aiCount}/{timeline.length} chunks high-confidence AI ({Math.round((aiCount / timeline.length) * 100)}%)
        </span>
      </div>
    </div>
  )
}

// ─── DisclaimerCallout ────────────────────────────────────────────────────────
function DisclaimerCallout() {
  return (
    <div className="disclaimer-callout">
      <span className="disclaimer-icon" aria-hidden="true">⚠️</span>
      <div className="disclaimer-body">
        <strong>Model Reliability Notice</strong>
        <p>
          This model was trained on the <strong>ASVspoof 2019</strong> dataset, which predates modern AI voice
          generators (ElevenLabs, Suno, Udio, etc.). Scores in the <strong>30–90%</strong> range are
          classified as <em>"Unverifiable / Degraded Audio"</em> and should not be treated as conclusive.
          Always corroborate results with additional signals.
        </p>
      </div>
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
          <div className="skel skel--stat" />
          <div className="skel skel--stat" />
          <div className="skel skel--stat" />
          <div className="skel skel--stat" />
        </div>
      </div>
      <div className="skel skel--spec" />
      <div className="skel skel--timeline" />
    </div>
  )
}

// ─── Main App ─────────────────────────────────────────────────────────────────
function App() {
  const [mode, setMode]           = useState('youtube')
  const [youtubeUrl, setYoutubeUrl] = useState('')
  const [audioFile, setAudioFile] = useState(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [result, setResult]       = useState(null)
  const [error, setError]         = useState('')
  const audioFileRef              = useRef(null)

  const apiBaseUrl = useMemo(
    () => import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
    []
  )

  const predictionLabel = result?.prediction || result?.binary_prediction
  const isAi            = predictionLabel === 'AI-Generated'
  const isHuman         = predictionLabel === 'Human Voice'
  const bannerVariant   = isAi ? 'ai' : isHuman ? 'human' : 'uncertain'

  const handleModeChange = (next) => {
    setMode(next)
    setResult(null)
    setError('')
  }

  const handleFileChange = (e) => {
    const f = e.target.files?.[0] || null
    setAudioFile(f)
    audioFileRef.current = f
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setResult(null)

    try {
      setIsSubmitting(true)
      let response

      if (mode === 'youtube') {
        if (!youtubeUrl.trim()) throw new Error('Please paste a YouTube link first.')
        response = await fetch(`${apiBaseUrl}/predict/youtube/`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: youtubeUrl.trim() }),
        })
      } else {
        if (!audioFile) throw new Error('Please choose an audio file first.')
        const fd = new FormData()
        fd.append('file', audioFile)
        response = await fetch(`${apiBaseUrl}/predict/`, { method: 'POST', body: fd })
      }

      const data = await response.json()
      if (!response.ok) throw new Error(data?.detail || 'Prediction failed.')
      setResult(data)
    } catch (err) {
      setError(err.message || 'Something went wrong.')
    } finally {
      setIsSubmitting(false)
    }
  }

  const avgScore      = result?.average_ai_probability_score ?? 0
  const confidenceBand = result?.confidence_band ?? ''

  return (
    <main className="dashboard-shell">
      <section className="dashboard-card">

        {/* ── Masthead ── */}
        <div className="hero-copy">
          <span className="eyebrow">EchoAuthentic</span>
          <h1>Audio Deepfake Detection</h1>
          <p>
            Analyse a YouTube link or upload a file to detect AI-generated speech.
            Results include a confidence score, mel spectrogram, and per-segment timeline.
          </p>
        </div>

        {/* ── Input form ── */}
        <form className="predict-form" onSubmit={handleSubmit}>
          <div className="mode-switch" role="tablist" aria-label="Input source">
            <button
              id="tab-youtube"
              type="button"
              role="tab"
              aria-selected={mode === 'youtube'}
              className={mode === 'youtube' ? 'mode-button active' : 'mode-button'}
              onClick={() => handleModeChange('youtube')}
            >
              YouTube Link
            </button>
            <button
              id="tab-upload"
              type="button"
              role="tab"
              aria-selected={mode === 'upload'}
              className={mode === 'upload' ? 'mode-button active' : 'mode-button'}
              onClick={() => handleModeChange('upload')}
            >
              Audio File Upload
            </button>
          </div>

          {mode === 'youtube' ? (
            <label className="field" htmlFor="youtube-url">
              <span>YouTube URL</span>
              <input
                id="youtube-url"
                type="url"
                placeholder="https://www.youtube.com/watch?v=..."
                value={youtubeUrl}
                onChange={(e) => setYoutubeUrl(e.target.value)}
              />
            </label>
          ) : (
            <label className="field" htmlFor="audio-file">
              <span>
                Audio File
                <small className="format-hint">.wav · .mp3 · .flac</small>
              </span>
              <input
                id="audio-file"
                type="file"
                accept=".wav,.mp3,.flac,audio/*"
                onChange={handleFileChange}
              />
              <small className="file-name">
                {audioFile ? `Selected: ${audioFile.name}` : 'No file selected.'}
              </small>
            </label>
          )}

          <button id="predict-btn" type="submit" className="predict-button" disabled={isSubmitting}>
            {isSubmitting
              ? <span className="btn-inner"><span className="spinner" aria-hidden="true" />Analysing…</span>
              : 'Run Detection'}
          </button>
        </form>

        {/* ── Error ── */}
        {error && <p className="error-box" role="alert">{error}</p>}

        {/* ── Skeleton while loading ── */}
        {isSubmitting && <SkeletonLoader />}

        {/* ── Results ── */}
        {result && !isSubmitting && (
          <section className="results-area" aria-live="polite" aria-label="Detection results">

            {/* Prediction banner */}
            <div className={`prediction-banner prediction-banner--${bannerVariant}`}>
              <span className="pred-icon" aria-hidden="true">
                {isAi ? '🤖' : isHuman ? '✅' : '❓'}
              </span>
              <div className="pred-text">
                <div className="pred-label">Prediction</div>
                <div className="pred-value">{predictionLabel}</div>
                {result.filename && (
                  <div className="pred-file" title={result.filename}>
                    <span className="pred-file-icon">📁</span>
                    <span className="pred-file-name">{result.filename}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Gauge + stats */}
            <div className="gauge-stats-row">
              <ConfidenceGauge score={avgScore} confidenceBand={confidenceBand} />

              <div className="stats-grid">
                <div className="stat-card">
                  <span>AI Probability Score</span>
                  <strong style={{
                    color: avgScore < 30 ? '#34d399' : avgScore < 90 ? '#fbbf24' : '#fb7185',
                  }}>
                    {avgScore.toFixed(2)}%
                  </strong>
                </div>
                <div className="stat-card">
                  <span>Decision Threshold</span>
                  <strong>{result.decision_threshold}%</strong>
                </div>
                <div className="stat-card">
                  <span>Chunks Analysed</span>
                  <strong>{result.chunk_count}</strong>
                </div>
                <div className="stat-card">
                  <span>High-Conf AI Chunks</span>
                  <strong>
                    {result.chunk_timeline
                      ? `${result.chunk_timeline.filter(c => c.score >= 90).length} / ${result.chunk_count}`
                      : '—'}
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

            {/* Audio player (upload mode only) */}
            {mode === 'upload' && <AudioPlayer file={audioFileRef.current} />}

            {/* Mel spectrogram */}
            <SpectrogramPanel spectrogramB64={result.spectrogram_b64} />

            {/* Segment timeline */}
            <SegmentTimeline timeline={result.chunk_timeline} />

            {/* Disclaimer */}
            <DisclaimerCallout />
          </section>
        )}
      </section>
    </main>
  )
}

export default App
