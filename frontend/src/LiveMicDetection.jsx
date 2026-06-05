import { useState, useRef, useCallback, useEffect } from 'react'

// ─── Constants — must match backend/app/audio_utils.py ───────────────────────
const SR             = 16000
const N_FFT          = 1024
const HOP            = 512
const N_MELS         = 128
const FMAX           = 8000
const TARGET_SAMPLES = 64000       // 4 s at 16 kHz
const N_FRAMES       = Math.floor(TARGET_SAMPLES / HOP) + 1   // 126

// ─── VAD tuning ──────────────────────────────────────────────────────────────
const ENERGY_THRESH  = 0.012       // RMS threshold for speech vs. silence
const SILENCE_MS     = 600         // consecutive silence before speech-end
const MIN_SPEECH_MS  = 300         // ignore segments shorter than this
const MAX_SPEECH_MS  = 4000        // hard cap — flush buffer at 4 s

// ─── Mel-scale helpers — librosa htk=False (Slaney) ──────────────────────────
function _hzToMel(hz) {
  const f_sp = 200.0 / 3, minLogHz = 1000.0, minLogMel = minLogHz / f_sp
  const logStep = Math.log(6.4) / 27.0
  return hz >= minLogHz ? minLogMel + Math.log(hz / minLogHz) / logStep : hz / f_sp
}
function _melToHz(mel) {
  const f_sp = 200.0 / 3, minLogHz = 1000.0, minLogMel = minLogHz / f_sp
  const logStep = Math.log(6.4) / 27.0
  return mel >= minLogMel ? minLogHz * Math.exp(logStep * (mel - minLogMel)) : f_sp * mel
}

// ─── Mel filterbank — librosa norm='slaney' ───────────────────────────────────
function _buildMelFilters() {
  const n_freqs = N_FFT / 2 + 1
  const fftFreqs = Float32Array.from({ length: n_freqs }, (_, k) => k * SR / N_FFT)
  const melMin = _hzToMel(0), melMax = _hzToMel(FMAX)
  const mel_f = Float32Array.from({ length: N_MELS + 2 }, (_, i) =>
    _melToHz(melMin + (i / (N_MELS + 1)) * (melMax - melMin)))
  const fdiff = Float32Array.from({ length: N_MELS + 1 }, (_, i) => mel_f[i + 1] - mel_f[i])
  const enorm = Float32Array.from({ length: N_MELS }, (_, m) => 2.0 / (mel_f[m + 2] - mel_f[m]))
  return Array.from({ length: N_MELS }, (_, m) => {
    const filt = new Float32Array(n_freqs)
    for (let k = 0; k < n_freqs; k++) {
      const lower = (fftFreqs[k] - mel_f[m]) / fdiff[m]
      const upper = (mel_f[m + 2] - fftFreqs[k]) / fdiff[m + 1]
      filt[k] = Math.max(0.0, Math.min(lower, upper)) * enorm[m]
    }
    return filt
  })
}

// ─── Radix-2 FFT (in-place) ───────────────────────────────────────────────────
function _inplaceFFT(re, im) {
  const n = re.length
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2.0 * Math.PI / len
    const wc0 = Math.cos(ang), ws0 = Math.sin(ang)
    for (let i = 0; i < n; i += len) {
      let wc = 1.0, ws = 0.0
      const half = len >> 1
      for (let j = 0; j < half; j++) {
        const h = i + j + half
        const uRe = re[i+j], uIm = im[i+j]
        const vRe = re[h]*wc - im[h]*ws, vIm = re[h]*ws + im[h]*wc
        re[i+j] = uRe+vRe; im[i+j] = uIm+vIm; re[h] = uRe-vRe; im[h] = uIm-vIm
        const nc = wc*wc0 - ws*ws0; ws = wc*ws0 + ws*wc0; wc = nc
      }
    }
  }
}

// Pre-built (once at module load, no I/O)
const MEL_FILTERS = _buildMelFilters()
const HANN_WINDOW = Float32Array.from({ length: N_FFT }, (_, i) =>
  0.5 * (1.0 - Math.cos(2.0 * Math.PI * i / N_FFT)))   // periodic Hann (librosa default)

// ─── Mel spectrogram — mirrors waveform_to_model_input() in audio_utils.py ───
function computeMelSpec(audio) {
  const y = new Float32Array(TARGET_SAMPLES)
  y.set(audio.subarray(0, Math.min(audio.length, TARGET_SAMPLES)))
  const pad = N_FFT >> 1, pLen = TARGET_SAMPLES + 2 * pad
  const padded = new Float32Array(pLen)
  padded.set(y, pad)
  for (let i = 0; i < pad; i++) {
    padded[pad - 1 - i]              = y[Math.min(i + 1, TARGET_SAMPLES - 1)]
    padded[pad + TARGET_SAMPLES + i] = y[Math.max(TARGET_SAMPLES - 2 - i, 0)]
  }
  const n_freqs = N_FFT / 2 + 1
  const nFrames = Math.floor((pLen - N_FFT) / HOP) + 1   // 126
  const spec    = new Float32Array(N_MELS * nFrames)
  const re = new Float32Array(N_FFT), im = new Float32Array(N_FFT)
  for (let f = 0; f < nFrames; f++) {
    const start = f * HOP
    for (let i = 0; i < N_FFT; i++) { re[i] = padded[start+i] * HANN_WINDOW[i]; im[i] = 0 }
    _inplaceFFT(re, im)
    const pwr = new Float32Array(n_freqs)
    for (let i = 0; i < n_freqs; i++) pwr[i] = re[i]*re[i] + im[i]*im[i]
    for (let m = 0; m < N_MELS; m++) {
      let e = 0; const filt = MEL_FILTERS[m]
      for (let k = 0; k < n_freqs; k++) e += filt[k] * pwr[k]
      spec[m * nFrames + f] = e
    }
  }
  return spec
}

// ─── Linear resampler: browser audio (44100/48000) → 16 kHz ─────────────────
function resampleTo16k(data, fromSR) {
  if (fromSR === SR) return data.slice()
  const ratio  = fromSR / SR
  const newLen = Math.floor(data.length / ratio)
  const out    = new Float32Array(newLen)
  for (let i = 0; i < newLen; i++) {
    const src = i * ratio
    const lo  = Math.floor(src)
    const hi  = Math.min(lo + 1, data.length - 1)
    out[i]    = data[lo] + (data[hi] - data[lo]) * (src - lo)
  }
  return out
}

// ─── ONNX Runtime — lazy dynamic import (avoids CJS/ESM build issues) ────────
let _ort = null, _session = null

async function getOrtRuntime() {
  if (_ort) return _ort
  _ort = await import('onnxruntime-web')
  _ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/'
  return _ort
}

async function getOrtSession() {
  if (_session) return _session
  const ort  = await getOrtRuntime()
  _session   = await ort.InferenceSession.create('/models/echo_authentic_v1.onnx', {
    executionProviders: ['wasm'],
  })
  return _session
}

async function runInference(audio) {
  const ort     = await getOrtRuntime()
  const session = await getOrtSession()
  const spec    = computeMelSpec(audio)
  const tensor  = new ort.Tensor('float32', spec, [1, 1, N_MELS, N_FRAMES])
  const outputs = await session.run({ [session.inputNames[0]]: tensor })
  return outputs[session.outputNames[0]].data[0]
}

// ─── Inline gauge ─────────────────────────────────────────────────────────────
function LiveGauge({ score }) {
  const R = 78, CX = 110, CY = 108, AL = Math.PI * R
  const pct = Math.min(100, Math.max(0, score ?? 0))
  const off = AL * (1 - pct / 100)
  const col = pct < 30 ? '#10b981' : pct < 90 ? '#f59e0b' : '#f43f5e'
  return (
    <svg viewBox="0 0 220 145" xmlns="http://www.w3.org/2000/svg" style={{ width: '100%', overflow: 'visible' }}>
      <path d={`M ${CX-R},${CY} A ${R},${R} 0 0,1 ${CX+R},${CY}`}
        fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="14" strokeLinecap="round" />
      <path d={`M ${CX-R},${CY} A ${R},${R} 0 0,1 ${CX+R},${CY}`}
        fill="none" stroke={col} strokeWidth="14" strokeLinecap="round"
        strokeDasharray={AL} strokeDashoffset={off}
        style={{ transition: 'stroke-dashoffset 0.9s cubic-bezier(0.4,0,0.2,1), stroke 0.4s ease', filter: `drop-shadow(0 0 10px ${col}90)` }}
      />
      <text x="110" y="97"  textAnchor="middle" fontSize="27" fontWeight="800" fill="#f8fafc" fontFamily="Inter,sans-serif">{pct.toFixed(1)}%</text>
      <text x="110" y="116" textAnchor="middle" fontSize="9"  fill="#475569" fontFamily="Inter,sans-serif" letterSpacing="0.14em">AI PROBABILITY</text>
      <text x="26"  y={CY+20} textAnchor="middle" fontSize="8" fill="#334155" fontFamily="Inter,sans-serif">0%</text>
      <text x="194" y={CY+20} textAnchor="middle" fontSize="8" fill="#334155" fontFamily="Inter,sans-serif">100%</text>
    </svg>
  )
}

// ─── LiveMicDetection ─────────────────────────────────────────────────────────
export default function LiveMicDetection() {
  const [status,  setStatus]  = useState('idle')
  const [score,   setScore]   = useState(null)
  const [history, setHistory] = useState([])
  const [error,   setError]   = useState('')
  const [active,  setActive]  = useState(false)

  // Audio refs (mutable, no re-render needed)
  const audioCtxRef    = useRef(null)
  const streamRef      = useRef(null)
  const processorRef   = useRef(null)
  const analyserRef    = useRef(null)
  const canvasRef      = useRef(null)
  const animFrameRef   = useRef(null)

  // VAD state (in refs to avoid stale closure inside onaudioprocess)
  const isActiveRef      = useRef(false)
  const vadStateRef      = useRef('idle')   // idle | listening | collecting
  const speechBufRef     = useRef([])
  const speechStartRef   = useRef(0)
  const silenceStartRef  = useRef(null)

  // ── Waveform oscilloscope ───────────────────────────────────────────────────
  const drawWaveform = useCallback(() => {
    const canvas   = canvasRef.current
    const analyser = analyserRef.current
    if (!canvas || !analyser) return

    const ctx    = canvas.getContext('2d')
    const W = canvas.width, H = canvas.height
    const bufLen = analyser.frequencyBinCount
    const data   = new Uint8Array(bufLen)
    analyser.getByteTimeDomainData(data)

    ctx.clearRect(0, 0, W, H)

    // Centre line
    ctx.strokeStyle = 'rgba(255,255,255,0.05)'
    ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(0, H/2); ctx.lineTo(W, H/2); ctx.stroke()

    // Waveform
    const grad = ctx.createLinearGradient(0, 0, W, 0)
    grad.addColorStop(0,   '#7c3aed')
    grad.addColorStop(0.5, '#3b82f6')
    grad.addColorStop(1,   '#7c3aed')
    ctx.strokeStyle = grad
    ctx.lineWidth = 2.5
    ctx.shadowColor = '#3b82f680'
    ctx.shadowBlur  = 6
    ctx.beginPath()
    const sw = W / bufLen
    for (let i = 0; i < bufLen; i++) {
      const y = ((data[i] / 128.0) * H) / 2
      i === 0 ? ctx.moveTo(i * sw, y) : ctx.lineTo(i * sw, y)
    }
    ctx.stroke()
    ctx.shadowBlur = 0

    animFrameRef.current = requestAnimationFrame(drawWaveform)
  }, [])

  // ── Tear down all audio resources ───────────────────────────────────────────
  const stopAudio = useCallback(() => {
    if (animFrameRef.current) { cancelAnimationFrame(animFrameRef.current); animFrameRef.current = null }
    if (processorRef.current) {
      processorRef.current.onaudioprocess = null
      processorRef.current.disconnect()
      processorRef.current = null
    }
    if (analyserRef.current) { analyserRef.current.disconnect(); analyserRef.current = null }
    if (streamRef.current)   { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null }
    if (audioCtxRef.current) { audioCtxRef.current.close().catch(() => {}); audioCtxRef.current = null }
    vadStateRef.current   = 'idle'
    speechBufRef.current  = []
    isActiveRef.current   = false
    silenceStartRef.current = null
  }, [])

  // ── Process a completed speech segment ─────────────────────────────────────
  const processSpeechSegment = useCallback(async (chunks, sampleRate) => {
    const totalLen = chunks.reduce((s, c) => s + c.length, 0)
    const raw = new Float32Array(totalLen)
    let off = 0
    for (const c of chunks) { raw.set(c, off); off += c.length }

    const audio16k = resampleTo16k(raw, sampleRate)

    setStatus('analysing')
    try {
      const rawScore = await runInference(audio16k)
      const pct = Math.min(100, Math.max(0, rawScore * 100))
      setScore(pct)
      setHistory(prev => [...prev.slice(-7), pct])
    } catch (err) {
      console.error('[LiveMic]', err)
      setError('Inference failed — ' + (err.message || 'unknown'))
    }
    if (isActiveRef.current) setStatus('listening')
  }, [])

  // ── Start ───────────────────────────────────────────────────────────────────
  const handleStart = useCallback(async () => {
    setError(''); setScore(null); setHistory([])
    setActive(true); isActiveRef.current = true
    setStatus('initialising')

    try {
      // Pre-load ONNX model concurrently while waiting for mic permission
      const modelPromise = getOrtSession()

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      })
      streamRef.current = stream
      await modelPromise   // ensure model is ready before we start collecting

      const audioCtx   = new AudioContext()
      audioCtxRef.current = audioCtx
      const sampleRate = audioCtx.sampleRate   // 44100 or 48000 typically

      const source  = audioCtx.createMediaStreamSource(stream)

      // Analyser for oscilloscope
      const analyser = audioCtx.createAnalyser()
      analyser.fftSize = 1024
      analyser.smoothingTimeConstant = 0.7
      analyserRef.current = analyser

      // ScriptProcessorNode for raw PCM (deprecated but universally supported)
      const PROC_BUF  = 4096
      const processor = audioCtx.createScriptProcessor(PROC_BUF, 1, 1)
      processorRef.current = processor

      // Silent gain node — prevents mic feedback while keeping events firing
      const silence = audioCtx.createGain()
      silence.gain.value = 0

      source.connect(analyser)
      analyser.connect(processor)
      processor.connect(silence)
      silence.connect(audioCtx.destination)

      vadStateRef.current = 'listening'

      processor.onaudioprocess = (ev) => {
        if (!isActiveRef.current) return
        const input = ev.inputBuffer.getChannelData(0)

        // RMS energy
        let sq = 0
        for (let i = 0; i < input.length; i++) sq += input[i] * input[i]
        const rms      = Math.sqrt(sq / input.length)
        const isSpeech = rms > ENERGY_THRESH
        const now      = performance.now()

        if (vadStateRef.current === 'listening') {
          if (isSpeech) {
            vadStateRef.current  = 'collecting'
            speechStartRef.current = now
            speechBufRef.current = [input.slice()]
            silenceStartRef.current = null
            setStatus('speaking')
          }
        } else if (vadStateRef.current === 'collecting') {
          speechBufRef.current.push(input.slice())

          if (!isSpeech) {
            if (silenceStartRef.current === null) silenceStartRef.current = now
            else if (now - silenceStartRef.current >= SILENCE_MS) {
              // Speech ended naturally
              vadStateRef.current = 'listening'
              silenceStartRef.current = null
              const speechDur = now - speechStartRef.current
              if (speechDur >= MIN_SPEECH_MS) {
                const chunks = speechBufRef.current.slice()
                speechBufRef.current = []
                processSpeechSegment(chunks, sampleRate)
              } else {
                speechBufRef.current = []
                setStatus('listening')
              }
            }
          } else {
            silenceStartRef.current = null
            // Hard cap — flush at MAX_SPEECH_MS
            if (now - speechStartRef.current >= MAX_SPEECH_MS) {
              vadStateRef.current = 'listening'
              const chunks = speechBufRef.current.slice()
              speechBufRef.current = []
              processSpeechSegment(chunks, sampleRate)
            }
          }
        }
      }

      setStatus('listening')
      drawWaveform()

    } catch (err) {
      const msg = err?.message || ''
      setError(
        msg.toLowerCase().includes('permission') || msg.toLowerCase().includes('denied')
          ? 'Microphone access denied. Allow mic access in your browser settings and try again.'
          : 'Could not start: ' + msg
      )
      setStatus('error')
      setActive(false)
      stopAudio()
    }
  }, [drawWaveform, processSpeechSegment, stopAudio])

  // ── Stop ────────────────────────────────────────────────────────────────────
  const handleStop = useCallback(() => {
    stopAudio()
    setActive(false)
    setStatus('idle')
  }, [stopAudio])

  // Cleanup on tab switch / unmount
  useEffect(() => () => stopAudio(), [stopAudio])

  // ── Derived ─────────────────────────────────────────────────────────────────
  const verdict = score === null ? null
    : score < 30 ? { text: '🟢 Human Voice',  cls: 'human' }
    : score < 90 ? { text: '🟡 Uncertain',     cls: 'uncertain' }
    :              { text: '🔴 AI-Generated',   cls: 'ai' }

  const avgScore = history.length ? history.reduce((a, b) => a + b, 0) / history.length : null

  const STATUS_MAP = {
    idle: '● Ready', initialising: '⏳ Loading…', listening: '👂 Listening…',
    speaking: '🎤 Speaking…', analysing: '🧠 Analysing…', error: '⚠ Error',
  }

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div className="live-mic-panel">

      {/* ── Header ── */}
      <div className="live-mic-header">
        <span className="live-mic-icon-badge" aria-hidden="true">🎙</span>
        <div className="live-mic-title-wrap">
          <h3 className="live-mic-title">Live Mic Detection</h3>
          <p className="live-mic-subtitle">
            Web Audio VAD · EchoAuthentic ONNX · 100% on-device · no audio leaves your device
          </p>
        </div>
        {active && (
          <span className={`live-status-chip live-status-chip--${status}`}>
            {STATUS_MAP[status] ?? status}
          </span>
        )}
      </div>

      {/* ── Error ── */}
      {error && (
        <div className="live-error-box" role="alert">
          <span aria-hidden="true">⚠️</span><span>{error}</span>
        </div>
      )}

      {/* ── Mic button ── */}
      <div className="mic-btn-wrap">
        {active && <span className="mic-ring mic-ring--1" aria-hidden="true" />}
        {active && <span className="mic-ring mic-ring--2" aria-hidden="true" />}
        <button
          id="live-mic-start-stop"
          className={`mic-btn${active ? ' mic-btn--active' : ''}`}
          onClick={active ? handleStop : handleStart}
          disabled={status === 'initialising'}
          aria-label={active ? 'Stop live detection' : 'Start live detection'}
        >
          {status === 'initialising' ? (
            <span className="mic-spinner" aria-hidden="true" />
          ) : active ? (
            <svg viewBox="0 0 24 24" fill="currentColor" width="30" height="30" aria-hidden="true">
              <rect x="5" y="5" width="14" height="14" rx="2" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" fill="currentColor" width="30" height="30" aria-hidden="true">
              <path d="M12 1a4 4 0 0 1 4 4v7a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4z"/>
              <path d="M19 12a7 7 0 0 1-14 0H3a9 9 0 0 0 18 0h-2z"/>
              <line x1="12" y1="21" x2="12" y2="23" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          )}
        </button>
        <p className="mic-btn-label">
          {status === 'initialising' ? 'Loading model…' : active ? 'Click to stop' : 'Start Detection'}
        </p>
      </div>

      {/* ── Live oscilloscope ── */}
      {active && (
        <div className="live-waveform-wrap">
          <canvas
            ref={canvasRef}
            width={640}
            height={72}
            className="live-waveform-canvas"
            aria-label="Live microphone waveform"
          />
          <span
            className="live-speak-label"
            style={{ color: status === 'speaking' ? '#fbbf24' : '#475569' }}
          >
            {status === 'speaking'
              ? '🎤 Voice detected — will analyse when you pause'
              : status === 'analysing'
              ? '🧠 Analysing…'
              : '👂 Waiting for speech…'}
          </span>
        </div>
      )}

      {/* ── Analysing spinner (outside waveform section) ── */}
      {status === 'analysing' && !active && (
        <div className="live-analysing-indicator" aria-live="polite">
          <span className="live-analysing-spinner" aria-hidden="true" />
          <span className="live-analysing-text">Running deepfake analysis…</span>
        </div>
      )}

      {/* ── Results ── */}
      {score !== null && (
        <div className="live-results-area">

          <div className={`live-verdict-banner live-verdict-banner--${verdict?.cls}`} aria-live="polite">
            <span className="live-verdict-text">{verdict?.text}</span>
          </div>

          <div className="live-gauge-row">
            <div className="live-gauge-wrap">
              <LiveGauge score={score} />
            </div>
            <div className="live-stats-wrap">
              <div className="live-stat">
                <span className="live-stat-label">Latest Score</span>
                <strong className="live-stat-value"
                  style={{ color: score < 30 ? '#34d399' : score < 90 ? '#fbbf24' : '#fb7185' }}>
                  {score.toFixed(2)}%
                </strong>
              </div>
              <div className="live-stat">
                <span className="live-stat-label">Segments Analysed</span>
                <strong className="live-stat-value">{history.length}</strong>
              </div>
              {avgScore !== null && history.length > 1 && (
                <div className="live-stat">
                  <span className="live-stat-label">Session Average</span>
                  <strong className="live-stat-value"
                    style={{ color: avgScore < 30 ? '#34d399' : avgScore < 90 ? '#fbbf24' : '#fb7185' }}>
                    {avgScore.toFixed(1)}%
                  </strong>
                </div>
              )}
            </div>
          </div>

          <div className="live-sparkline-panel">
            <span className="live-sparkline-title">Segment History ({history.length})</span>
            <div className="live-sparkline">
              {history.map((s, i) => {
                const col = s < 30 ? '#10b981' : s < 90 ? '#f59e0b' : '#f43f5e'
                return (
                  <div key={i} className="live-spark-col" title={`Segment ${i+1}: ${s.toFixed(1)}%`}>
                    <div className="live-spark-bar-bg">
                      <div className="live-spark-bar-fill"
                        style={{ height: `${Math.max(s, 3)}%`, background: col, boxShadow: `0 0 6px ${col}50` }} />
                    </div>
                    <span className="live-spark-num">{s.toFixed(0)}%</span>
                  </div>
                )
              })}
            </div>
          </div>

        </div>
      )}

      {/* ── Idle / listening hint ── */}
      {score === null && !error && (
        <div className="live-hint">
          {!active ? (
            <p>Click <strong>Start Detection</strong> to begin. Speak naturally — analysis triggers automatically after each speech segment.</p>
          ) : status === 'listening' ? (
            <p>🎤 Say something for a few seconds, then pause. The model will analyse each segment automatically.</p>
          ) : null}
        </div>
      )}

      {/* ── Privacy strip ── */}
      <div className="live-privacy-strip">
        <span aria-hidden="true">🔒</span>
        <span>Audio processed entirely on-device via WebAssembly</span>
        <span className="live-privacy-sep" aria-hidden="true">·</span>
        <span aria-hidden="true">⚠️</span>
        <span>Trained on ASVspoof 2019 — may not catch the latest AI voices</span>
      </div>

    </div>
  )
}
