import os
import io
import asyncio
import base64
import json
from concurrent.futures import ThreadPoolExecutor
import tempfile
from urllib.parse import urlparse

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import onnxruntime as ort
from pydantic import BaseModel
from app.audio_utils import load_waveform
from app.inference_pipeline import (
    analyze_waveform_pipeline,
    apply_vad_to_waveform as pipeline_apply_vad_to_waveform,
    split_overlapping_chunks_with_offsets,
    score_waveforms as pipeline_score_waveforms,
    summarize_scores as pipeline_summarize_scores,
)
from app.youtube_utils import download_youtube_audio as download_youtube_audio_shared
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display

app = FastAPI(title="EchoAuthentic Voice Spoofing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/echo_authentic_v1.onnx"
INFERENCE_THRESHOLD = 0.98549
CONFIDENCE_HUMAN_MAX = 0.30
CONFIDENCE_AI_MIN = 0.90
CHUNK_SIZE_SECONDS = 4.0
CHUNK_OVERLAP_SECONDS = 0.5
CHUNK_ALERT_THRESHOLD = 0.90
CHUNK_ALERT_RATIO = 0.66
VAD_THRESHOLD = 0.50
ALLOWED_AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac')
ALLOWED_YOUTUBE_HOSTNAMES = {
    'youtube.com',
    'www.youtube.com',
    'm.youtube.com',
    'music.youtube.com',
    'youtu.be',
}
try:
    _sess_opts = ort.SessionOptions()
    # Suppress benign shape-mismatch warnings that appear when batching chunks
    # (model exported with static batch=1 but we run variable batch sizes).
    # 0=Verbose 1=Info 2=Warning 3=Error 4=Fatal
    _sess_opts.log_severity_level = 3
    ort_session = ort.InferenceSession(MODEL_PATH, sess_options=_sess_opts)
    print("ONNX Model loaded successfully into memory.")
except Exception as e:
    print(f"Error loading ONNX model: {e}")


class YouTubeRequest(BaseModel):
    url: str



def validate_youtube_url(url: str) -> None:
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or parsed_url.hostname not in ALLOWED_YOUTUBE_HOSTNAMES:
        raise HTTPException(
            status_code=400,
            detail="Please provide a valid YouTube link from youtube.com or youtu.be.",
        )


def generate_spectrogram_b64(waveform: np.ndarray, sample_rate: int) -> str | None:
    """Render a full-length mel spectrogram of the waveform as a base64 PNG.
    Caps display at 120 s for performance.
    """
    try:
        max_samples = 120 * sample_rate
        display_waveform = waveform[:max_samples] if len(waveform) > max_samples else waveform

        mel_spec = librosa.feature.melspectrogram(
            y=display_waveform,
            sr=sample_rate,
            n_mels=128,
            n_fft=512,
            hop_length=256,
            fmax=8000,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        fig, ax = plt.subplots(figsize=(14, 3.5))
        fig.patch.set_facecolor('#0a0d14')
        ax.set_facecolor('#0a0d14')

        img = librosa.display.specshow(
            mel_db,
            sr=sample_rate,
            hop_length=256,
            x_axis='time',
            y_axis='mel',
            ax=ax,
            fmax=8000,
            cmap='magma',
        )

        cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
        cbar.ax.yaxis.set_tick_params(color='#64748b')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#64748b', fontsize=8)

        ax.set_xlabel('Time (s)', color='#64748b', fontsize=9)
        ax.set_ylabel('Frequency', color='#64748b', fontsize=9)
        ax.tick_params(colors='#64748b', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e293b')

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=80, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as exc:
        print(f"[spectrogram] generation failed: {exc}")
        return None


@app.post("/predict/")
async def predict_audio(
    file: UploadFile = File(...),
    threshold: float = Query(INFERENCE_THRESHOLD, ge=0.0, le=1.0),
    confidence_low: float = Query(CONFIDENCE_HUMAN_MAX, ge=0.0, le=1.0),
    confidence_high: float = Query(CONFIDENCE_AI_MIN, ge=0.0, le=1.0),
    chunk_alert_threshold: float = Query(CHUNK_ALERT_THRESHOLD, ge=0.0, le=1.0),
    chunk_alert_ratio: float = Query(CHUNK_ALERT_RATIO, ge=0.0, le=1.0),
    chunk_size_seconds: float = Query(CHUNK_SIZE_SECONDS, ge=1.0, le=30.0),
    chunk_overlap_seconds: float = Query(CHUNK_OVERLAP_SECONDS, ge=0.0, le=29.0),
    vad_threshold: float = Query(VAD_THRESHOLD, ge=0.0, le=1.0),
):
    if not file.filename.endswith(ALLOWED_AUDIO_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Invalid audio format. Please upload a .wav, .mp3, or .flac file.")

    try:
        audio_bytes = await file.read()
        waveform, sample_rate = load_waveform(audio_bytes)

        # Run inference and spectrogram generation concurrently
        with ThreadPoolExecutor(max_workers=2) as pool:
            inference_future = pool.submit(
                analyze_waveform_pipeline,
                ort_session, waveform, sample_rate,
                threshold=threshold,
                human_max=confidence_low,
                ai_min=confidence_high,
                chunk_alert_threshold=chunk_alert_threshold,
                chunk_alert_ratio=chunk_alert_ratio,
                chunk_size_seconds=chunk_size_seconds,
                chunk_overlap_seconds=chunk_overlap_seconds,
                vad_threshold=vad_threshold,
            )
            spectrogram_future = pool.submit(generate_spectrogram_b64, waveform, sample_rate)
            response = inference_future.result()
            spectrogram_b64 = spectrogram_future.result()

        response["status"] = "success"
        response["filename"] = file.filename
        response["prediction"] = response["binary_prediction"]
        response["spectrogram_b64"] = spectrogram_b64
        response["confidence_band_thresholds"] = {
            "human_max": round(confidence_low, 2),
            "ai_min": round(confidence_high, 2),
        }
        response["chunk_thresholds"] = {
            "alert_score": round(chunk_alert_threshold, 2),
            "alert_ratio": round(chunk_alert_ratio, 2),
            "chunk_size_seconds": round(chunk_size_seconds, 2),
            "chunk_overlap_seconds": round(chunk_overlap_seconds, 2),
        }
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")


@app.post("/predict/youtube/")
async def predict_youtube(
    request: YouTubeRequest,
    threshold: float = Query(INFERENCE_THRESHOLD, ge=0.0, le=1.0),
    confidence_low: float = Query(CONFIDENCE_HUMAN_MAX, ge=0.0, le=1.0),
    confidence_high: float = Query(CONFIDENCE_AI_MIN, ge=0.0, le=1.0),
    chunk_alert_threshold: float = Query(CHUNK_ALERT_THRESHOLD, ge=0.0, le=1.0),
    chunk_alert_ratio: float = Query(CHUNK_ALERT_RATIO, ge=0.0, le=1.0),
    chunk_size_seconds: float = Query(CHUNK_SIZE_SECONDS, ge=1.0, le=30.0),
    chunk_overlap_seconds: float = Query(CHUNK_OVERLAP_SECONDS, ge=0.0, le=29.0),
    vad_threshold: float = Query(VAD_THRESHOLD, ge=0.0, le=1.0),
):
    try:
        audio_bytes, filename = download_youtube_audio_shared(request.url)
        waveform, sample_rate = load_waveform(audio_bytes)
        summary = analyze_waveform_pipeline(
            ort_session,
            waveform,
            sample_rate,
            threshold=threshold,
            human_max=confidence_low,
            ai_min=confidence_high,
            chunk_size_seconds=chunk_size_seconds,
            chunk_overlap_seconds=chunk_overlap_seconds,
            chunk_alert_threshold=chunk_alert_threshold,
            chunk_alert_ratio=chunk_alert_ratio,
            vad_threshold=vad_threshold,
        )

        return {
            "status": "success",
            "filename": filename,
            "source_url": request.url,
            **summary,
            "prediction": summary["binary_prediction"],
            "spectrogram_b64": generate_spectrogram_b64(waveform, sample_rate),
            "confidence_band_thresholds": {
                "human_max": round(confidence_low, 2),
                "ai_min": round(confidence_high, 2),
            },
            "chunk_thresholds": {
                "alert_score": round(chunk_alert_threshold, 2),
                "alert_ratio": round(chunk_alert_ratio, 2),
                "chunk_size_seconds": round(chunk_size_seconds, 2),
                "chunk_overlap_seconds": round(chunk_overlap_seconds, 2),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the YouTube link: {str(e)}")


# ── Streaming SSE endpoint for YouTube (shows per-stage progress) ─────────────

@app.post("/predict/youtube/stream/")
async def predict_youtube_stream(
    request: YouTubeRequest,
    threshold: float = Query(INFERENCE_THRESHOLD, ge=0.0, le=1.0),
    confidence_low: float = Query(CONFIDENCE_HUMAN_MAX, ge=0.0, le=1.0),
    confidence_high: float = Query(CONFIDENCE_AI_MIN, ge=0.0, le=1.0),
    chunk_alert_threshold: float = Query(CHUNK_ALERT_THRESHOLD, ge=0.0, le=1.0),
    chunk_alert_ratio: float = Query(CHUNK_ALERT_RATIO, ge=0.0, le=1.0),
    chunk_size_seconds: float = Query(CHUNK_SIZE_SECONDS, ge=1.0, le=30.0),
    chunk_overlap_seconds: float = Query(CHUNK_OVERLAP_SECONDS, ge=0.0, le=29.0),
    vad_threshold: float = Query(VAD_THRESHOLD, ge=0.0, le=1.0),
):
    """Server-Sent Events endpoint: streams progress events then the final result."""

    def sse(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    async def event_stream():
        try:
            validate_youtube_url(request.url)

            yield sse({"stage": "downloading", "label": "Downloading audio from YouTube…", "pct": 5})

            # 1. Download
            audio_bytes, filename = await asyncio.to_thread(
                download_youtube_audio_shared, request.url
            )

            yield sse({"stage": "loading", "label": "Loading and resampling audio…", "pct": 32})

            # 2. Load waveform
            waveform, sample_rate = await asyncio.to_thread(load_waveform, audio_bytes)

            yield sse({"stage": "vad", "label": "Detecting speech segments (VAD)…", "pct": 48})

            # 3. VAD
            speech_waveform, vad_summary = await asyncio.to_thread(
                pipeline_apply_vad_to_waveform, waveform, sample_rate, vad_threshold
            )

            # 4. Chunk (fast — no yield needed)
            chunks, offsets = split_overlapping_chunks_with_offsets(
                speech_waveform, sample_rate, chunk_size_seconds, chunk_overlap_seconds
            )

            yield sse({"stage": "inference", "label": f"Running AI analysis on {len(chunks)} chunks…", "pct": 62})

            # 5. Inference + spectrogram concurrently
            inference_task = asyncio.to_thread(pipeline_score_waveforms, ort_session, chunks)
            spectrogram_task = asyncio.to_thread(generate_spectrogram_b64, waveform, sample_rate)

            chunk_scores = await inference_task

            yield sse({"stage": "spectrogram", "label": "Generating mel spectrogram…", "pct": 88})

            spectrogram_b64 = await spectrogram_task

            # 6. Summarise
            summary = pipeline_summarize_scores(
                chunk_scores,
                human_max=confidence_low,
                ai_min=confidence_high,
                chunk_alert_threshold=chunk_alert_threshold,
                chunk_alert_ratio=chunk_alert_ratio,
            )

            average_probability = summary["average_raw_probability"]
            average_prediction = "AI-Generated" if average_probability >= threshold else "Human Voice"

            chunk_timeline = [
                {
                    "start_sec": start,
                    "end_sec": end,
                    "score": round(score * 100, 2),
                    "is_ai": score >= threshold,
                }
                for (start, end), score in zip(offsets, chunk_scores)
            ]

            result = {
                "status": "success",
                "filename": filename,
                "source_url": request.url,
                "average_prediction": average_prediction,
                "binary_prediction": summary["chunk_vote_prediction"],
                "prediction": summary["chunk_vote_prediction"],
                "decision_threshold": round(threshold * 100, 2),
                "chunk_count": len(chunk_scores),
                "chunk_scores": [round(s, 6) for s in chunk_scores],
                "chunk_timeline": chunk_timeline,
                "vad_summary": vad_summary,
                **summary,
                "spectrogram_b64": spectrogram_b64,
                "confidence_band_thresholds": {
                    "human_max": round(confidence_low, 2),
                    "ai_min": round(confidence_high, 2),
                },
                "chunk_thresholds": {
                    "alert_score": round(chunk_alert_threshold, 2),
                    "alert_ratio": round(chunk_alert_ratio, 2),
                    "chunk_size_seconds": round(chunk_size_seconds, 2),
                    "chunk_overlap_seconds": round(chunk_overlap_seconds, 2),
                },
            }

            yield sse({"stage": "done", "pct": 100, "result": result})

        except HTTPException as exc:
            yield sse({"stage": "error", "message": exc.detail})
        except Exception as exc:
            yield sse({"stage": "error", "message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )