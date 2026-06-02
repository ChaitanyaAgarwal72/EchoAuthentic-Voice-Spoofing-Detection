import os
import io
import base64
import importlib
from functools import lru_cache
import tempfile
from urllib.parse import urlparse

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
from pydantic import BaseModel
from app.audio_utils import load_waveform, process_audio, waveform_to_model_input
from app.inference_pipeline import analyze_audio_bytes_pipeline, analyze_waveform_pipeline, classify_confidence_band as pipeline_classify_confidence_band
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
CHUNK_OVERLAP_SECONDS = 1.3
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
    ort_session = ort.InferenceSession(MODEL_PATH)
    print("ONNX Model loaded successfully into memory.")
except Exception as e:
    print(f"Error loading ONNX model: {e}")


class YouTubeRequest(BaseModel):
    url: str


def classify_confidence_band(score: float, human_max: float = CONFIDENCE_HUMAN_MAX, ai_min: float = CONFIDENCE_AI_MIN) -> str:
    if score < human_max:
        return "High Confidence Human"
    if score <= ai_min:
        return "Unverifiable / Degraded Audio"
    return "High Confidence AI"


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
            n_fft=1024,
            hop_length=512,
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
            hop_length=512,
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
        fig.savefig(buf, format='png', dpi=110, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as exc:
        print(f"[spectrogram] generation failed: {exc}")
        return None


def predict_probability_from_waveform(waveform: np.ndarray) -> float:
    input_tensor = waveform_to_model_input(waveform)
    onnx_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    onnx_outputs = ort_session.run(None, onnx_inputs)
    return float(onnx_outputs[0][0])


def analyze_audio_bytes(audio_bytes: bytes, source_name: str, threshold: float = INFERENCE_THRESHOLD) -> dict:
    waveform, _ = load_waveform(audio_bytes)
    ai_probability = predict_probability_from_waveform(waveform)
    score_percent = round(ai_probability * 100, 2)
    threshold_percent = round(threshold * 100, 2)
    confidence_band = classify_confidence_band(ai_probability)

    result = "AI-Generated" if ai_probability >= threshold else "Human Voice"

    return {
        "status": "success",
        "filename": source_name,
        "prediction": result,
        "confidence_band": confidence_band,
        "ai_probability_score": score_percent,
        "raw_ai_probability": round(ai_probability, 6),
        "decision_threshold": threshold_percent,
    }


def load_vad_components():
    try:
        silero_vad = importlib.import_module("silero_vad")
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="Missing VAD dependencies. Install silero-vad and torch in your venv.",
        ) from exc

    vad_model = silero_vad.load_silero_vad()
    return silero_vad, torch, vad_model


@lru_cache(maxsize=1)
def get_cached_vad_model():
    return load_vad_components()


def apply_vad_to_waveform(waveform: np.ndarray, sample_rate: int, vad_threshold: float) -> tuple[np.ndarray, dict]:
    silero_vad, torch, vad_model = get_cached_vad_model()
    waveform_tensor = torch.from_numpy(waveform.astype(np.float32))

    speech_timestamps = silero_vad.get_speech_timestamps(
        waveform_tensor,
        vad_model,
        sampling_rate=sample_rate,
        threshold=vad_threshold,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
        speech_pad_ms=150,
    )

    if not speech_timestamps:
        return waveform.astype(np.float32), {
            "vad_applied": False,
            "speech_segments": 0,
            "speech_samples": int(len(waveform)),
            "vad_note": "No speech detected; using the full audio as fallback.",
        }

    speech_chunks = [waveform[item["start"]:item["end"]] for item in speech_timestamps]
    speech_waveform = np.concatenate(speech_chunks).astype(np.float32) if speech_chunks else waveform.astype(np.float32)

    return speech_waveform, {
        "vad_applied": True,
        "speech_segments": len(speech_timestamps),
        "speech_samples": int(len(speech_waveform)),
        "vad_note": "Speech-only audio extracted successfully.",
    }


def split_overlapping_chunks(waveform: np.ndarray, sample_rate: int, chunk_size_seconds: float, chunk_overlap_seconds: float) -> list[np.ndarray]:
    chunk_size_samples = max(1, int(chunk_size_seconds * sample_rate))
    chunk_overlap_samples = max(0, int(chunk_overlap_seconds * sample_rate))
    step = max(1, chunk_size_samples - chunk_overlap_samples)

    if len(waveform) <= chunk_size_samples:
        return [waveform_to_model_input(waveform, sample_rate).squeeze(0).squeeze(0)]

    chunks: list[np.ndarray] = []
    start = 0
    while start < len(waveform):
        end = start + chunk_size_samples
        chunk = waveform[start:end]
        if len(chunk) < chunk_size_samples:
            chunk = np.pad(chunk, (0, chunk_size_samples - len(chunk))).astype(np.float32)
        chunks.append(chunk.astype(np.float32))
        if end >= len(waveform):
            break
        start += step

    return chunks


def score_waveform_chunks(waveform_chunks: list[np.ndarray]) -> list[float]:
    scores: list[float] = []
    for chunk in waveform_chunks:
        input_tensor = waveform_to_model_input(chunk)
        onnx_inputs = {ort_session.get_inputs()[0].name: input_tensor}
        onnx_outputs = ort_session.run(None, onnx_inputs)
        scores.append(float(onnx_outputs[0][0]))
    return scores


def summarize_chunk_scores(scores: list[float], band_low: float, band_high: float, chunk_alert_threshold: float, chunk_alert_ratio: float) -> dict:
    average_score = sum(scores) / len(scores)
    confidence_band = classify_confidence_band(average_score, band_low, band_high)
    high_chunk_ratio = sum(score >= chunk_alert_threshold for score in scores) / len(scores)
    video_flagged = high_chunk_ratio > chunk_alert_ratio

    return {
        "average_raw_probability": round(average_score, 6),
        "average_ai_probability_score": round(average_score * 100, 2),
        "confidence_band": confidence_band,
        "chunk_alert_threshold": round(chunk_alert_threshold * 100, 2),
        "chunk_alert_ratio": round(chunk_alert_ratio, 2),
        "chunk_high_score_ratio": round(high_chunk_ratio, 4),
        "video_flagged": video_flagged,
    }


def validate_youtube_url(url: str) -> None:
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or parsed_url.hostname not in ALLOWED_YOUTUBE_HOSTNAMES:
        raise HTTPException(
            status_code=400,
            detail="Please provide a valid YouTube link from youtube.com or youtu.be.",
        )


def download_youtube_audio(url: str) -> tuple[bytes, str]:
    validate_youtube_url(url)

    try:
        yt_dlp = importlib.import_module("yt_dlp")
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="Missing dependencies for YouTube processing. Install yt-dlp and imageio-ffmpeg in your venv.",
        ) from exc

    with tempfile.TemporaryDirectory() as temp_dir:
        output_template = os.path.join(temp_dir, "youtube_audio.%(ext)s")
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        ydl_options = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "ffmpeg_location": ffmpeg_path,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
        }

        with yt_dlp.YoutubeDL(ydl_options) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title") or "youtube_audio"

        audio_path = os.path.join(temp_dir, "youtube_audio.wav")
        if not os.path.exists(audio_path):
            raise HTTPException(
                status_code=500,
                detail="YouTube audio download completed, but the WAV file could not be found.",
            )

        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        return audio_bytes, f"{title}.wav"

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

        response = analyze_waveform_pipeline(
            ort_session,
            waveform,
            sample_rate,
            threshold=threshold,
            human_max=confidence_low,
            ai_min=confidence_high,
            chunk_alert_threshold=chunk_alert_threshold,
            chunk_alert_ratio=chunk_alert_ratio,
            chunk_size_seconds=chunk_size_seconds,
            chunk_overlap_seconds=chunk_overlap_seconds,
            vad_threshold=vad_threshold,
        )

        response["status"] = "success"
        response["filename"] = file.filename
        response["prediction"] = response["binary_prediction"]
        response["spectrogram_b64"] = generate_spectrogram_b64(waveform, sample_rate)
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