from __future__ import annotations

from functools import lru_cache

import importlib
from fastapi import HTTPException
import numpy as np

from app.audio_utils import load_waveform, waveform_to_model_input

DEFAULT_INFERENCE_THRESHOLD = 0.98549
DEFAULT_CONFIDENCE_HUMAN_MAX = 0.30
DEFAULT_CONFIDENCE_AI_MIN = 0.90
DEFAULT_CHUNK_SIZE_SECONDS = 4.0
DEFAULT_CHUNK_OVERLAP_SECONDS = 2.0
DEFAULT_CHUNK_ALERT_THRESHOLD = 0.90
DEFAULT_CHUNK_ALERT_RATIO = 0.75
DEFAULT_VAD_THRESHOLD = 0.50
TARGET_SAMPLE_RATE = 16000


def classify_confidence_band(
    score: float,
    human_max: float = DEFAULT_CONFIDENCE_HUMAN_MAX,
    ai_min: float = DEFAULT_CONFIDENCE_AI_MIN,
) -> str:
    if score < human_max:
        return "High Confidence Human"
    if score <= ai_min:
        return "Unverifiable / Degraded Audio"
    return "High Confidence AI"


@lru_cache(maxsize=1)
def get_vad_components():
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


def apply_vad_to_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    vad_threshold: float = DEFAULT_VAD_THRESHOLD,
) -> tuple[np.ndarray, dict]:
    silero_vad, torch, vad_model = get_vad_components()
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


def split_overlapping_chunks(
    waveform: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    chunk_size_seconds: float = DEFAULT_CHUNK_SIZE_SECONDS,
    chunk_overlap_seconds: float = DEFAULT_CHUNK_OVERLAP_SECONDS,
) -> list[np.ndarray]:
    chunk_size_samples = max(1, int(chunk_size_seconds * sample_rate))
    chunk_overlap_samples = max(0, int(chunk_overlap_seconds * sample_rate))
    step = max(1, chunk_size_samples - chunk_overlap_samples)

    if len(waveform) <= chunk_size_samples:
        return [waveform.astype(np.float32)]

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


def predict_probability_from_waveform(ort_session, waveform: np.ndarray) -> float:
    input_tensor = waveform_to_model_input(waveform)
    onnx_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    onnx_outputs = ort_session.run(None, onnx_inputs)
    return float(onnx_outputs[0][0])


def score_waveforms(ort_session, waveforms: list[np.ndarray]) -> list[float]:
    return [predict_probability_from_waveform(ort_session, waveform) for waveform in waveforms]


def summarize_scores(
    scores: list[float],
    human_max: float = DEFAULT_CONFIDENCE_HUMAN_MAX,
    ai_min: float = DEFAULT_CONFIDENCE_AI_MIN,
    chunk_alert_threshold: float = DEFAULT_CHUNK_ALERT_THRESHOLD,
    chunk_alert_ratio: float = DEFAULT_CHUNK_ALERT_RATIO,
) -> dict:
    if not scores:
        raise HTTPException(status_code=500, detail="No chunk scores were produced for the audio.")

    average_score = sum(scores) / len(scores)
    confidence_band = classify_confidence_band(average_score, human_max, ai_min)
    high_chunk_ratio = sum(score >= chunk_alert_threshold for score in scores) / len(scores)
    video_flagged = high_chunk_ratio > chunk_alert_ratio
    chunk_vote_prediction = "AI-Generated" if video_flagged else "Human Voice"

    return {
        "average_raw_probability": round(average_score, 6),
        "average_ai_probability_score": round(average_score * 100, 2),
        "confidence_band": confidence_band,
        "chunk_vote_prediction": chunk_vote_prediction,
        "chunk_alert_threshold": round(chunk_alert_threshold * 100, 2),
        "chunk_alert_ratio": round(chunk_alert_ratio, 2),
        "chunk_high_score_ratio": round(high_chunk_ratio, 4),
        "video_flagged": video_flagged,
    }


def analyze_waveform_pipeline(
    ort_session,
    waveform: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    *,
    threshold: float = DEFAULT_INFERENCE_THRESHOLD,
    human_max: float = DEFAULT_CONFIDENCE_HUMAN_MAX,
    ai_min: float = DEFAULT_CONFIDENCE_AI_MIN,
    chunk_size_seconds: float = DEFAULT_CHUNK_SIZE_SECONDS,
    chunk_overlap_seconds: float = DEFAULT_CHUNK_OVERLAP_SECONDS,
    chunk_alert_threshold: float = DEFAULT_CHUNK_ALERT_THRESHOLD,
    chunk_alert_ratio: float = DEFAULT_CHUNK_ALERT_RATIO,
    vad_threshold: float = DEFAULT_VAD_THRESHOLD,
) -> dict:
    speech_waveform, vad_summary = apply_vad_to_waveform(waveform, sample_rate, vad_threshold)
    chunks = split_overlapping_chunks(
        speech_waveform,
        sample_rate=sample_rate,
        chunk_size_seconds=chunk_size_seconds,
        chunk_overlap_seconds=chunk_overlap_seconds,
    )
    chunk_scores = score_waveforms(ort_session, chunks)
    summary = summarize_scores(
        chunk_scores,
        human_max=human_max,
        ai_min=ai_min,
        chunk_alert_threshold=chunk_alert_threshold,
        chunk_alert_ratio=chunk_alert_ratio,
    )

    average_probability = summary["average_raw_probability"]
    average_prediction = "AI-Generated" if average_probability >= threshold else "Human Voice"

    return {
        "average_prediction": average_prediction,
        "binary_prediction": summary["chunk_vote_prediction"],
        "decision_threshold": round(threshold * 100, 2),
        "chunk_count": len(chunk_scores),
        "chunk_scores": [round(score, 6) for score in chunk_scores],
        "vad_summary": vad_summary,
        **summary,
    }


def analyze_audio_bytes_pipeline(
    ort_session,
    audio_bytes: bytes,
    *,
    threshold: float = DEFAULT_INFERENCE_THRESHOLD,
    human_max: float = DEFAULT_CONFIDENCE_HUMAN_MAX,
    ai_min: float = DEFAULT_CONFIDENCE_AI_MIN,
    chunk_size_seconds: float = DEFAULT_CHUNK_SIZE_SECONDS,
    chunk_overlap_seconds: float = DEFAULT_CHUNK_OVERLAP_SECONDS,
    chunk_alert_threshold: float = DEFAULT_CHUNK_ALERT_THRESHOLD,
    chunk_alert_ratio: float = DEFAULT_CHUNK_ALERT_RATIO,
    vad_threshold: float = DEFAULT_VAD_THRESHOLD,
) -> dict:
    waveform, sample_rate = load_waveform(audio_bytes)
    return analyze_waveform_pipeline(
        ort_session,
        waveform,
        sample_rate,
        threshold=threshold,
        human_max=human_max,
        ai_min=ai_min,
        chunk_size_seconds=chunk_size_seconds,
        chunk_overlap_seconds=chunk_overlap_seconds,
        chunk_alert_threshold=chunk_alert_threshold,
        chunk_alert_ratio=chunk_alert_ratio,
        vad_threshold=vad_threshold,
    )
