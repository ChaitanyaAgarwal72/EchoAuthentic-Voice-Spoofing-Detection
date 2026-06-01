import io

import librosa
import numpy as np

TARGET_SAMPLE_RATE = 16000
TARGET_AUDIO_SAMPLES = 64000
MEL_BANDS = 128
N_FFT = 1024
HOP_LENGTH = 512
FMAX = 8000


def load_waveform(audio_bytes: bytes, sample_rate: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sample_rate, mono=True)
    return y.astype(np.float32), sr


def pad_or_trim_waveform(y: np.ndarray, max_length: int = TARGET_AUDIO_SAMPLES) -> np.ndarray:
    if len(y) > max_length:
        return y[:max_length]
    if len(y) < max_length:
        return np.pad(y, (0, max_length - len(y))).astype(np.float32)
    return y.astype(np.float32)


def waveform_to_model_input(y: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    y = pad_or_trim_waveform(y)

    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=MEL_BANDS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmax=FMAX,
        power=2.0,
    )

    processed_image = mel_spec[np.newaxis, np.newaxis, :, :]
    return processed_image.astype(np.float32)

def process_audio(audio_bytes: bytes) -> np.ndarray:
    y, sr = load_waveform(audio_bytes)
    return waveform_to_model_input(y, sr)