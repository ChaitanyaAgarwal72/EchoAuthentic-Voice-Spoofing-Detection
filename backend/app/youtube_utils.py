from __future__ import annotations

import importlib
import os
import tempfile
from urllib.parse import urlparse

from fastapi import HTTPException

ALLOWED_YOUTUBE_HOSTNAMES = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
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
