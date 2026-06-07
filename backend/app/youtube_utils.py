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

MAX_YOUTUBE_DURATION_SECONDS = 300 

def _match_filter(info_dict: dict, *, incomplete: bool) -> str | None:
    """yt-dlp match_filter hook — called before any download begins.

    Returning a non-None string causes yt-dlp to skip the video with that
    message, which we turn into an HTTP 400 error in the caller.
    """
    duration = info_dict.get("duration")  # seconds, may be None for live streams
    if duration is None:
        return "Cannot determine video duration (may be a live stream)."
    if duration > MAX_YOUTUBE_DURATION_SECONDS:
        mins = MAX_YOUTUBE_DURATION_SECONDS // 60
        return (
            f"Video is {duration / 60:.1f} min long — "
            f"only videos up to {mins} minutes are supported."
        )
    return None  # allow download


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
            "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
            "outtmpl": output_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "ffmpeg_location": ffmpeg_path,
            "legacyserverconnect": True,
            "source_address": "0.0.0.0",
            "match_filter": _match_filter,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
        }

        try:
            from yt_dlp.networking.impersonate import ImpersonateTarget
            ydl_options["impersonate"] = ImpersonateTarget(client="chrome")
        except ImportError:
            pass

        # Securely handle cookies if provided via Hugging Face Secrets
        cookies_env = os.environ.get("YOUTUBE_COOKIES")
        if cookies_env:
            cookies_path = os.path.join(temp_dir, "cookies.txt")
            with open(cookies_path, "w") as f:
                f.write(cookies_env)
            ydl_options["cookiefile"] = cookies_path

        with yt_dlp.YoutubeDL(ydl_options) as ydl:
            info = ydl.extract_info(url, download=False)  # metadata first
            # match_filter runs during extract_info; check result before download
            if info is None:
                raise HTTPException(
                    status_code=400,
                    detail="Video was rejected: could not retrieve metadata.",
                )
            skip_reason = _match_filter(info, incomplete=False)
            if skip_reason:
                raise HTTPException(status_code=400, detail=skip_reason)
            # Metadata passed — proceed with download
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
