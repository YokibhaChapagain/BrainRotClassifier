"""Audio is used for ONE thing: Whisper fills in missing transcripts so the TEACHER has more text to read.
The student never hears anything."""

from __future__ import annotations

import subprocess
import tempfile
from functools import lru_cache


def extract_wav(video_path: str) -> str:
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name  # noqa: SIM115
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_path, "-ac", "1", "-ar", "16000", out],
                   check=True)
    return out


@lru_cache(maxsize=1)
def _whisper():
    import whisper

    return whisper.load_model("small")


def transcribe(video_path: str) -> str:
    return _whisper().transcribe(extract_wav(video_path), fp16=False)["text"].strip()
