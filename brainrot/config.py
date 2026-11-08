"""Runtime settings, read from environment variables (or a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader so we don't need python-dotenv. Existing env vars win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


@dataclass
class Settings:
    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
    # "native": upload the whole video (frames + audio) to Gemini.
    # "frames": extract keyframes + audio locally and send only those (cheaper, works offline-ish).
    mode: str = field(default_factory=lambda: os.getenv("BRAINROT_MODE", "native"))
    # Score (0-100) at or above which a video is labelled brain rot.
    threshold: int = field(default_factory=lambda: int(os.getenv("BRAINROT_THRESHOLD", "50")))
    # Frames per second Gemini samples in native mode (Gemini default is 1.0; brain rot is fast,
    # so we sample a little denser). Valid range (0, 24].
    native_fps: float = field(default_factory=lambda: float(os.getenv("BRAINROT_NATIVE_FPS", "2")))
    # Number of keyframes sent in "frames" mode.
    max_frames: int = field(default_factory=lambda: int(os.getenv("BRAINROT_MAX_FRAMES", "16")))
    # Only analyse the first N seconds (0 = whole video). Keeps cost bounded for long videos.
    max_seconds: float = field(default_factory=lambda: float(os.getenv("BRAINROT_MAX_SECONDS", "180")))
    temperature: float = 0.0
