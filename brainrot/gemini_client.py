"""Thin wrapper around the google-genai SDK.

Kept small and swappable so tests can inject a fake backend without network access.
"""

from __future__ import annotations

import logging
import mimetypes
import time
from pathlib import Path
from typing import Any

from .schema import GeminiVerdict

logger = logging.getLogger(__name__)

RETRYABLE_CODES = {429, 500, 502, 503, 504}


class GeminiBackend:
    def __init__(self, api_key: str, model: str, temperature: float = 0.0):
        if not api_key:
            raise RuntimeError(
                "No Gemini API key found. Set GEMINI_API_KEY (get one at https://aistudio.google.com/apikey)."
            )
        # Imported lazily so the rest of the package (and tests) work without the SDK configured.
        from google import genai
        from google.genai import types

        self._types = types
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature

    # ---------- parts ----------
    def video_file_part(self, path: Path, fps: float | None = None, timeout_s: float = 600) -> tuple[Any, str]:
        """Upload a video via the File API, wait until it's processed, return (part, file_name)."""
        types = self._types
        mime = mimetypes.guess_type(str(path))[0] or "video/mp4"
        logger.info("Uploading %s (%.1f MB) to Gemini File API", path.name, path.stat().st_size / 1e6)
        f = self.client.files.upload(file=str(path), config={"mime_type": mime, "display_name": path.name})
        deadline = time.time() + timeout_s
        while f.state and f.state.name == "PROCESSING":
            if time.time() > deadline:
                raise TimeoutError(f"Gemini did not finish processing {path.name} in {timeout_s}s")
            time.sleep(2)
            f = self.client.files.get(name=f.name)
        if f.state and f.state.name == "FAILED":
            raise RuntimeError(f"Gemini failed to process video: {f.error}")
        part = types.Part(
            file_data=types.FileData(file_uri=f.uri, mime_type=f.mime_type or mime),
            video_metadata=types.VideoMetadata(fps=fps) if fps else None,
        )
        return part, f.name

    def youtube_part(self, url: str, fps: float | None = None) -> Any:
        """Gemini can watch public YouTube videos directly by URL - no download needed."""
        types = self._types
        return types.Part(
            file_data=types.FileData(file_uri=url, mime_type="video/*"),
            video_metadata=types.VideoMetadata(fps=fps) if fps else None,
        )

    def bytes_part(self, data: bytes, mime_type: str) -> Any:
        return self._types.Part.from_bytes(data=data, mime_type=mime_type)

    def text_part(self, text: str) -> Any:
        return self._types.Part.from_text(text=text)

    def delete_file(self, name: str) -> None:
        try:
            self.client.files.delete(name=name)
        except Exception as e:  # best effort; files auto-expire after 48h anyway
            logger.debug("Could not delete uploaded file %s: %s", name, e)

    # ---------- generation ----------
    def classify(self, parts: list[Any], prompt: str, retries: int = 4) -> GeminiVerdict:
        from google.genai import errors

        config = self._types.GenerateContentConfig(
            temperature=self.temperature,
            response_mime_type="application/json",
            response_json_schema=GeminiVerdict.model_json_schema(),
        )
        contents = [self._types.Content(role="user", parts=[*parts, self.text_part(prompt)])]
        delay = 2.0
        for attempt in range(retries + 1):
            try:
                resp = self.client.models.generate_content(model=self.model, contents=contents, config=config)
                if not resp.text:
                    raise RuntimeError(f"Empty response from Gemini (prompt feedback: {resp.prompt_feedback})")
                return GeminiVerdict.model_validate_json(resp.text)
            except errors.APIError as e:
                if e.code == 429 and "PerDay" in str(e):
                    # daily quota: retrying in seconds can't help, so fail fast with a clear message
                    raise RuntimeError("Gemini daily request quota is used up for this API key and model "
                                       "(free tier). It resets in about a day; enable billing or switch "
                                       "GEMINI_MODEL to keep going.") from e
                if e.code in RETRYABLE_CODES and attempt < retries:
                    logger.warning("Gemini error %s, retrying in %.0fs", e.code, delay)
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
        raise RuntimeError("unreachable")
