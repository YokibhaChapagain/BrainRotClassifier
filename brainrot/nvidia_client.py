"""Backend for NVIDIA's API catalog (build.nvidia.com), an OpenAI-style chat completions endpoint.

Drop-in replacement for GeminiBackend. Differences that matter:
- Only images and text go in (no video upload, no audio), so the classifier always sends keyframes
  (frames mode) and leaves the audio track out. The local pacing signals still go in the prompt.
- There's no schema-enforced output. The model reasons first (more accurate), then we parse the JSON
  out of its answer; if that fails we retry once with the API's JSON mode.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from .schema import GeminiVerdict

logger = logging.getLogger(__name__)

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
RETRYABLE_CODES = {429, 500, 502, 503, 504}
MAX_TOKENS = 8192  # room for the model's reasoning plus the JSON answer


class NvidiaBackend:
    supports_video = False  # no video upload: the classifier sends keyframes instead
    supports_audio = False

    def __init__(self, api_key: str, model: str, temperature: float = 0.0, timeout_s: float = 240):
        if not api_key:
            raise RuntimeError("No NVIDIA API key found. Set NVIDIA_API_KEY (get one at https://build.nvidia.com).")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.http = httpx.Client(timeout=timeout_s)

    # ---------- parts (OpenAI-style content items) ----------
    def video_file_part(self, path: Path, fps: float | None = None) -> tuple[Any, str]:
        raise RuntimeError("The NVIDIA backend can't take whole videos; use frames mode.")

    def youtube_part(self, url: str, fps: float | None = None) -> Any:
        raise RuntimeError("The NVIDIA backend can't watch YouTube URLs; the video is downloaded instead.")

    def bytes_part(self, data: bytes, mime_type: str) -> Any:
        if not mime_type.startswith("image/"):
            raise ValueError(f"The NVIDIA backend only accepts images, not {mime_type}")
        return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64.b64encode(data).decode()}"}}

    def text_part(self, text: str) -> Any:
        return {"type": "text", "text": text}

    def delete_file(self, name: str) -> None:
        pass  # nothing is uploaded

    # ---------- generation ----------
    def classify(self, parts: list[Any], prompt: str, retries: int = 4) -> GeminiVerdict:
        schema = json.dumps(GeminiVerdict.model_json_schema(), separators=(",", ":"))
        prompt = (f"{prompt}\n\nThe JSON must match this JSON Schema exactly (all fields required, "
                  f"integers within their ranges):\n{schema}\n"
                  f"Reply with the JSON object only: no markdown fences, no text before or after it.")
        messages = [{"role": "user", "content": [*parts, self.text_part(prompt)]}]

        text = self._complete(messages, retries=retries)
        try:
            return _parse_verdict(text)
        except (ValueError, ValidationError) as e:
            logger.warning("Model answer wasn't valid verdict JSON (%s); retrying in JSON mode", str(e)[:200])
        # second try: force JSON output and show the model what went wrong
        messages += [{"role": "assistant", "content": text[-4000:]},
                     {"role": "user", "content": "That was not valid JSON for the schema. Reply again with only the "
                                                 "corrected JSON object."}]
        return _parse_verdict(self._complete(messages, retries=retries, json_mode=True))

    def _complete(self, messages: list[dict], retries: int, json_mode: bool = False) -> str:
        body = {"model": self.model, "messages": messages, "max_tokens": MAX_TOKENS,
                "temperature": self.temperature, "top_p": 0.95, "stream": False}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        delay = 2.0
        for attempt in range(retries + 1):
            try:
                r = self.http.post(API_URL, json=body, headers=headers)
            except httpx.TransportError as e:
                if attempt < retries:
                    logger.warning("NVIDIA API connection error (%s), retrying in %.0fs", e, delay)
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(f"Couldn't reach the NVIDIA API: {e}") from e
            if r.status_code in RETRYABLE_CODES and attempt < retries:
                logger.warning("NVIDIA API error %s, retrying in %.0fs", r.status_code, delay)
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code != 200:
                raise RuntimeError(f"NVIDIA API error {r.status_code}: {r.text[:300]}")
            choice = r.json()["choices"][0]
            text = (choice.get("message") or {}).get("content") or ""
            if not text.strip():
                raise RuntimeError(f"Empty answer from the NVIDIA API (finish_reason: {choice.get('finish_reason')})")
            return text
        raise RuntimeError("unreachable")


def _parse_verdict(text: str) -> GeminiVerdict:
    """Pull the JSON object out of a model answer (tolerates code fences and stray text around it)."""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t).strip()
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the answer")
    return GeminiVerdict.model_validate_json(t[start:end + 1])
