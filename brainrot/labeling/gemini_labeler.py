"""Gemini, two jobs:

1. TEACHER (offline): reads a video's TEXT ONLY (caption, hashtags, on-screen text, transcript) and returns
   a 0-10 brain rot score. It never sees pixels, so the student has to learn the visual signal on its own.
2. NARRATOR (live, optional): turns the student's measured reasons + a couple of frames into a playful
   one-liner for the overlay. It doesn't change the score.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

from brainrot.labeling.rubric import BRAINROT_THRESHOLD, TEACHER_PROMPT
from brainrot.schema import Reel


@dataclass
class TeacherLabel:
    score: int
    label: int
    reasons: list[str]
    confidence: float
    enough_info: bool
    raw: str = ""


def parse_label(text: str) -> TeacherLabel:
    """Robustly pull the JSON object out of a model response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON in response: {text[:200]!r}")
    data = json.loads(match.group(0))
    score = max(0, min(10, int(data.get("score", 5))))
    return TeacherLabel(
        score=score,
        label=int(score >= BRAINROT_THRESHOLD),
        reasons=[str(r) for r in data.get("reasons", [])][:3],
        confidence=float(data.get("confidence", 0.5)),
        enough_info=bool(data.get("enough_info", True)),
        raw=text,
    )


class GeminiTeacher:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        from google import genai  # lazy import

        self.client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def label(self, reel: Reel, retries: int = 3) -> TeacherLabel:
        from google.genai import types

        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=[TEACHER_PROMPT, reel.teacher_text()[:6000]],
                    config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
                )
                return parse_label(resp.text)
            except Exception as e:  # noqa: BLE001  rate limits, bad JSON
                last_err = e
                time.sleep(2**attempt)
        raise RuntimeError(f"Gemini labeling failed for {reel.video_id}") from last_err

    def narrate(self, prob: float, reasons: list[str], frames_jpeg: list[bytes]) -> str:
        """One playful line (<= 15 words) built from the student's reasons and 2 frames."""
        from google.genai import types

        parts: list = [
            f"A video classifier scored this reel {prob:.0%} brain rot because: {'; '.join(reasons)}. "
            "Look at the frames and write ONE playful Gen-Z line (max 15 words) explaining the score. "
            "Don't invent things you can't see."
        ]
        parts += [types.Part.from_bytes(data=f, mime_type="image/jpeg") for f in frames_jpeg[:2]]
        return self.client.models.generate_content(model=self.model, contents=parts).text.strip()
