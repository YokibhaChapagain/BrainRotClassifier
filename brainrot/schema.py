"""Reel record shared by the pipeline and the API.

Text fields are for the TEACHER only (Gemini reads them to make labels).
The STUDENT never sees them — it only gets frames.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


def _maybe_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return default
    return value


@dataclass
class Reel:
    video_id: str
    creator: str = ""
    description: str = ""
    transcript: str = ""
    sticker_text: str = ""  # on-screen text
    hashtags: list[str] = field(default_factory=list)
    duration_ms: int | None = None
    is_ai_generated: bool | None = None
    plays: int | None = None
    likes: int | None = None
    n_comments: int | None = None
    shares: int | None = None
    video_path: str | None = None

    @classmethod
    def from_hf_row(cls, row: dict[str, Any]) -> Reel:
        """Map a row of Shofo/shofo-tiktok-general-small to a Reel."""
        tags = _maybe_json(row.get("hashtags"), [])
        eng = _maybe_json(row.get("engagement_metrics"), {}) or {}
        return cls(
            video_id=str(row.get("video_id")),
            creator=str(row.get("creator") or ""),
            description=row.get("description") or "",
            transcript=row.get("transcript") or "",
            sticker_text=row.get("sticker_text") or "",
            hashtags=[str(t).lstrip("#") for t in tags] if isinstance(tags, list) else [],
            duration_ms=row.get("duration_ms"),
            is_ai_generated=row.get("is_ai_generated"),
            plays=eng.get("play_count") or eng.get("plays"),
            likes=eng.get("like_count") or eng.get("likes"),
            n_comments=eng.get("comment_count") or eng.get("comments"),
            shares=eng.get("share_count") or eng.get("shares"),
            video_path=row.get("video_path"),
        )

    def teacher_text(self) -> str:
        """Everything the teacher reads: caption, hashtags, on-screen text, transcript."""
        parts = [
            f"CAPTION: {self.description}" if self.description else "",
            "HASHTAGS: " + " ".join(f"#{h}" for h in self.hashtags) if self.hashtags else "",
            f"ON-SCREEN TEXT: {self.sticker_text}" if self.sticker_text else "",
            f"TRANSCRIPT: {self.transcript}" if self.transcript else "",
        ]
        return "\n".join(p for p in parts if p).strip()

    def has_rich_text(self, min_chars: int = 40) -> bool:
        """Teacher only labels videos with enough real text (transcript or on-screen text),
        because a caption like '#fyp #viral' alone tells it nothing."""
        return len(self.transcript.strip()) >= min_chars or len(self.sticker_text.strip()) >= min_chars
