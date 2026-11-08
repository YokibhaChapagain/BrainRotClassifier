"""Structured output schema shared by the Gemini call and the rest of the app."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CriterionScores(BaseModel):
    """Each criterion is scored 0 (absent) to 10 (extreme)."""

    overstimulation: int = Field(ge=0, le=10, description="Rapid cuts, flashing, loud SFX, zooms, constant captions.")
    low_informational_value: int = Field(ge=0, le=10, description="Little to learn, no real point, filler.")
    incoherence_absurdism: int = Field(ge=0, le=10, description="Nonsensical, random, surreal for its own sake.")
    meme_slang_density: int = Field(ge=0, le=10, description="Skibidi/rizz/gyatt/sigma/Ohio-style slang or recycled meme audio.")
    split_screen_or_filler_gameplay: int = Field(ge=0, le=10, description="Subway-surfer/parkour/satisfying clips stacked under unrelated content.")
    ai_slop_or_tts: int = Field(ge=0, le=10, description="Low-effort AI-generated imagery, AI characters, robotic TTS narration.")
    engagement_bait: int = Field(ge=0, le=10, description="'Wait for it', fake suspense, ragebait, follow-for-part-2.")


class GeminiVerdict(BaseModel):
    """What we ask Gemini to return (enforced via response_json_schema)."""

    content_summary: str = Field(description="2-3 sentence neutral description of what the video shows and says.")
    criteria: CriterionScores
    detected_tropes: list[str] = Field(description="Specific brain rot tropes observed, e.g. 'Subway Surfers split-screen', 'TTS Reddit story'.")
    redeeming_qualities: list[str] = Field(description="Things that argue AGAINST brain rot (educational, skilled, coherent narrative, ...).")
    brain_rot_score: int = Field(ge=0, le=100, description="Overall brain rot score, 0 = clearly not, 100 = extreme brain rot.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the score given what could be observed.")
    reasoning: str = Field(description="Short justification citing concrete evidence (timestamps where possible).")


class LocalSignals(BaseModel):
    """Objective measurements computed locally with OpenCV/ffmpeg."""

    duration_s: float
    fps: float
    width: int
    height: int
    is_vertical: bool
    scene_cuts: int
    cuts_per_minute: float
    mean_motion: float
    has_audio: bool
    mean_volume_db: float | None = None


class ClassificationResult(BaseModel):
    source: str
    label: Literal["brain_rot", "not_brain_rot"]
    is_brain_rot: bool
    brain_rot_score: int
    threshold: int
    confidence: float
    mode: str
    model: str
    verdict: GeminiVerdict
    signals: LocalSignals | None = None
    elapsed_s: float
