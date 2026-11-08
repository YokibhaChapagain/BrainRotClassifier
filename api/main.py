"""FastAPI service behind the Chrome extension and the dashboard.

POST /predict       frames in -> score + reasons out (the student; pixels only)
POST /explain       Gemini turns the reasons + 2 frames into a playful one-liner (optional)
POST /events/view   log one watched reel to Tiger Data
GET  /stats/*       dashboard data
GET  /dashboard     analytics page
"""

from __future__ import annotations

import base64
import io
import logging
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api import db
from brainrot.explain import build_reasons, summary_line
from brainrot.features.concepts import CONCEPT_NAMES
from brainrot.features.video import motion_features
from brainrot.models.student import PixelRow, StudentModel

load_dotenv()
log = logging.getLogger("brainrot")
app = FastAPI(title="BrainRot Meter API", version="0.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

THRESHOLD = float(os.getenv("FLAG_THRESHOLD", "0.6"))
USE_CLIP = os.getenv("USE_CLIP", "1") == "1"  # needs sentence-transformers; set 0 for a motion-only server
STATIC = Path(__file__).parent / "static"
# Placeholder "typical" values for the heuristic fallback; the trained model carries real dataset medians.
FALLBACK_REFERENCE = {"cuts_per_sec": 0.5, "motion_energy": 0.05, "split_screen": 0.3, "saturation": 0.35,
                      "brightness": 0.5, "colorfulness": 0.35, "flicker": 0.02}
_cache: dict[tuple[str, int], PredictResponse] = {}


class PredictRequest(BaseModel):
    video_id: str
    platform: Literal["tiktok", "instagram"] = "tiktok"
    frames_b64: list[str] = Field(default_factory=list, description="JPEG frames (base64), ~2 per second")
    cuts_per_sec: float | None = Field(None, description="Measured in-page by the extension, if possible")


class Concept(BaseModel):
    name: str
    p: float


class PredictResponse(BaseModel):
    video_id: str
    prob: float = Field(description="Calibrated probability of brain rot, 0-1 (the meter)")
    flagged: bool = Field(description="prob >= threshold (the 🚩)")
    label: Literal["brainrot", "not_brainrot"]
    threshold: float
    reasons: list[str] = Field(description="Most important first, built from measured pixel features")
    summary: str = Field(description='e.g. "0.87 · looks like Minecraft gameplay + split screen · '
                                     '3× the typical cut rate"')
    top_concepts: list[Concept]
    pace: dict[str, float] = Field(description="Raw motion features + cut rate relative to typical")
    n_frames: int
    final: bool = Field(description="False for the early 4-frame estimate; True once 8 frames are in")
    model_version: str


@lru_cache(maxsize=1)
def get_model() -> StudentModel | None:
    path = os.getenv("MODEL_PATH", "artifacts/student.joblib")
    if os.path.exists(path):
        return StudentModel.load(path)
    log.warning("No model at %s — using heuristic fallback", path)
    return None


def decode_frames(frames_b64: list[str]) -> list[np.ndarray]:
    from PIL import Image

    return [np.asarray(Image.open(io.BytesIO(base64.b64decode(b.split(",")[-1]))).convert("RGB"))
            for b in frames_b64[:8]]


def heuristic_prob(m: dict[str, float], ref: dict[str, float]) -> float:
    """Keeps the demo alive before the student is trained. Not a model."""
    z = (-1.0 + 0.8 * (m["cuts_per_sec"] / ref["cuts_per_sec"] - 1) + 2.0 * max(0.0, m["split_screen"] - 0.4)
         + 1.5 * (m["colorfulness"] / ref["colorfulness"] - 1) + 10 * (m["motion_energy"] - ref["motion_energy"]))
    return float(1 / (1 + np.exp(-z)))


@app.get("/health")
def health():
    return {"ok": True, "model_loaded": get_model() is not None, "clip": USE_CLIP, "db": db.enabled()}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    key = (req.video_id, len(req.frames_b64))
    if key in _cache:
        return _cache[key]
    frames = decode_frames(req.frames_b64)
    motion = motion_features(frames, req.cuts_per_sec or 0.0)
    concepts = clip_emb = None
    if USE_CLIP and frames:
        from brainrot.features.video import clip_features

        clip_emb, concepts = clip_features(frames)
    row = PixelRow(motion=motion, concepts=concepts, clip_emb=clip_emb)

    model = get_model()
    if model is not None:
        prob, thr, version, ref = float(model.predict_proba([row])[0]), model.threshold, model.version, model.reference
        contrib = model.contributions(row)
    else:
        ref = FALLBACK_REFERENCE
        prob, thr, version, contrib = heuristic_prob(motion, ref), THRESHOLD, "heuristic-v0", {}

    flagged = prob >= thr
    reasons = build_reasons(motion, concepts, ref, contrib, flagged)
    top = [] if concepts is None else [
        Concept(name=CONCEPT_NAMES[i], p=round(float(concepts[i]), 3)) for i in np.argsort(-concepts)[:3]]
    resp = PredictResponse(
        video_id=req.video_id, prob=round(prob, 4), flagged=flagged,
        label="brainrot" if flagged else "not_brainrot", threshold=thr,
        reasons=[r["text"] for r in reasons], summary=summary_line(prob, reasons), top_concepts=top,
        pace={**{k: round(v, 4) for k, v in motion.items()},
              "cut_rate_vs_typical": round(motion["cuts_per_sec"] / ref["cuts_per_sec"], 2)
              if ref["cuts_per_sec"] else 0.0},
        n_frames=len(frames), final=len(frames) >= 8, model_version=version,
    )
    _cache[key] = resp
    return resp


class ExplainRequest(BaseModel):
    prob: float
    reasons: list[str] = Field(default_factory=list)
    frames_b64: list[str] = Field(default_factory=list, description="1-2 JPEG frames")


@app.post("/explain")
def explain(req: ExplainRequest):
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(503, "GEMINI_API_KEY not set")
    from brainrot.labeling.gemini_labeler import GeminiTeacher

    frames = [base64.b64decode(b.split(",")[-1]) for b in req.frames_b64[:2]]
    return {"explanation": GeminiTeacher().narrate(req.prob, req.reasons, frames)}


class ViewEvent(BaseModel):
    user_id: str
    session_id: str
    video_id: str
    platform: Literal["tiktok", "instagram"] = "tiktok"
    started_at: datetime
    watch_ms: int = Field(ge=0)
    video_duration_ms: int | None = None
    loops: int = 0
    skipped: bool = False
    prob: float | None = None
    flagged: bool | None = None


@app.post("/events/view")
def log_view(ev: ViewEvent):
    db.log_view(ev.model_dump())
    return {"ok": True, "stored": db.enabled()}


# ------------------------------------------------------------------ dashboard
@app.get("/stats/summary")
def stats_summary(user_id: str, days: int = Query(7, ge=1, le=90)):
    return db.summary(user_id, days)


@app.get("/stats/timeline")
def stats_timeline(user_id: str, days: int = Query(7, ge=1, le=90), bucket: Literal["hour", "day"] = "hour"):
    return db.timeline(user_id, days, bucket)


@app.get("/stats/heatmap")
def stats_heatmap(user_id: str, days: int = Query(30, ge=1, le=90)):
    return db.heatmap(user_id, days)


@app.get("/stats/sessions")
def stats_sessions(user_id: str, days: int = Query(7, ge=1, le=90)):
    return db.sessions(user_id, days)


@app.get("/stats/engagement")
def stats_engagement():
    return db.engagement()


@app.get("/stats/trend")
def stats_trend():
    return db.trend()


@app.get("/dashboard")
def dashboard():
    return FileResponse(STATIC / "dashboard.html")
