"""Minimal web UI: upload a video or paste a URL, get a verdict."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
from importlib import resources
from pathlib import Path

import base64
import binascii
from collections import OrderedDict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .. import feed
from ..classifier import BrainRotClassifier
from ..config import Settings

logger = logging.getLogger(__name__)
app = FastAPI(title="Brain Rot Detector")
MAX_UPLOAD_MB = 500
MAX_FRAMES = 16
CACHE_SIZE = 2000
_cache: OrderedDict[tuple, dict] = OrderedDict()


class PredictRequest(BaseModel):
    """What the browser extension sends for each video."""

    video_id: str
    platform: str = "instagram"
    frames_b64: list[str] = Field(default_factory=list)
    cuts_per_sec: float | None = None


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return resources.files("brainrot.web").joinpath("templates/index.html").read_text()


@app.get("/feed", response_class=HTMLResponse)
def feed_page() -> str:
    return resources.files("brainrot.web").joinpath("templates/feed.html").read_text()


@app.get("/api/feed")
def feed_list() -> dict:
    """Videos with their stored verdicts. Reads feed.json only - no model call happens here."""
    folder = feed.feed_dir()
    items, pending = feed.feed_items(folder)
    return {"folder": str(folder), "items": items, "pending": pending}


@app.post("/api/feed/upload")
async def feed_upload(file: UploadFile = File(...)) -> dict:
    """Add a video to the feed: save it, have Gemini classify it once, store the verdict.

    This is the "platform" side: the one time a video is classified is when it's uploaded."""
    folder = feed.feed_dir()
    folder.mkdir(parents=True, exist_ok=True)
    try:
        clf = BrainRotClassifier(Settings())
    except (RuntimeError, ValueError) as e:
        raise HTTPException(400, str(e))
    name = feed.safe_name(folder, file.filename or "")
    part = folder / f".{name}.part"  # hidden until classified, so the feed never lists it half-done
    try:
        size = 0
        with part.open("wb") as out:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_MB * 1_000_000:
                    raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
                out.write(chunk)
        if size == 0:
            raise HTTPException(400, "The file is empty.")
        dest = folder / feed.safe_name(folder, name)  # re-check: another upload may have taken the name
        part.replace(dest)
        part = dest
        try:
            r = await run_in_threadpool(clf.classify, dest)
        except Exception as e:
            logger.exception("Upload classification failed")
            raise HTTPException(502, f"Gemini couldn't classify this video: {e}")
        entry = feed.make_entry(dest, r, uploaded_at=time.time())
        feed.put_entry(folder, dest.name, entry)
        part = None  # keep it
        return feed.to_item(dest.name, entry)
    finally:
        if part is not None:
            part.unlink(missing_ok=True)  # failed upload: don't leave an unclassified file behind


@app.get("/media/{name}")
def media(name: str) -> FileResponse:
    # only serve videos that are listed in the feed folder (no path tricks)
    folder = feed.feed_dir()
    match = next((p for p in feed.list_videos(folder) if p.name == name), None)
    if match is None:
        raise HTTPException(404, "No such video")
    return FileResponse(match)  # supports Range requests, so the player can seek


@app.get("/api/config")
def config() -> dict:
    s = Settings()
    return {"model": s.model, "mode": s.mode, "threshold": s.threshold, "has_api_key": bool(s.api_key)}


@app.get("/health")
def health() -> dict:
    s = Settings()
    return {"ok": True, "model": s.model, "threshold": s.threshold, "has_api_key": bool(s.api_key)}


@app.post("/predict")
async def predict(req: PredictRequest) -> dict:
    """Score one video from the extension's screenshots. Same video asked twice = cached answer."""
    if not req.frames_b64:
        raise HTTPException(400, "frames_b64 is empty")
    key = (req.platform, req.video_id)
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    try:
        frames = [base64.b64decode(f.split(",")[-1], validate=True) for f in req.frames_b64[:MAX_FRAMES]]
    except (binascii.Error, ValueError):
        raise HTTPException(400, "frames_b64 must be base64 JPEGs")
    try:
        clf = BrainRotClassifier(Settings())
    except (RuntimeError, ValueError) as e:
        raise HTTPException(400, str(e))
    try:
        r = await run_in_threadpool(clf.classify_frames, frames, f"{req.platform}:{req.video_id}")
    except Exception as e:
        logger.exception("Frame classification failed")
        raise HTTPException(502, f"{type(e).__name__}: {e}")
    return _remember(key, _extension_answer(r, req.video_id, req.platform))


@app.post("/predict_clip")
async def predict_clip(
    file: UploadFile = File(...),
    video_id: str = Form(...),
    platform: str = Form("instagram"),
):
    """Score one video from a clip the extension recorded off the playing <video> element.

    The extension records what the browser is already playing (no extra requests to the site), so the
    model gets real motion and audio instead of 8 screenshots. Runs the normal native/frames pipeline,
    so BRAINROT_MODE and BRAINROT_MAX_FRAMES apply here."""
    key = (platform, video_id, "clip")
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    try:
        clf = BrainRotClassifier(Settings())
    except (RuntimeError, ValueError) as e:
        raise HTTPException(400, str(e))

    tmpdir = Path(tempfile.mkdtemp(prefix="brainrot_clip_"))
    try:
        raw = tmpdir / ("clip" + (Path(file.filename or "").suffix or ".webm"))
        size = 0
        with raw.open("wb") as out:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_MB * 1_000_000:
                    raise HTTPException(413, f"Clip larger than {MAX_UPLOAD_MB} MB")
                out.write(chunk)
        if size == 0:
            raise HTTPException(400, "clip is empty")
        # MediaRecorder webm has no duration/frame count in its header, which breaks OpenCV's
        # frame counting, so normalise to a plain mp4 first.
        clip = await run_in_threadpool(_to_mp4, raw, tmpdir / "clip.mp4")
        r = await run_in_threadpool(clf.classify, clip)
        r.source = f"{platform}:{video_id}"
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Clip classification failed")
        raise HTTPException(502, f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    out = _extension_answer(r, video_id, platform)
    out["mode"] = r.mode
    out["signals"] = r.signals.model_dump() if r.signals else None
    return _remember(key, out)


def _to_mp4(src: Path, dst: Path) -> Path:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to read recorded clips: brew install ffmpeg")
    r = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-movflags", "+faststart", str(dst)],
                       capture_output=True, text=True, timeout=180)
    if r.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        raise ValueError(f"could not read the recorded clip: {r.stderr.strip()[-300:]}")
    return dst


def _remember(key: tuple, out: dict) -> dict:
    _cache[key] = out
    if len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)
    return out


def _extension_answer(r, video_id: str, platform: str) -> dict:
    return {
        "video_id": video_id,
        "platform": platform,
        "prob": round(r.brain_rot_score / 100, 2),
        "flagged": r.is_brain_rot,
        "label": r.label,
        "score": r.brain_rot_score,
        "threshold": r.threshold,
        "confidence": r.confidence,
        "reasons": r.verdict.detected_tropes[:3],
        "summary": r.verdict.reasoning,
        "model": r.model,
        "elapsed_s": r.elapsed_s,
    }


@app.post("/api/classify")
async def classify(
    file: UploadFile | None = File(None),
    url: str = Form(""),
    mode: str = Form("native"),
    threshold: int = Form(50),
    context: str = Form(""),
):
    if not file and not url.strip():
        raise HTTPException(400, "Upload a video or provide a URL.")
    settings = Settings()
    settings.mode = mode
    settings.threshold = threshold
    try:
        clf = BrainRotClassifier(settings)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(400, str(e))

    tmpdir = Path(tempfile.mkdtemp(prefix="brainrot_"))
    try:
        if file and file.filename:
            dest = tmpdir / Path(file.filename).name
            with dest.open("wb") as out:
                size = 0
                while chunk := await file.read(1 << 20):
                    size += len(chunk)
                    if size > MAX_UPLOAD_MB * 1_000_000:
                        raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
                    out.write(chunk)
            result = await run_in_threadpool(clf.classify, dest, context)
            result.source = file.filename
        else:
            result = await run_in_threadpool(clf.classify, url.strip(), context)
        return result.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Classification failed")
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
