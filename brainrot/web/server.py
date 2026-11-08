"""Minimal web UI: upload a video or paste a URL, get a verdict."""

from __future__ import annotations

import logging
import shutil
import tempfile
from importlib import resources
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse

from ..classifier import BrainRotClassifier
from ..config import Settings

logger = logging.getLogger(__name__)
app = FastAPI(title="Brain Rot Detector")
MAX_UPLOAD_MB = 500


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return resources.files("brainrot.web").joinpath("templates/index.html").read_text()


@app.get("/api/config")
def config() -> dict:
    s = Settings()
    return {"model": s.model, "mode": s.mode, "threshold": s.threshold, "has_api_key": bool(s.api_key)}


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
