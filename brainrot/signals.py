"""Local, deterministic video analysis (no LLM).

Two jobs:
1. Measure objective pacing signals (scene cuts/minute, motion, loudness). Gemini samples video at
   ~1-2 fps so it under-counts fast editing - these numbers are handed to it as extra evidence.
2. Pick keyframes for "frames" mode. Inspired by byjlw/video-analyzer's difference-based selection,
   but split into equal segments so the whole video is always covered.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .schema import LocalSignals

logger = logging.getLogger(__name__)

ANALYSIS_FPS = 8.0          # how densely we sample for pacing analysis
ANALYSIS_WIDTH = 160        # downscale before comparing frames (speed)
CUT_HIST_THRESHOLD = 0.55   # histogram correlation below this = hard cut
CUT_DIFF_THRESHOLD = 40.0   # ...or mean abs pixel diff above this
KEYFRAME_MAX_SIDE = 768


@dataclass
class Keyframe:
    index: int
    timestamp: float
    jpeg: bytes


@dataclass
class _Sample:
    frame_idx: int
    timestamp: float
    diff: float


def _small_gray_and_hist(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (ANALYSIS_WIDTH, max(1, int(h * ANALYSIS_WIDTH / w))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return gray, hist


def _probe_audio(path: Path) -> tuple[bool, float | None]:
    """Return (has_audio, mean_volume_db) using ffprobe/ffmpeg if available."""
    if not shutil.which("ffprobe") or not shutil.which("ffmpeg"):
        return False, None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout
        if not json.loads(out or "{}").get("streams"):
            return False, None
        err = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn", "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True, text=True, timeout=300,
        ).stderr
        for line in err.splitlines():
            if "mean_volume:" in line:
                return True, float(line.split("mean_volume:")[1].split("dB")[0].strip())
        return True, None
    except (subprocess.SubprocessError, ValueError, json.JSONDecodeError) as e:
        logger.warning("Audio probe failed: %s", e)
        return False, None


def analyze_video(path: Path, max_seconds: float = 0, max_frames: int = 16,
                  extract_keyframes: bool = False) -> tuple[LocalSignals, list[Keyframe]]:
    """Single sequential pass for pacing stats, then seek to grab keyframes if requested."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if max_seconds and max_seconds > 0:
            total = min(total, int(max_seconds * fps))
        step = max(1, round(fps / ANALYSIS_FPS))

        samples: list[_Sample] = []
        cuts = 0
        motion: list[float] = []
        prev_gray = prev_hist = None
        idx = 0
        while idx < total:
            # grab() is much cheaper than read() for frames we skip
            if idx % step != 0:
                if not cap.grab():
                    break
                idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            gray, hist = _small_gray_and_hist(frame)
            diff = 0.0
            if prev_gray is not None:
                diff = float(np.mean(cv2.absdiff(gray, prev_gray)))
                corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if corr < CUT_HIST_THRESHOLD or diff > CUT_DIFF_THRESHOLD:
                    cuts += 1
                else:
                    motion.append(diff)  # motion within a shot, not across cuts
            samples.append(_Sample(idx, idx / fps, diff))
            prev_gray, prev_hist = gray, hist
            idx += 1

        duration = (samples[-1].timestamp + step / fps) if samples else 0.0
        has_audio, volume = _probe_audio(path)
        signals = LocalSignals(
            duration_s=round(duration, 2),
            fps=round(fps, 2),
            width=width,
            height=height,
            is_vertical=height > width,
            scene_cuts=cuts,
            cuts_per_minute=round(cuts / (duration / 60), 1) if duration > 0 else 0.0,
            mean_motion=round(float(np.mean(motion)) if motion else 0.0, 2),
            has_audio=has_audio,
            mean_volume_db=volume,
        )

        keyframes: list[Keyframe] = []
        if extract_keyframes and samples:
            keyframes = _grab_keyframes(cap, samples, max_frames)
        return signals, keyframes
    finally:
        cap.release()


def _grab_keyframes(cap: cv2.VideoCapture, samples: list[_Sample], max_frames: int) -> list[Keyframe]:
    """Split the timeline into max_frames segments; in each, take the most 'changed' frame."""
    n = max(1, min(max_frames, len(samples)))
    bounds = np.linspace(0, len(samples), n + 1).astype(int)
    chosen = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        segment = samples[a:b] or samples[a:a + 1]
        chosen.append(max(segment, key=lambda s: s.diff))

    keyframes = []
    for i, s in enumerate(chosen):
        cap.set(cv2.CAP_PROP_POS_FRAMES, s.frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        scale = KEYFRAME_MAX_SIDE / max(h, w)
        if scale < 1:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            keyframes.append(Keyframe(i, s.timestamp, buf.tobytes()))
    return keyframes


def extract_audio(path: Path, out_path: Path, max_seconds: float = 0) -> Path | None:
    """Extract a small mono mp3 for 'frames' mode. Returns None if no audio / no ffmpeg."""
    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg not found - audio will not be analysed in frames mode")
        return None
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(path), "-vn",
           "-ac", "1", "-ar", "16000", "-b:a", "32k"]
    if max_seconds and max_seconds > 0:
        cmd += ["-t", str(max_seconds)]
    cmd.append(str(out_path))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size < 1000:
        return None
    return out_path
