"""The STUDENT's eyes: everything here is computed from pixels only.

Two feature groups (used for the ablation "motion-only vs motion + CLIP"):

* MOTION (pace & style): cuts/sec, motion energy, split-screen score, colour intensity
  (saturation, brightness, colourfulness) and brightness flicker.
* CLIP (content): mean CLIP image embedding of the frames + similarity to a bank of plain-English concepts
  ("Minecraft gameplay", "someone cooking", ...).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from brainrot.features.concepts import CONCEPT_PROMPTS

CLIP_MODEL = "clip-ViT-B-32"  # via sentence-transformers
CLIP_DIM = 512

# "Live view" = what the Chrome extension can see: the first 4 s at 2 fps (8 frames).
LIVE_SECONDS = 4.0
LIVE_FPS = 2.0

MOTION_FEATURES = ["cuts_per_sec", "motion_energy", "split_screen", "saturation", "brightness",
                   "colorfulness", "flicker"]


# ------------------------------------------------------------------ decoding
def sample_frames(video_path: str, fps: float = 1.0, max_frames: int = 30,
                  max_seconds: float | None = None) -> list[np.ndarray]:
    """Decode RGB frames at ~`fps` with OpenCV."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    native = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(native / fps))
    last = int(max_seconds * native) if max_seconds else None
    frames, i = [], 0
    while len(frames) < max_frames and (last is None or i < last):
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return frames


def cut_rate(video_path: str, threshold: float = 0.5, max_seconds: float | None = None) -> float:
    """Scene cuts per second via HSV-histogram distance between consecutive frames (~10 fps).
    The extension measures the same thing in-page on tiny thumbnails (extension/content.js)."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    native = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(native // 10))
    max_samples = int(max_seconds * 10) if max_seconds else None
    prev, cuts, i, n = None, 0, 0, 0
    while max_samples is None or n < max_samples:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            hsv = cv2.cvtColor(cv2.resize(frame, (160, 284)), cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            if prev is not None and cv2.compareHist(prev, hist, cv2.HISTCMP_BHATTACHARYYA) > threshold:
                cuts += 1
            prev, n = hist, n + 1
        i += 1
    cap.release()
    return cuts / max(n / 10, 1e-6) if n else 0.0


# ------------------------------------------------------------------ pace & style (numpy only)
def split_screen_score(frames: list[np.ndarray]) -> float:
    """0-1, high for the stacked "clip on top, gameplay below" layout. Two cues:
    * independence: both halves move, but their motion is uncorrelated over time
    * imbalance: one half is (nearly) still, e.g. a Reddit screenshot, while the other moves a lot
    """
    if len(frames) < 3:
        return 0.0
    gray = [f.mean(axis=2) for f in frames]
    h = gray[0].shape[0] // 2
    pairs = list(zip(gray, gray[1:], strict=False))
    top = np.array([np.abs(b[:h] - a[:h]).mean() for a, b in pairs])
    bot = np.array([np.abs(b[h:] - a[h:]).mean() for a, b in pairs])
    if max(top.mean(), bot.mean()) < 5:  # almost nothing moves anywhere (pixel units, 0-255)
        return 0.0
    independence = 0.0
    if top.std() > 1e-6 and bot.std() > 1e-6:
        independence = 1 - abs(float(np.corrcoef(top, bot)[0, 1]))
    imbalance = abs(top.mean() - bot.mean()) / (top.mean() + bot.mean())
    return float(max(independence, 0.9 * imbalance))


def motion_energy(frames: list[np.ndarray]) -> float:
    if len(frames) < 2:
        return 0.0
    diffs = [np.abs(b.astype(np.float32) - a).mean() for a, b in zip(frames, frames[1:], strict=False)]
    return float(np.mean(diffs) / 255)


def colour_stats(frames: list[np.ndarray]) -> dict[str, float]:
    """Saturation & brightness (HSV-style, 0-1), Hasler–Süsstrunk colourfulness, and brightness flicker."""
    if not frames:
        return {"saturation": 0.0, "brightness": 0.0, "colorfulness": 0.0, "flicker": 0.0}
    sats, vals, cols = [], [], []
    for f in frames:
        x = f.astype(np.float32) / 255
        mx, mn = x.max(axis=2), x.min(axis=2)
        sats.append(float(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0).mean()))
        vals.append(float(mx.mean()))
        r, g, b = x[..., 0], x[..., 1], x[..., 2]
        rg, yb = r - g, 0.5 * (r + g) - b
        cols.append(float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())))
    return {"saturation": float(np.mean(sats)), "brightness": float(np.mean(vals)),
            "colorfulness": float(np.mean(cols)), "flicker": float(np.std(vals))}


def motion_features(frames: list[np.ndarray], cuts_per_sec: float) -> dict[str, float]:
    return {"cuts_per_sec": float(cuts_per_sec), "motion_energy": motion_energy(frames),
            "split_screen": split_screen_score(frames), **colour_stats(frames)}


# ------------------------------------------------------------------ CLIP content
@lru_cache(maxsize=1)
def _clip():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(CLIP_MODEL)
    prompts = model.encode(CONCEPT_PROMPTS, normalize_embeddings=True, show_progress_bar=False)
    return model, prompts


def clip_features(frames: list[np.ndarray], max_frames: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mean frame embedding (512,), concept probabilities (n_concepts,))."""
    from PIL import Image

    model, prompts = _clip()
    if not frames:
        return np.zeros(CLIP_DIM, np.float32), np.full(len(prompts), 1 / len(prompts), np.float32)
    idx = np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).astype(int)
    emb = model.encode([Image.fromarray(frames[i]) for i in idx], normalize_embeddings=True,
                       show_progress_bar=False)
    mean = emb.mean(0)
    mean = mean / (np.linalg.norm(mean) + 1e-8)
    logits = 100 * emb @ prompts.T               # (frames, concepts), CLIP's usual temperature
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs = (probs / probs.sum(1, keepdims=True)).mean(0)  # average per-frame concept probabilities
    return mean.astype(np.float32), probs.astype(np.float32)


# ------------------------------------------------------------------ one entry point
def pixel_features(video_path: str | None = None, frames: list[np.ndarray] | None = None, live: bool = False,
                   cuts_per_sec: float | None = None, use_clip: bool = True) -> dict:
    """Full view: 1 fps over the whole video. Live view (`live=True`): first 4 s at 2 fps.
    At serve time pass `frames` from the extension and (if measured in-page) `cuts_per_sec`."""
    if frames is None and video_path:
        frames = (sample_frames(video_path, fps=LIVE_FPS, max_frames=8, max_seconds=LIVE_SECONDS) if live
                  else sample_frames(video_path))
    frames = frames or []
    if cuts_per_sec is None:
        cuts_per_sec = cut_rate(video_path, max_seconds=LIVE_SECONDS if live else None) if video_path else 0.0
    out = {"motion": motion_features(frames, cuts_per_sec), "clip_emb": None, "concepts": None}
    if use_clip and frames:
        out["clip_emb"], out["concepts"] = clip_features(frames)
    return out
