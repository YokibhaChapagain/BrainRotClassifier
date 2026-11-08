"""Pre-classified video feed for the demo.

This is how a platform would use the classifier: score each video once, when it is uploaded, store
the verdict, and read the stored verdict when people scroll. Nothing is classified while you watch.

`brainrot index samples/` classifies every video in the folder and writes samples/feed.json.
The web app's /feed page then plays the folder's videos with those verdicts attached.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Callable

from .schema import ClassificationResult

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None

logger = logging.getLogger(__name__)
_thread_lock = threading.RLock()

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".mpeg", ".mpg", ".3gp", ".flv", ".wmv"}
CATALOG_NAME = "feed.json"


def feed_dir() -> Path:
    return Path(os.getenv("BRAINROT_FEED_DIR", "samples"))


def list_videos(folder: Path) -> list[Path]:
    """Videos directly inside the folder (not subfolders), in name order."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def _fingerprint(path: Path) -> dict:
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def load_catalog(folder: Path) -> dict[str, dict]:
    """{file name: {"size", "mtime_ns", "result": ClassificationResult as dict}}"""
    path = folder / CATALOG_NAME
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text()).get("videos", {})
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Ignoring unreadable %s", path)
        return {}


def save_catalog(folder: Path, videos: dict[str, dict]) -> None:
    path = folder / CATALOG_NAME
    tmp = path.with_suffix(f".json.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps({"videos": videos}, indent=2))
    tmp.replace(path)  # atomic, so the server never reads a half-written file


@contextlib.contextmanager
def _locked(folder: Path):
    """Exclusive lock on the catalog, across processes (the `index` CLI and the web server's uploads
    can run at the same time). On Windows, where fcntl doesn't exist, only threads are serialised."""
    with _thread_lock:
        if fcntl is None:
            yield
            return
        with open(folder / ".feed.lock", "w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)


def put_entry(folder: Path, name: str, entry: dict) -> None:
    """Add or replace one video's verdict without clobbering what other writers saved meanwhile."""
    with _locked(folder):
        catalog = load_catalog(folder)
        catalog[name] = entry
        save_catalog(folder, catalog)


def make_entry(path: Path, result: ClassificationResult, **extra) -> dict:
    result.source = path.name
    return {**_fingerprint(path), **extra, "result": json.loads(result.model_dump_json())}


def is_current(entry: dict | None, path: Path) -> bool:
    """True if the stored verdict belongs to this exact file (same size and modified time)."""
    return bool(entry and entry.get("result")) and {k: entry.get(k) for k in ("size", "mtime_ns")} == _fingerprint(path)


def index_folder(classify: Callable[[Path], ClassificationResult], folder: Path, force: bool = False,
                 on_result: Callable[[Path, ClassificationResult | None, Exception | None], None] | None = None,
                 ) -> tuple[int, int, int]:
    """Classify every new or changed video in `folder`. Saves after each video, so an interrupted
    run keeps what it finished. Returns (classified, skipped, failed)."""
    videos = list_videos(folder)
    with _locked(folder):  # drop verdicts for deleted files
        names = {p.name for p in videos}
        catalog = load_catalog(folder)
        if any(k not in names for k in catalog):
            save_catalog(folder, {k: v for k, v in catalog.items() if k in names})
    done = skipped = failed = 0
    for path in videos:
        entry = load_catalog(folder).get(path.name)  # fresh read: an upload may have added it meanwhile
        if not force and is_current(entry, path):
            skipped += 1
            continue
        try:
            r = classify(path)
        except Exception as e:  # keep going: one bad file shouldn't stop the batch
            failed += 1
            logger.error("Failed on %s: %s", path.name, e)
            if on_result:
                on_result(path, None, e)
            continue
        keep = {"uploaded_at": entry["uploaded_at"]} if entry and "uploaded_at" in entry else {}
        put_entry(folder, path.name, make_entry(path, r, **keep))
        done += 1
        if on_result:
            on_result(path, r, None)
    return done, skipped, failed


def safe_name(folder: Path, filename: str) -> str:
    """A file name for an upload: only safe characters, a video extension, and not already taken."""
    p = Path(filename or "upload.mp4")
    ext = p.suffix.lower() if p.suffix.lower() in VIDEO_EXTS else ".mp4"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", p.stem).strip("._-")[:60] or "upload"
    name, n = f"{stem}{ext}", 2
    while (folder / name).exists():
        name, n = f"{stem}-{n}{ext}", n + 1
    return name


def to_item(name: str, entry: dict) -> dict:
    r = entry["result"]
    v = r["verdict"]
    return {
        "id": name,
        "src": f"/media/{name}",
        "label": r["label"],
        "is_brain_rot": r["is_brain_rot"],
        "score": r["brain_rot_score"],
        "threshold": r["threshold"],
        "confidence": r["confidence"],
        "summary": v["content_summary"],
        "reasoning": v["reasoning"],
        "criteria": v["criteria"],
        "tropes": v["detected_tropes"],
        "redeeming": v["redeeming_qualities"],
        "signals": r.get("signals"),
        "model": r["model"],
        "mode": r["mode"],
        "uploaded": "uploaded_at" in entry,
    }


def feed_items(folder: Path) -> tuple[list[dict], list[str]]:
    """What the /feed page needs: uploads (newest first), then the rest in name order, plus names
    still waiting to be indexed."""
    catalog = load_catalog(folder)
    ready, pending = [], []
    for path in list_videos(folder):
        entry = catalog.get(path.name)
        if is_current(entry, path):
            ready.append((path.name, entry))
        else:
            pending.append(path.name)
    ready.sort(key=lambda ne: (-ne[1].get("uploaded_at", 0), ne[0]))
    return [to_item(n, e) for n, e in ready], pending
