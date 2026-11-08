import base64
import io
import os

import numpy as np

os.environ["USE_CLIP"] = "0"  # tests run without downloading CLIP
os.environ["MODEL_PATH"] = "does-not-exist.joblib"

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from api.main import app  # noqa: E402

client = TestClient(app)


def _jpeg(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def split_frames(n=8):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        top = np.full((64, 72, 3), 100 + 2 * i, np.uint8)
        bot = (rng.random((64, 72, 3)) * 255 * rng.random()).astype(np.uint8)
        out.append(_jpeg(np.vstack([top, bot])))
    return out


def calm_frames(n=8):
    return [_jpeg(np.full((128, 72, 3), 120 + i, np.uint8)) for i in range(n)]


def test_health():
    assert client.get("/health").json()["ok"] is True


def test_predict_is_pixels_only_and_schema():
    r = client.post("/predict", json={"video_id": "a1", "frames_b64": split_frames(4), "cuts_per_sec": 2.0})
    assert r.status_code == 200
    out = r.json()
    assert set(out) >= {"prob", "flagged", "label", "threshold", "reasons", "summary", "top_concepts", "pace",
                        "n_frames", "final", "model_version"}
    assert out["n_frames"] == 4 and out["final"] is False
    assert out["label"] == ("brainrot" if out["flagged"] else "not_brainrot")
    assert out["summary"].startswith(f"{out['prob']:.2f}")
    assert "split screen" in out["reasons"]
    assert out["pace"]["cut_rate_vs_typical"] == 4.0
    # extra text fields are simply ignored: the model can't be fooled by captions
    r2 = client.post("/predict", json={"video_id": "a1-cap", "frames_b64": split_frames(4), "cuts_per_sec": 2.0,
                                       "caption": "calm cooking tutorial #education"})
    assert r2.json()["prob"] == out["prob"]


def test_rot_scores_above_calm():
    rot = client.post("/predict", json={"video_id": "rot", "frames_b64": split_frames(), "cuts_per_sec": 2.5}).json()
    calm = client.post("/predict", json={"video_id": "calm", "frames_b64": calm_frames(), "cuts_per_sec": 0.1}).json()
    assert rot["prob"] > calm["prob"] and rot["final"] is True


def test_view_event_and_stats_without_db():
    r = client.post("/events/view", json={"user_id": "u", "session_id": "s", "video_id": "v",
                                          "started_at": "2026-11-07T20:00:00Z", "watch_ms": 4200, "prob": 0.8,
                                          "flagged": True})
    assert r.json()["ok"] is True
    assert client.get("/stats/summary?user_id=u").json()["views"] == 0
    for path in ("/stats/timeline?user_id=u", "/stats/heatmap?user_id=u", "/stats/sessions?user_id=u",
                 "/stats/engagement", "/stats/trend"):
        assert client.get(path).json() == []


def test_dashboard_served():
    assert "Brain Rot Report" in client.get("/dashboard").text
