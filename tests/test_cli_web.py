import base64
import json

from fastapi.testclient import TestClient

from brainrot import classifier as clf_mod
from brainrot import cli
from brainrot.web import server
from conftest import FakeBackend


def _patch(monkeypatch, score=80):
    orig = clf_mod.BrainRotClassifier.__init__

    def init(self, settings=None, backend=None):
        orig(self, settings, backend or FakeBackend(score))

    monkeypatch.setattr(clf_mod.BrainRotClassifier, "__init__", init)


def test_cli_classify_json(monkeypatch, calm_video, capsys, tmp_path):
    _patch(monkeypatch)
    out = tmp_path / "r.jsonl"
    rc = cli.main(["classify", str(calm_video), "--mode", "frames", "--json", "-o", str(out)])
    assert rc == 0
    assert json.loads(out.read_text())["label"] == "brain_rot"


def test_cli_eval(monkeypatch, calm_video, hyper_video, tmp_path, capsys):
    _patch(monkeypatch, score=80)  # fake always says brain rot
    csv = tmp_path / "labels.csv"
    csv.write_text(f"source,label\n{hyper_video},brain_rot\n{calm_video},not_brain_rot\n")
    assert cli.main(["eval", str(csv), "--mode", "frames", "--max-frames", "2"]) == 0
    out = capsys.readouterr().out
    assert "accuracy 50.0%" in out and "TP=1 FP=1 TN=0 FN=0" in out


def test_web_upload(monkeypatch, hyper_video):
    _patch(monkeypatch, score=30)
    c = TestClient(server.app)
    assert "Brain Rot Detector" in c.get("/").text
    with open(hyper_video, "rb") as f:
        r = c.post("/api/classify", files={"file": ("hyper.mp4", f, "video/mp4")},
                   data={"mode": "frames", "threshold": "50"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["label"] == "not_brain_rot" and d["source"] == "hyper.mp4"
    assert d["signals"]["is_vertical"]


def test_web_requires_input(monkeypatch):
    _patch(monkeypatch)
    r = TestClient(server.app).post("/api/classify", data={"mode": "native"})
    assert r.status_code == 400


def _jpeg_b64() -> str:
    import cv2
    import numpy as np

    ok, buf = cv2.imencode(".jpg", np.zeros((64, 36, 3), dtype=np.uint8))
    return base64.b64encode(buf.tobytes()).decode()


def test_predict_scores_frames_and_caches(monkeypatch):
    calls = []
    orig = clf_mod.BrainRotClassifier.__init__

    def init(self, settings=None, backend=None):
        fb = FakeBackend(80)
        calls.append(fb)
        orig(self, settings, fb)

    monkeypatch.setattr(clf_mod.BrainRotClassifier, "__init__", init)
    server._cache.clear()
    c = TestClient(server.app)
    body = {"video_id": "C9xyz", "platform": "instagram", "frames_b64": [_jpeg_b64()] * 8}
    r = c.post("/predict", json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["prob"] == 0.8 and d["flagged"] is True and d["label"] == "brain_rot"
    assert d["reasons"] == ["rapid cuts"]
    parts, prompt = calls[0].calls[0]
    assert [p[1] for p in parts if p[0] == "bytes"].count("image/jpeg") == 8
    assert "8 screenshots" in prompt and "captured live" in prompt

    assert c.post("/predict", json=body).json() == d
    assert len(calls) == 1  # second call came from the cache


def test_predict_rejects_bad_input(monkeypatch):
    _patch(monkeypatch)
    server._cache.clear()
    c = TestClient(server.app)
    assert c.post("/predict", json={"video_id": "a", "frames_b64": []}).status_code == 400
    assert c.post("/predict", json={"video_id": "b", "frames_b64": ["not base64!!"]}).status_code == 400


def test_health():
    d = TestClient(server.app).get("/health").json()
    assert d["ok"] is True and "model" in d


def test_predict_clip_scores_recorded_webm_and_caches(monkeypatch, hyper_video, tmp_path):
    import subprocess

    calls = []
    orig = clf_mod.BrainRotClassifier.__init__

    def init(self, settings=None, backend=None):
        fb = FakeBackend(80)
        calls.append(fb)
        orig(self, settings, fb)

    monkeypatch.setattr(clf_mod.BrainRotClassifier, "__init__", init)
    monkeypatch.setenv("BRAINROT_MODE", "frames")
    monkeypatch.setenv("BRAINROT_MAX_FRAMES", "4")
    # same kind of file MediaRecorder produces: webm, vp8 + opus
    webm = tmp_path / "clip.webm"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(hyper_video), "-t", "4",
                    "-c:v", "libvpx", "-b:v", "500k", "-c:a", "libopus", str(webm)], check=True)
    server._cache.clear()
    c = TestClient(server.app)
    form = {"video_id": "7301", "platform": "tiktok"}
    with open(webm, "rb") as f:
        r = c.post("/predict_clip", files={"file": ("clip.webm", f, "video/webm")}, data=form)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["flagged"] is True and d["prob"] == 0.8 and d["mode"] == "frames"
    assert d["signals"]["has_audio"] and d["signals"]["scene_cuts"] > 5
    parts, _ = calls[0].calls[0]
    kinds = [p[1] for p in parts if p[0] == "bytes"]
    assert kinds.count("image/jpeg") == 4 and "audio/mp3" in kinds  # BRAINROT_MAX_FRAMES applies here

    with open(webm, "rb") as f:
        assert c.post("/predict_clip", files={"file": ("clip.webm", f, "video/webm")}, data=form).json() == d
    assert len(calls) == 1  # cached by video id


def test_predict_clip_rejects_garbage(monkeypatch):
    _patch(monkeypatch)
    server._cache.clear()
    c = TestClient(server.app)
    r = c.post("/predict_clip", files={"file": ("clip.webm", b"not a video", "video/webm")},
               data={"video_id": "x", "platform": "tiktok"})
    assert r.status_code == 502 and "could not read" in r.text
    r = c.post("/predict_clip", files={"file": ("clip.webm", b"", "video/webm")},
               data={"video_id": "y", "platform": "tiktok"})
    assert r.status_code == 400
