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
