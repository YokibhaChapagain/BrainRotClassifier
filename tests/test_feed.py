import os
import shutil

from fastapi.testclient import TestClient

from brainrot import classifier as clf_mod
from brainrot import cli, feed
from brainrot.web import server
from conftest import FakeBackend


def _feed_folder(tmp_path, calm_video, hyper_video):
    d = tmp_path / "samples"
    d.mkdir()
    shutil.copy(calm_video, d / "a_calm.mp4")
    shutil.copy(hyper_video, d / "b_hyper.mp4")
    (d / "notes.txt").write_text("not a video")
    return d


def _patch(monkeypatch, scores):
    """Fake Gemini: score depends on the file name, counts calls."""
    calls = []
    orig = clf_mod.BrainRotClassifier.__init__

    class Backend(FakeBackend):
        def video_file_part(self, path, fps=None):
            self.score = scores[path.name]
            calls.append(path.name)
            return super().video_file_part(path, fps)

    def init(self, settings=None, backend=None):
        orig(self, settings, backend or Backend())

    monkeypatch.setattr(clf_mod.BrainRotClassifier, "__init__", init)
    return calls


def test_index_then_feed(monkeypatch, tmp_path, calm_video, hyper_video, capsys):
    d = _feed_folder(tmp_path, calm_video, hyper_video)
    calls = _patch(monkeypatch, {"a_calm.mp4": 10, "b_hyper.mp4": 90})
    monkeypatch.setenv("BRAINROT_FEED_DIR", str(d))
    monkeypatch.setenv("BRAINROT_MODE", "native")  # don't depend on the developer's .env

    assert cli.main(["index"]) == 0
    assert sorted(calls) == ["a_calm.mp4", "b_hyper.mp4"]
    assert "2 classified, 0 skipped" in capsys.readouterr().out
    assert (d / "feed.json").is_file()

    # second run: nothing changed, so no model calls
    assert cli.main(["index"]) == 0
    assert len(calls) == 2 and "0 classified, 2 skipped" in capsys.readouterr().out

    c = TestClient(server.app)
    assert "Brain Rot Feed" in c.get("/feed").text
    data = c.get("/api/feed").json()
    assert data["pending"] == []
    assert [i["id"] for i in data["items"]] == ["a_calm.mp4", "b_hyper.mp4"]  # the feed opens with non brain rot
    calm, hyper = data["items"]
    assert calm["is_brain_rot"] is False and calm["score"] == 10
    assert hyper["is_brain_rot"] is True and hyper["score"] == 90 and hyper["tropes"] == ["rapid cuts"]
    assert hyper["signals"]["is_vertical"] and hyper["src"] == "/media/b_hyper.mp4"

    # changing a file makes it pending until it's indexed again
    os.utime(d / "a_calm.mp4", ns=(1, 1))
    data = c.get("/api/feed").json()
    assert data["pending"] == ["a_calm.mp4"] and [i["id"] for i in data["items"]] == ["b_hyper.mp4"]
    assert cli.main(["index"]) == 0 and calls[-1] == "a_calm.mp4" and len(calls) == 3


def test_media_serves_ranges_and_blocks_other_files(monkeypatch, tmp_path, calm_video, hyper_video):
    d = _feed_folder(tmp_path, calm_video, hyper_video)
    monkeypatch.setenv("BRAINROT_FEED_DIR", str(d))
    c = TestClient(server.app)
    size = (d / "a_calm.mp4").stat().st_size

    r = c.get("/media/a_calm.mp4")
    assert r.status_code == 200 and len(r.content) == size and r.headers["content-type"] == "video/mp4"
    r = c.get("/media/a_calm.mp4", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100

    assert c.get("/media/notes.txt").status_code == 404
    assert c.get("/media/feed.json").status_code == 404
    assert c.get("/media/..%2F..%2Fetc%2Fpasswd").status_code == 404


def test_index_keeps_going_after_a_failure(monkeypatch, tmp_path, calm_video, hyper_video):
    d = _feed_folder(tmp_path, calm_video, hyper_video)

    def classify(path):
        if path.name == "a_calm.mp4":
            raise RuntimeError("boom")
        r = clf_mod.BrainRotClassifier(backend=FakeBackend(70)).classify(path)
        return r

    done, skipped, failed = feed.index_folder(classify, d)
    assert (done, skipped, failed) == (1, 0, 1)
    items, pending = feed.feed_items(d)
    assert [i["id"] for i in items] == ["b_hyper.mp4"] and pending == ["a_calm.mp4"]


def test_empty_feed(monkeypatch, tmp_path):
    monkeypatch.setenv("BRAINROT_FEED_DIR", str(tmp_path / "nothing"))
    assert TestClient(server.app).get("/api/feed").json()["items"] == []
    assert cli.main(["index"]) == 1


def test_upload_classifies_once_and_joins_feed_first(monkeypatch, tmp_path, calm_video, hyper_video):
    d = _feed_folder(tmp_path, calm_video, hyper_video)
    calls = _patch(monkeypatch, {"a_calm.mp4": 10, "b_hyper.mp4": 90, "my_clip.mp4": 75})
    monkeypatch.setenv("BRAINROT_FEED_DIR", str(d))
    monkeypatch.setenv("BRAINROT_MODE", "native")
    assert cli.main(["index"]) == 0
    c = TestClient(server.app)

    with open(hyper_video, "rb") as f:
        r = c.post("/api/feed/upload", files={"file": ("my clip!.mp4", f, "video/mp4")})
    assert r.status_code == 200, r.text
    item = r.json()
    assert item["id"] == "my_clip.mp4" and item["uploaded"] and item["is_brain_rot"] and item["score"] == 75
    assert (d / "my_clip.mp4").is_file() and not list(d.glob(".*.part"))
    assert calls.count("my_clip.mp4") == 1

    feed_ids = [i["id"] for i in c.get("/api/feed").json()["items"]]
    assert feed_ids == ["my_clip.mp4", "a_calm.mp4", "b_hyper.mp4"]  # uploads first
    assert c.get("/media/my_clip.mp4").status_code == 200

    # already classified at upload time, so index doesn't call Gemini again, and keeps it marked as an upload
    assert cli.main(["index"]) == 0 and calls.count("my_clip.mp4") == 1
    assert c.get("/api/feed").json()["items"][0]["uploaded"] is True

    # same name again gets a new file name instead of overwriting
    calls_before = len(calls)
    with open(calm_video, "rb") as f:
        r = c.post("/api/feed/upload", files={"file": ("a_calm.mp4", f, "video/mp4")})
    assert r.status_code == 502  # fake backend has no score for "a_calm-2.mp4" -> classification error
    assert not (d / "a_calm-2.mp4").exists() and not list(d.glob(".*.part"))  # failed upload leaves nothing behind
    assert len(calls) == calls_before




def test_upload_rejects_empty_and_reports_gemini_errors(monkeypatch, tmp_path):
    d = tmp_path / "feed"
    monkeypatch.setenv("BRAINROT_FEED_DIR", str(d))
    c = TestClient(server.app)
    _patch(monkeypatch, {})
    r = c.post("/api/feed/upload", files={"file": ("x.mp4", b"", "video/mp4")})
    assert r.status_code == 400
    r = c.post("/api/feed/upload", files={"file": ("../../evil.sh", b"not a video", "video/mp4")})
    assert r.status_code == 502
    assert sorted(p.name for p in d.iterdir() if not p.name.startswith(".feed.lock")) == []  # nothing left, nothing outside
    assert not (tmp_path / "evil.sh").exists()


def test_safe_name(tmp_path):
    assert feed.safe_name(tmp_path, "../../etc/passwd") == "passwd.mp4"
    assert feed.safe_name(tmp_path, "My Video (1).MOV") == "My_Video_1.mov"
    assert feed.safe_name(tmp_path, "") == "upload.mp4"
    (tmp_path / "clip.mp4").write_bytes(b"x")
    assert feed.safe_name(tmp_path, "clip.mp4") == "clip-2.mp4"


def test_concurrent_writers_keep_each_others_entries(tmp_path, calm_video):
    import threading

    from conftest import make_verdict
    from brainrot.schema import ClassificationResult

    names = [f"v{i}.mp4" for i in range(12)]
    for n in names:
        shutil.copy(calm_video, tmp_path / n)
    r = ClassificationResult(source="", label="brain_rot", is_brain_rot=True, brain_rot_score=80, threshold=50,
                             confidence=0.9, mode="frames", model="fake", verdict=make_verdict(80), elapsed_s=0.1)
    threads = [threading.Thread(target=feed.put_entry, args=(tmp_path, n, feed.make_entry(tmp_path / n, r.model_copy())))
               for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(feed.load_catalog(tmp_path)) == sorted(names)


def test_feed_uses_captions_when_available(monkeypatch, tmp_path, calm_video, hyper_video):
    import json as _json

    d = _feed_folder(tmp_path, calm_video, hyper_video)
    _patch(monkeypatch, {"a_calm.mp4": 10, "b_hyper.mp4": 90})
    monkeypatch.setenv("BRAINROT_FEED_DIR", str(d))
    monkeypatch.setenv("BRAINROT_MODE", "native")
    assert cli.main(["index"]) == 0
    (d / "captions.json").write_text(_json.dumps({"b_hyper.mp4": {"creator": "someone", "caption": "wait for it #fyp"}}))
    items = {i["id"]: i for i in TestClient(server.app).get("/api/feed").json()["items"]}
    assert items["b_hyper.mp4"]["creator"] == "someone" and items["b_hyper.mp4"]["caption"] == "wait for it #fyp"
    assert items["a_calm.mp4"]["creator"] is None and items["a_calm.mp4"]["caption"] is None  # falls back to file name
    (d / "captions.json").write_text("not json")
    assert TestClient(server.app).get("/api/feed").status_code == 200


def _item(name, score, rot, uploaded=False):
    return {"id": name, "score": score, "is_brain_rot": rot, "uploaded": uploaded}


def test_arrange_opens_clean_then_alternates_worst_brain_rot():
    items = [_item("a", 10, False), _item("b", 60, True), _item("c", 20, False), _item("d", 90, True),
             _item("e", 15, False), _item("f", 75, True), _item("g", 55, True), _item("h", 5, False),
             _item("up", 30, False, uploaded=True)]
    order = [it["id"] for it in feed.arrange(items)]
    # upload first, then two clean, then rot/clean alternating with the highest scores first
    # (clean runs out, so the last slot goes to the remaining brain rot)
    assert order == ["up", "a", "c", "d", "e", "f", "h", "b", "g"]


def test_arrange_fills_from_the_other_kind_when_one_runs_out():
    items = [_item("a", 10, False), _item("b", 70, True), _item("c", 20, False), _item("d", 30, False)]
    assert [it["id"] for it in feed.arrange(items)] == ["a", "c", "b", "d"]
    assert [it["id"] for it in feed.arrange([_item("x", 80, True), _item("y", 90, True)])] == ["y", "x"]


def test_arrange_with_too_few_clean_videos_to_lead():
    items = [_item("a", 10, False), _item("x", 80, True), _item("y", 90, True)]
    assert [it["id"] for it in feed.arrange(items)] == ["a", "y", "x"]
