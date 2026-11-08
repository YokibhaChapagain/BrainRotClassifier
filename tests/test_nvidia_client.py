import json

import httpx
import pytest

from brainrot import nvidia_client
from brainrot.classifier import BrainRotClassifier, make_backend
from brainrot.config import Settings
from brainrot.nvidia_client import NvidiaBackend, _parse_verdict
from conftest import FakeBackend, make_verdict


def _backend(handler, monkeypatch):
    b = NvidiaBackend("nvapi-test", "minimaxai/minimax-m3.1-flash")
    b.http = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(nvidia_client.time, "sleep", lambda s: None)
    return b


def _answer(text, status=200):
    return httpx.Response(status, json={"choices": [{"message": {"role": "assistant", "content": text},
                                                     "finish_reason": "stop"}]})


VERDICT_JSON = make_verdict(72).model_dump_json()


def test_sends_images_and_schema_and_parses_answer(monkeypatch):
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        assert req.headers["authorization"] == "Bearer nvapi-test"
        return _answer("Here you go:\n```json\n" + VERDICT_JSON + "\n```")

    b = _backend(handler, monkeypatch)
    v = b.classify([b.text_part("Keyframe 0"), b.bytes_part(b"\xff\xd8jpeg", "image/jpeg")], "Classify this.")
    assert v.brain_rot_score == 72
    body = seen[0]
    assert body["model"] == "minimaxai/minimax-m3.1-flash" and "response_format" not in body
    content = body["messages"][0]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "brain_rot_score" in content[-1]["text"] and content[-1]["text"].startswith("Classify this.")


def test_bad_json_retries_once_in_json_mode(monkeypatch):
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return _answer("I think it's brain rot, score 80." if len(bodies) == 1 else VERDICT_JSON)

    v = _backend(handler, monkeypatch).classify([], "Classify.")
    assert v.brain_rot_score == 72
    assert len(bodies) == 2 and bodies[1]["response_format"] == {"type": "json_object"}


def test_retries_rate_limits_then_reports_errors(monkeypatch):
    codes = iter([429, 503, 200])

    def handler(req):
        c = next(codes)
        return _answer(VERDICT_JSON) if c == 200 else httpx.Response(c, text="slow down")

    assert _backend(handler, monkeypatch).classify([], "x").brain_rot_score == 72

    b = _backend(lambda req: httpx.Response(401, text="bad key"), monkeypatch)
    with pytest.raises(RuntimeError, match="401"):
        b.classify([], "x")


def test_parse_tolerates_think_tags_and_rejects_garbage():
    assert _parse_verdict("<think>{not this}</think>" + VERDICT_JSON).brain_rot_score == 72
    with pytest.raises(ValueError):
        _parse_verdict("no json here")


def test_only_images_go_in():
    b = NvidiaBackend("k", "m")
    with pytest.raises(ValueError):
        b.bytes_part(b"x", "audio/mp3")
    with pytest.raises(RuntimeError):
        b.video_file_part(None)


class ImagesOnly(FakeBackend):
    supports_video = False
    supports_audio = False


def test_images_only_backend_gets_keyframes_without_audio(calm_video):
    fb = ImagesOnly(30)
    s = Settings()
    s.mode, s.max_frames = "native", 3
    r = BrainRotClassifier(s, backend=fb).classify(calm_video)
    assert r.mode == "frames" and fb.uploaded == []
    parts, prompt = fb.calls[0]
    kinds = [p[1] for p in parts if p[0] == "bytes"]
    assert kinds == ["image/jpeg"] * 3  # no audio/mp3
    assert "can't hear it" in prompt


def test_backend_choice(monkeypatch):
    monkeypatch.delenv("BRAINROT_BACKEND", raising=False)
    assert Settings().backend == "gemini"  # the default
    monkeypatch.setenv("BRAINROT_BACKEND", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-x")
    assert isinstance(make_backend(Settings()), NvidiaBackend)
    monkeypatch.setenv("NVIDIA_API_KEY", "")
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        make_backend(Settings())
    monkeypatch.setenv("BRAINROT_BACKEND", "openai")
    with pytest.raises(ValueError):
        make_backend(Settings())
