"""Exercise the real GeminiBackend against the real SDK types, with the network call mocked."""
from types import SimpleNamespace

import pytest
from google.genai import errors, types

from brainrot import gemini_client
from brainrot.gemini_client import GeminiBackend
from conftest import make_verdict


def test_classify_builds_request_and_parses_json(monkeypatch):
    b = GeminiBackend(api_key="test-key", model="gemini-test")
    seen = {}

    def fake_generate(model, contents, config):
        seen.update(model=model, contents=contents, config=config)
        return SimpleNamespace(text=make_verdict(65).model_dump_json(), prompt_feedback=None)

    monkeypatch.setattr(b.client.models, "generate_content", fake_generate)
    part = b.youtube_part("https://youtu.be/x", fps=2)
    v = b.classify([part, b.bytes_part(b"\xff\xd8jpeg", "image/jpeg")], "prompt here")
    assert v.brain_rot_score == 65
    assert seen["model"] == "gemini-test"
    assert seen["config"].response_mime_type == "application/json"
    assert "brain_rot_score" in seen["config"].response_json_schema["properties"]
    parts = seen["contents"][0].parts
    assert parts[0].file_data.file_uri == "https://youtu.be/x" and parts[0].video_metadata.fps == 2
    assert parts[1].inline_data.mime_type == "image/jpeg"
    assert parts[-1].text == "prompt here"


def test_retries_on_429(monkeypatch):
    b = GeminiBackend(api_key="k", model="m")
    monkeypatch.setattr(gemini_client.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky(**kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise errors.ClientError(429, {"error": {"code": 429, "message": "rate", "status": "RESOURCE_EXHAUSTED"}})
        return SimpleNamespace(text=make_verdict().model_dump_json(), prompt_feedback=None)

    monkeypatch.setattr(b.client.models, "generate_content", flaky)
    assert b.classify([], "p").brain_rot_score == 80
    assert calls["n"] == 3


def test_does_not_retry_on_400(monkeypatch):
    b = GeminiBackend(api_key="k", model="m")

    def bad(**kw):
        raise errors.ClientError(400, {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}})

    monkeypatch.setattr(b.client.models, "generate_content", bad)
    with pytest.raises(errors.ClientError):
        b.classify([], "p")


def test_upload_waits_for_active(monkeypatch, calm_video):
    b = GeminiBackend(api_key="k", model="m")
    monkeypatch.setattr(gemini_client.time, "sleep", lambda s: None)
    states = iter(["PROCESSING", "ACTIVE"])
    mk = lambda st: types.File(name="files/1", uri="https://gen/files/1", mime_type="video/mp4", state=st)
    monkeypatch.setattr(b.client.files, "upload", lambda file, config: mk("PROCESSING"))
    monkeypatch.setattr(b.client.files, "get", lambda name: mk(next(states)))
    part, name = b.video_file_part(calm_video, fps=2)
    assert name == "files/1" and part.file_data.file_uri.endswith("files/1")
