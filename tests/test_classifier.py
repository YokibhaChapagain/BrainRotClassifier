import pytest

from brainrot.classifier import BrainRotClassifier
from brainrot.config import Settings
from conftest import FakeBackend


def settings(**kw):
    s = Settings()
    s.api_key = ""
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_native_mode_uploads_video_and_includes_signals(hyper_video, fake_backend):
    clf = BrainRotClassifier(settings(mode="native", native_fps=2), backend=fake_backend)
    r = clf.classify(hyper_video, context="#skibidi #fyp")
    assert r.label == "brain_rot" and r.is_brain_rot and r.brain_rot_score == 80
    assert r.signals.cuts_per_minute > 100
    assert fake_backend.uploaded[0][1] == 2
    assert fake_backend.deleted == ["files/abc"]  # cleaned up
    parts, prompt = fake_backend.calls[0]
    assert parts[0][0] == "video"
    assert "cuts/minute" in prompt and "#skibidi" in prompt
    assert "{SIGNALS}" not in prompt and "{USER_CONTEXT}" not in prompt


def test_native_mode_trims_long_videos(calm_video, fake_backend):
    clf = BrainRotClassifier(settings(mode="native", max_seconds=5), backend=fake_backend)
    clf.classify(calm_video)
    sent = fake_backend.uploaded[0][0]
    assert sent != calm_video and sent.name.startswith("clip")


def test_frames_mode_sends_keyframes_and_audio(calm_video, fake_backend):
    clf = BrainRotClassifier(settings(mode="frames", max_frames=5), backend=fake_backend)
    clf.classify(calm_video)
    parts, prompt = fake_backend.calls[0]
    mimes = [p[1] for p in parts if p[0] == "bytes"]
    assert mimes.count("image/jpeg") == 5
    assert mimes.count("audio/mp3") == 1
    assert "5 keyframes" in prompt
    assert fake_backend.uploaded == []  # nothing uploaded in frames mode


@pytest.mark.parametrize("score,threshold,label", [(49, 50, "not_brain_rot"), (50, 50, "brain_rot"), (70, 75, "not_brain_rot")])
def test_threshold(calm_video, score, threshold, label):
    clf = BrainRotClassifier(settings(mode="frames", max_frames=2, threshold=threshold), backend=FakeBackend(score))
    assert clf.classify(calm_video).label == label


def test_youtube_url_goes_direct(fake_backend):
    clf = BrainRotClassifier(settings(mode="native"), backend=fake_backend)
    r = clf.classify("https://www.youtube.com/shorts/abc123")
    assert r.signals is None
    parts, prompt = fake_backend.calls[0]
    assert parts[0] == ("youtube", "https://www.youtube.com/shorts/abc123")
    assert "not available" in prompt


def test_missing_file(fake_backend):
    with pytest.raises(FileNotFoundError):
        BrainRotClassifier(settings(), backend=fake_backend).classify("/nope.mp4")


def test_bad_mode(fake_backend):
    with pytest.raises(ValueError):
        BrainRotClassifier(settings(mode="wat"), backend=fake_backend)


def test_no_api_key_gives_clear_error():
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        BrainRotClassifier(settings(mode="native"))
