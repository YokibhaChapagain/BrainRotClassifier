import subprocess
from pathlib import Path

import pytest

from brainrot.schema import CriterionScores, GeminiVerdict


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


@pytest.fixture(scope="session")
def calm_video(tmp_path_factory) -> Path:
    """12s, one continuous slowly-panning shot, quiet tone. Should have ~0 cuts."""
    out = tmp_path_factory.mktemp("v") / "calm.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=12",
            "-f", "lavfi", "-i", "sine=frequency=220:duration=12",
            "-af", "volume=0.1", "-shortest", "-pix_fmt", "yuv420p", str(out))
    return out


@pytest.fixture(scope="session")
def hyper_video(tmp_path_factory) -> Path:
    """12s vertical video that hard-cuts between very different scenes every 0.25s, loud audio."""
    d = tmp_path_factory.mktemp("v")
    colours = ["red", "blue", "yellow", "green", "magenta", "white", "black", "cyan"]
    srcs = []
    for i in range(48):
        c = colours[i % len(colours)]
        srcs += ["-f", "lavfi", "-i", f"color=c={c}:size=360x640:rate=30:duration=0.25"]
    concat = "".join(f"[{i}:v]" for i in range(48)) + "concat=n=48:v=1:a=0[v]"
    out = d / "hyper.mp4"
    _ffmpeg(*srcs, "-f", "lavfi", "-i", "sine=frequency=880:duration=12",
            "-filter_complex", concat, "-map", "[v]", "-map", "48:a", "-shortest",
            "-pix_fmt", "yuv420p", str(out))
    return out


def make_verdict(score: int = 80) -> GeminiVerdict:
    return GeminiVerdict(
        content_summary="Test video.",
        criteria=CriterionScores(overstimulation=9, low_informational_value=8, incoherence_absurdism=7,
                                 meme_slang_density=6, split_screen_or_filler_gameplay=5,
                                 ai_slop_or_tts=4, engagement_bait=3),
        detected_tropes=["rapid cuts"],
        redeeming_qualities=[],
        brain_rot_score=score,
        confidence=0.9,
        reasoning="Because.",
    )


class FakeBackend:
    """Records what the classifier would have sent to Gemini."""

    model = "fake-gemini"

    def __init__(self, score: int = 80):
        self.score = score
        self.calls = []
        self.uploaded = []
        self.deleted = []

    def video_file_part(self, path, fps=None):
        self.uploaded.append((Path(path), fps))
        return ("video", str(path)), "files/abc"

    def youtube_part(self, url, fps=None):
        return ("youtube", url)

    def bytes_part(self, data, mime_type):
        return ("bytes", mime_type, len(data))

    def text_part(self, text):
        return ("text", text)

    def delete_file(self, name):
        self.deleted.append(name)

    def classify(self, parts, prompt):
        self.calls.append((parts, prompt))
        return make_verdict(self.score)


@pytest.fixture
def fake_backend():
    return FakeBackend()
