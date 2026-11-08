import json

import numpy as np
import pandas as pd
import pytest

from brainrot.explain import build_reasons, summary_line
from brainrot.features.concepts import CONCEPT_NAMES
from brainrot.features.video import MOTION_FEATURES, colour_stats, motion_features, split_screen_score
from brainrot.labeling.gemini_labeler import parse_label
from brainrot.models.student import PixelRow, StudentModel, rows_from_frame, vectorize
from brainrot.schema import Reel

HF_ROW = {
    "video_id": 123, "creator": "c1", "description": "#fyp #viral", "transcript": None,
    "hashtags": json.dumps(["#brainrot", "fyp"]), "sticker_text": "when the skibidi toilet hits different fr fr",
    "engagement_metrics": json.dumps({"play_count": 1000, "like_count": 100}),
    "duration_ms": 12000, "is_ai_generated": True,
}


# ------------------------------------------------------------------ teacher side
def test_reel_teacher_text_and_rich_text_filter():
    r = Reel.from_hf_row(HF_ROW)
    assert r.video_id == "123" and r.hashtags == ["brainrot", "fyp"] and r.plays == 1000
    assert "ON-SCREEN TEXT: when the skibidi" in r.teacher_text()
    assert r.has_rich_text()                      # on-screen text is long enough
    assert not Reel("x", description="#fyp #viral").has_rich_text()


def test_parse_teacher_label():
    lab = parse_label('```json\n{"score": 14, "reasons": ["a"], "confidence": 0.9, "enough_info": true}\n```')
    assert lab.score == 10 and lab.label == 1 and lab.enough_info
    assert parse_label('{"score": 2, "enough_info": false}').label == 0
    with pytest.raises(ValueError):
        parse_label("no json here")


# ------------------------------------------------------------------ student features (pixels only)
def _frames(kind: str, n: int = 8, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        if kind == "split":   # top half: slow drift; bottom half: independent random flashing (gameplay)
            top = np.full((64, 72, 3), 100 + 2 * i, np.uint8)
            bot = (rng.random((64, 72, 3)) * 255 * rng.random()).astype(np.uint8)
            out.append(np.vstack([top, bot]))
        elif kind == "calm":  # whole frame changes slowly and together
            out.append(np.full((128, 72, 3), 120 + i, np.uint8))
        elif kind == "vivid":
            f = np.zeros((128, 72, 3), np.uint8)
            f[..., i % 3] = 255
            out.append(f)
    return out


def test_split_screen_detector():
    assert split_screen_score(_frames("split")) > 0.6
    assert split_screen_score(_frames("calm")) < 0.3


def test_colour_stats_vivid_vs_grey():
    vivid, grey = colour_stats(_frames("vivid")), colour_stats(_frames("calm"))
    assert vivid["saturation"] > 0.9 > grey["saturation"]
    assert vivid["colorfulness"] > grey["colorfulness"]


def test_motion_features_keys():
    m = motion_features(_frames("split"), cuts_per_sec=2.0)
    assert list(m) == MOTION_FEATURES and m["cuts_per_sec"] == 2.0


# ------------------------------------------------------------------ student model
def _synthetic(n=240, seed=0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    k = len(CONCEPT_NAMES)
    rows = []
    for yi in y:
        concepts = rng.dirichlet(np.ones(k))
        if yi:
            concepts[0] += 1.5            # "Minecraft gameplay"
        concepts /= concepts.sum()
        rows.append({"cuts_per_sec": rng.gamma(2, 0.3) + 1.2 * yi, "motion_energy": rng.random() * 0.1 + 0.05 * yi,
                     "split_screen": rng.random() * 0.4 + 0.5 * yi, "saturation": rng.random(),
                     "brightness": rng.random(), "colorfulness": rng.random() * 0.4 + 0.2 * yi,
                     "flicker": rng.random() * 0.02, "concepts": concepts.tolist(),
                     "clip_emb": rng.normal(0, 1, 512).tolist()})
    return pd.DataFrame(rows), y


@pytest.mark.parametrize("groups", ["motion", "clip", "motion+clip"])
def test_student_trains_per_group(groups):
    df, y = _synthetic()
    m = StudentModel(groups).fit(rows_from_frame(df), y)
    p = m.predict_proba(rows_from_frame(df))
    assert ((p >= 0.5) == y).mean() > 0.75


def test_vectorize_shapes():
    row = PixelRow(motion={k: 1.0 for k in MOTION_FEATURES})
    assert vectorize(row, ("motion",)).shape == (len(MOTION_FEATURES),)
    assert vectorize(row, ("motion", "clip")).shape == (len(MOTION_FEATURES) + len(CONCEPT_NAMES) + 512,)


def test_contributions_and_reasons(tmp_path):
    df, y = _synthetic()
    m = StudentModel("motion+clip").fit(rows_from_frame(df), y)
    m.tune_threshold(rows_from_frame(df), y)
    rot = rows_from_frame(df[y == 1].head(1))[0]
    rot.motion.update(cuts_per_sec=3 * m.reference["cuts_per_sec"], split_screen=0.9)
    rot.concepts = np.eye(len(CONCEPT_NAMES))[0] * 0.9 + 0.1 / len(CONCEPT_NAMES)
    contrib = m.contributions(rot)
    assert contrib["split_screen"] > 0 and contrib["cuts_per_sec"] > 0
    reasons = build_reasons(rot.motion, rot.concepts, m.reference, contrib, flagged=True)
    texts = [r["text"] for r in reasons]
    assert "looks like Minecraft gameplay" in texts and "split screen" in texts
    assert any("the typical cut rate" in t for t in texts)
    line = summary_line(0.87, reasons)
    assert line.startswith("0.87 · ") and "Minecraft" in line
    assert m.global_importance()[0][0] in {n for n in ["cuts_per_sec", "split_screen", "motion_energy",
                                                        "colorfulness"]} | {f"concept:{c}" for c in CONCEPT_NAMES}
    path = tmp_path / "s.joblib"
    m.save(str(path))
    assert np.allclose(StudentModel.load(str(path)).predict_proba([rot]), m.predict_proba([rot]))


def test_summary_format_matches_pitch():
    reasons = [{"text": "looks like Minecraft gameplay", "feature": "x", "kind": "content"},
               {"text": "split screen", "feature": "y", "kind": "content"},
               {"text": "3× the typical cut rate", "feature": "z", "kind": "pace"}]
    expected = "0.87 · looks like Minecraft gameplay + split screen · 3× the typical cut rate"
    assert summary_line(0.87, reasons) == expected
