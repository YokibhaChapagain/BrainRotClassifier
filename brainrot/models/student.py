"""The STUDENT: a pixels-only classifier that learns to predict the teacher's (Gemini's) labels.

Feature groups (for the ablation):
  motion : MOTION_FEATURES — pace & style (cuts/sec, motion, split screen, colour intensity, flicker)
  clip   : concept probabilities (one per CONCEPTS entry) + the 512-d mean CLIP embedding

Default model = standardized logistic regression + isotonic calibration. It's interpretable: each feature's
contribution (coefficient × standardized value) tells us *why* a video scored high, which drives the
explanations ("0.87 · looks like Minecraft gameplay + split screen · 3× the typical cut rate").
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from brainrot.features.concepts import CONCEPT_NAMES
from brainrot.features.video import CLIP_DIM, MOTION_FEATURES

GROUPS = {"motion": ("motion",), "clip": ("clip",), "motion+clip": ("motion", "clip")}


@dataclass
class PixelRow:
    motion: dict[str, float]
    concepts: np.ndarray | None = None  # (n_concepts,)
    clip_emb: np.ndarray | None = None  # (512,)


def feature_names(groups: tuple[str, ...]) -> list[str]:
    names = []
    if "motion" in groups:
        names += MOTION_FEATURES
    if "clip" in groups:
        names += [f"concept:{c}" for c in CONCEPT_NAMES] + [f"clip_{i}" for i in range(CLIP_DIM)]
    return names


def vectorize(row: PixelRow, groups: tuple[str, ...]) -> np.ndarray:
    parts = []
    if "motion" in groups:
        parts.append(np.array([row.motion.get(k, 0.0) for k in MOTION_FEATURES], np.float32))
    if "clip" in groups:
        n = len(CONCEPT_NAMES)
        parts.append(np.full(n, 1 / n, np.float32) if row.concepts is None else np.asarray(row.concepts, np.float32))
        parts.append(np.zeros(CLIP_DIM, np.float32) if row.clip_emb is None else np.asarray(row.clip_emb, np.float32))
    return np.concatenate(parts)


def rows_from_frame(frame) -> list[PixelRow]:
    """DataFrame with MOTION_FEATURES columns + `concepts` + `clip_emb` (lists) -> PixelRows."""
    out = []
    for d in frame.to_dict("records"):
        c, e = d.get("concepts"), d.get("clip_emb")
        out.append(PixelRow(motion={k: float(d.get(k) or 0.0) for k in MOTION_FEATURES},
                            concepts=None if c is None else np.asarray(c, np.float32),
                            clip_emb=None if e is None else np.asarray(e, np.float32)))
    return out


def make_base(params: dict):
    if params.get("model", "logreg") == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=params.get("C", 0.3), max_iter=4000,
                                                                  class_weight="balanced"))
    from lightgbm import LGBMClassifier  # comparison only: "does a non-linear model help?"

    return LGBMClassifier(n_estimators=params.get("n_estimators", 300), learning_rate=params.get("lr", 0.05),
                          num_leaves=params.get("num_leaves", 15), colsample_bytree=0.3, subsample=0.8,
                          subsample_freq=1, class_weight="balanced", verbose=-1)


class StudentModel:
    def __init__(self, groups: str = "motion+clip", params: dict | None = None, threshold: float = 0.6,
                 version: str = "dev"):
        self.groups_name = groups
        self.groups = GROUPS[groups]
        self.params = {"model": "logreg", "C": 0.3, **(params or {})}
        self.threshold = threshold
        self.version = version
        self.reference: dict[str, float] = {}  # "typical" (median) value of each motion feature
        self.clf = CalibratedClassifierCV(make_base(self.params), method="isotonic", cv=3)

    # ---------------------------------------------------------------- training
    def fit(self, rows: list[PixelRow], y: np.ndarray) -> StudentModel:
        X = np.stack([vectorize(r, self.groups) for r in rows])
        self.clf.fit(X, y)
        self.reference = {k: float(np.median([r.motion.get(k, 0.0) for r in rows])) for k in MOTION_FEATURES}
        return self

    def predict_proba(self, rows: list[PixelRow]) -> np.ndarray:
        X = np.stack([vectorize(r, self.groups) for r in rows])
        return self.clf.predict_proba(X)[:, 1]

    def tune_threshold(self, rows: list[PixelRow], y: np.ndarray) -> float:
        """Chosen on the VALIDATION set: the threshold with the best F1."""
        from sklearn.metrics import f1_score

        p = self.predict_proba(rows)
        grid = np.linspace(0.2, 0.9, 71)
        self.threshold = float(grid[np.argmax([f1_score(y, p >= t, zero_division=0) for t in grid])])
        return self.threshold

    # ---------------------------------------------------------------- interpretability
    def contributions(self, row: PixelRow) -> dict[str, float]:
        """Logistic-regression contribution of each named (non-embedding) feature: coef × standardized value,
        averaged over the calibration folds. Positive = pushes toward brain rot. Empty for LightGBM."""
        if self.params.get("model") != "logreg":
            return {}
        x = vectorize(row, self.groups)
        names = feature_names(self.groups)
        total = np.zeros_like(x, dtype=np.float64)
        for cc in self.clf.calibrated_classifiers_:
            scaler, lr = cc.estimator.named_steps["standardscaler"], cc.estimator.named_steps["logisticregression"]
            total += lr.coef_[0] * (x - scaler.mean_) / scaler.scale_
        total /= len(self.clf.calibrated_classifiers_)
        return {n: float(v) for n, v in zip(names, total, strict=True) if not n.startswith("clip_")}

    def global_importance(self, top: int = 15) -> list[tuple[str, float]]:
        """Average LR coefficient per named feature — for the slides ("what makes brain rot recognizable")."""
        if self.params.get("model") != "logreg":
            return []
        names = feature_names(self.groups)
        coef = np.mean([cc.estimator.named_steps["logisticregression"].coef_[0]
                        for cc in self.clf.calibrated_classifiers_], axis=0)
        pairs = [(n, float(c)) for n, c in zip(names, coef, strict=True) if not n.startswith("clip_")]
        return sorted(pairs, key=lambda p: -abs(p[1]))[:top]

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> StudentModel:
        return joblib.load(path)
