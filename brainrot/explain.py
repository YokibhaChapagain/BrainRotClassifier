"""Turn a score + measured pixel features into human-readable reasons.

Example: "0.87 · looks like Minecraft gameplay + split screen · 3× the typical cut rate"

Reasons are built only from things the student actually measured. When the model is logistic regression,
reasons are ordered by how much each feature pushed the score (its contribution); otherwise by strength.
"""

from __future__ import annotations

import numpy as np

from brainrot.features.concepts import CONCEPT_NAMES

CONCEPT_MIN_PROB = 0.25


def _ratio(value: float, typical: float) -> float:
    return value / typical if typical > 1e-6 else 0.0


def _fmt_ratio(r: float) -> str:
    return f"{r:.0f}×" if r >= 3 else f"{r:.1f}×"


def build_reasons(motion: dict[str, float], concepts: np.ndarray | None, reference: dict[str, float],
                  contributions: dict[str, float] | None = None, flagged: bool = True) -> list[dict]:
    """Returns [{"text", "feature", "kind": "content"|"pace"}], most important first."""
    contributions = contributions or {}
    cands: list[dict] = []

    if concepts is not None and len(concepts):
        i = int(np.argmax(concepts))
        if concepts[i] >= CONCEPT_MIN_PROB:
            name = CONCEPT_NAMES[i]
            cands.append({"text": f"looks like {name}", "feature": f"concept:{name}", "kind": "content",
                          "strength": float(concepts[i])})

    if motion.get("split_screen", 0) >= 0.6:
        cands.append({"text": "split screen", "feature": "split_screen", "kind": "content",
                      "strength": motion["split_screen"]})

    cut = _ratio(motion.get("cuts_per_sec", 0), reference.get("cuts_per_sec", 0))
    if cut >= 1.5:
        cands.append({"text": f"{_fmt_ratio(cut)} the typical cut rate", "feature": "cuts_per_sec",
                      "kind": "pace", "strength": cut / 3})
    elif 0 < cut <= 0.5 or (motion.get("cuts_per_sec", 0) == 0 and reference.get("cuts_per_sec", 0) > 0):
        cands.append({"text": "slow, calm editing", "feature": "cuts_per_sec", "kind": "pace", "strength": 0.5})

    mot = _ratio(motion.get("motion_energy", 0), reference.get("motion_energy", 0))
    if mot >= 1.5:
        cands.append({"text": f"{_fmt_ratio(mot)} the typical on-screen motion", "feature": "motion_energy",
                      "kind": "pace", "strength": mot / 3})

    col = _ratio(motion.get("colorfulness", 0), reference.get("colorfulness", 0))
    if col >= 1.3:
        cands.append({"text": "intensely saturated colours", "feature": "colorfulness", "kind": "pace",
                      "strength": col / 2})

    flick = _ratio(motion.get("flicker", 0), reference.get("flicker", 0))
    if flick >= 2:
        cands.append({"text": "flashing brightness", "feature": "flicker", "kind": "pace", "strength": flick / 4})

    if contributions:
        sign = 1 if flagged else -1
        for c in cands:
            c["strength"] = sign * contributions.get(c["feature"], 0.0)
        cands = [c for c in cands if c["strength"] > 0] or cands  # keep reasons that agree with the verdict
    cands.sort(key=lambda c: -c["strength"])
    return [{k: c[k] for k in ("text", "feature", "kind")} for c in cands[:3]]


def summary_line(prob: float, reasons: list[dict]) -> str:
    """'0.87 · looks like Minecraft gameplay + split screen · 3× the typical cut rate'"""
    content = [r["text"] for r in reasons if r["kind"] == "content"]
    pace = [r["text"] for r in reasons if r["kind"] == "pace"]
    parts = [f"{prob:.2f}"]
    if content:
        parts.append(" + ".join(content))
    if pace:
        parts.append(" · ".join(pace))
    return " · ".join(parts)
