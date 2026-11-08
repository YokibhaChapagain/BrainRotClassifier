# Databricks notebook source
# MAGIC %md # 06 · Evaluate on HUMAN labels (the numbers that go on the slides)
# MAGIC * How good is the **student** (pixels only) vs humans? Live and full view.
# MAGIC * How good is the **teacher** (Gemini, text only) vs humans? Can the student match it without any text?
# MAGIC * **No-caption test**: student accuracy on videos whose caption is empty or only generic hashtags,
# MAGIC   where a text-based classifier has nothing to go on.
# MAGIC * Calibration (ECE) and human-human agreement.

# COMMAND ----------
import re

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, cohen_kappa_score, f1_score, roc_auc_score

from brainrot.models.student import StudentModel, rows_from_frame  # noqa: F401  (StudentModel for unpickling)

human = spark.table("brainrot.silver.human_labels").toPandas()  # noqa: F821
agree = human.groupby("video_id").filter(lambda g: len(g) > 1)  # double-labelled videos
if len(agree):
    pairs = agree.groupby("video_id").human_score.apply(lambda s: s.astype(float).tolist()[:2])
    print("human-human kappa:", cohen_kappa_score([p[0] >= 6 for p in pairs], [p[1] >= 6 for p in pairs]))
human = human.groupby("video_id", as_index=False).human_score.mean()
human["y"] = (human.human_score.astype(float) >= 6).astype(int)

gold = spark.table("brainrot.gold.training_set").where("split = 'human'").toPandas()  # noqa: F821
meta = spark.table("brainrot.bronze.tiktok_raw").select("video_id", "description").toPandas()  # noqa: F821
test = gold.merge(human, on="video_id").merge(meta, on="video_id")
model = mlflow.sklearn.load_model("models:/brainrot-student/latest")

GENERIC = re.compile(r"#?(fyp|foryou|foryoupage|viral|trending|xyzbca|capcut)\b", re.I)


def no_caption(desc: str) -> bool:
    return len(GENERIC.sub("", desc or "").strip(" #\n")) < 5


def ece(y, p, bins=10):
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


rows = []
for view in ("live", "full"):
    t = test[test.view == view]
    p = model.predict_proba(rows_from_frame(t))
    pred = p >= model.threshold
    nc = t.description.map(no_caption).values
    rows.append({
        "view": view, "n": len(t),
        "student ROC-AUC": roc_auc_score(t.y, p), "student PR-AUC": average_precision_score(t.y, p),
        "student F1": f1_score(t.y, pred), "student ECE": ece(t.y.values, p),
        "kappa student-human": cohen_kappa_score(pred, t.y), "kappa teacher-human": cohen_kappa_score(t.label, t.y),
        "no-caption n": int(nc.sum()),
        "no-caption student F1": f1_score(t.y[nc], pred[nc]) if nc.sum() > 5 else None,
    })
pd.DataFrame(rows).round(3)
