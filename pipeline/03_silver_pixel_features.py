# Databricks notebook source
# MAGIC %md # 03 · SILVER — student features (PIXELS ONLY)
# MAGIC Two rows per video:
# MAGIC * `view = "full"`: whole video at 1 fps, cut rate over the whole video
# MAGIC * `view = "live"`: what the extension sees, the first 4 s at 2 fps (8 frames), cut rate over those 4 s
# MAGIC
# MAGIC Features: MOTION (cuts/sec, motion, split screen, saturation, brightness, colourfulness, flicker) and
# MAGIC CLIP (mean frame embedding + similarity to the concept bank in `brainrot/features/concepts.py`).
# MAGIC Output: `brainrot.silver.pixel_features`. A GPU cluster makes CLIP much faster but CPU works.

# COMMAND ----------
# MAGIC %pip install -q sentence-transformers opencv-python-headless
# MAGIC %pip install -q -e /Workspace/Repos/<you>/brainrot-classifier

# COMMAND ----------
import pandas as pd

from brainrot.features.video import pixel_features

ids = spark.table("brainrot.silver.teacher_labels").select("video_id")  # noqa: F821
vids = (spark.table("brainrot.bronze.tiktok_raw").join(ids, "video_id")  # noqa: F821
        .select("video_id", "creator", "video_path").toPandas())

rows = []
for d in vids.to_dict("records"):
    for view in ("full", "live"):
        try:
            f = pixel_features(video_path=d["video_path"], live=(view == "live"))
            rows.append({"video_id": d["video_id"], "creator": d["creator"], "view": view, **f["motion"],
                         "concepts": f["concepts"].tolist() if f["concepts"] is not None else None,
                         "clip_emb": f["clip_emb"].tolist() if f["clip_emb"] is not None else None})
        except Exception as e:  # noqa: BLE001  corrupt video etc.
            print("skip", d["video_id"], view, e)

feats = pd.DataFrame(rows)
spark.createDataFrame(feats).write.mode("overwrite").saveAsTable("brainrot.silver.pixel_features")  # noqa: F821
feats.groupby("view").size()
