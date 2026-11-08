# Databricks notebook source
# MAGIC %md # 02 · SILVER — teacher labels (Gemini reads TEXT ONLY)
# MAGIC
# MAGIC The **teacher** sees the caption, hashtags, on-screen text and transcript, never the pixels.
# MAGIC 1. Whisper fills in missing transcripts (so more videos have real text to judge).
# MAGIC 2. Keep only videos with rich text (transcript or on-screen text ≥ 40 chars). `#fyp #viral` alone is useless.
# MAGIC 3. Gemini scores each one 0–10 against the rubric. Drop low-confidence or "not enough info" answers.
# MAGIC
# MAGIC Output: `brainrot.silver.teacher_labels(video_id, score, label, confidence, reasons)`
# MAGIC Key in a secret scope: `databricks secrets put-secret brainrot gemini_key`

# COMMAND ----------
# MAGIC %pip install -q google-genai openai-whisper
# MAGIC %pip install -q -e /Workspace/Repos/<you>/brainrot-classifier

# COMMAND ----------
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from brainrot.features.audio import transcribe
from brainrot.labeling.gemini_labeler import GeminiTeacher
from brainrot.schema import Reel

os.environ["GEMINI_API_KEY"] = dbutils.secrets.get("brainrot", "gemini_key")  # noqa: F821
MIN_CONFIDENCE = 0.5
MAX_WORKERS = 8  # lower if Gemini returns 429s

raw = spark.table("brainrot.bronze.tiktok_raw").where("has_video").toPandas()  # noqa: F821
reels = [Reel.from_hf_row(d) for d in raw.to_dict("records")]

# COMMAND ----------
# 1) Whisper for missing transcripts (GPU strongly recommended; ~1-2 s per short video)
missing = [r for r in reels if len(r.transcript.strip()) < 40]
print(f"transcribing {len(missing)} of {len(reels)}")
for r in missing:
    try:
        r.transcript = transcribe(r.video_path)
    except Exception as e:  # noqa: BLE001
        print("whisper failed", r.video_id, e)
spark.createDataFrame(pd.DataFrame([{"video_id": r.video_id, "transcript": r.transcript} for r in reels])) \
    .write.mode("overwrite").saveAsTable("brainrot.silver.transcripts")  # noqa: F821

# COMMAND ----------
# 2) rich-text filter + 3) Gemini teacher
rich = [r for r in reels if r.has_rich_text()]
print(f"{len(rich)} videos have enough text for the teacher")
teacher = GeminiTeacher()


def label(r: Reel) -> dict:
    try:
        t = teacher.label(r)
        return {"video_id": r.video_id, "score": t.score, "label": t.label, "confidence": t.confidence,
                "enough_info": t.enough_info, "reasons": "; ".join(t.reasons)}
    except Exception as e:  # noqa: BLE001
        return {"video_id": r.video_id, "score": None, "label": None, "confidence": None,
                "enough_info": False, "reasons": f"ERR {e}"}


with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    labels = pd.DataFrame(list(ex.map(label, rich)))

kept = labels[labels.enough_info & (labels.confidence >= MIN_CONFIDENCE)].dropna(subset=["label"])
kept = kept.astype({"score": int, "label": int})
print(f"kept {len(kept)} / {len(labels)}  ·  brain rot share = {kept.label.mean():.2%}")
spark.createDataFrame(kept).write.mode("overwrite").saveAsTable("brainrot.silver.teacher_labels")  # noqa: F821

# COMMAND ----------
# MAGIC %md ### Alternative: label in SQL with `ai_query`
# MAGIC If you register Gemini as an external-model serving endpoint (check Databricks supports this on your
# MAGIC workspace), the same step is one SQL query, nice to show on Databricks:
# MAGIC ```sql
# MAGIC CREATE OR REPLACE TABLE brainrot.silver.teacher_labels_sql AS
# MAGIC SELECT video_id,
# MAGIC        ai_query('gemini-teacher', concat(:teacher_prompt, teacher_text),
# MAGIC                 responseFormat => 'STRUCT<score INT, label INT, confidence DOUBLE, enough_info BOOLEAN>') AS out
# MAGIC FROM brainrot.silver.teacher_inputs;
# MAGIC ```

# COMMAND ----------
# MAGIC %md ### Human gold set (the real test)
# MAGIC 400 videos, stratified by teacher label. Humans **watch the video** (not just read the text) and score
# MAGIC it with the rubric, without seeing Gemini's answer. ~50 are double-labelled to measure agreement.
# MAGIC Upload as `brainrot.silver.human_labels(video_id, human_score, annotator)`.

# COMMAND ----------
meta = raw[["video_id", "web_url"]].astype({"video_id": str})
gold = kept.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), 200), random_state=7))
gold.merge(meta, on="video_id")[["video_id", "web_url"]].assign(human_score="", annotator="") \
    .sample(frac=1, random_state=7).to_csv("/Volumes/brainrot/bronze/videos/gold_to_label.csv", index=False)
