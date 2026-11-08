# Databricks notebook source
# MAGIC %md # 01 · BRONZE — ingest + sample Shofo TikTok → Delta + videos → UC Volume
# MAGIC
# MAGIC **Sampling strategy** (the dataset is unlabelled, so we sample smartly before paying for Gemini labels):
# MAGIC 1. Filter: duration 3–180 s, English, has a video file.
# MAGIC 2. **Creator cap**: at most 3 videos per creator, so one prolific account can't dominate (and so we can
# MAGIC    split train/val/test by creator later without leakage).
# MAGIC 3. **Stratify** by `is_ai_generated` × duration bucket, so short/long and AI/non-AI are all represented.
# MAGIC 4. **Enrich positives**: brain rot may be a minority of TikTok. We add an extra slice of "suspected rot"
# MAGIC    (slang / meme hashtags) so the labelled set isn't 95% negatives. Gemini still decides the label.

# COMMAND ----------
# MAGIC %pip install -q huggingface_hub
# MAGIC %pip install -q -e /Workspace/Repos/<you>/brainrot-classifier

# COMMAND ----------
import glob
import os

import pandas as pd
from huggingface_hub import snapshot_download
from pyspark.sql import Window
from pyspark.sql import functions as F

CATALOG, SCHEMA = "brainrot", "bronze"
VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/videos"
DATASET = "Shofo/shofo-tiktok-general-small"
N_SAMPLE = 4000          # videos to download + label; raise once the pipeline works end-to-end
SUSPECT_FRACTION = 0.25  # share of the sample reserved for "suspected rot"
PER_CREATOR = 3
SEED = 42

for stmt in (f"CREATE CATALOG IF NOT EXISTS {CATALOG}", f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}",
             f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.silver", f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.gold",
             f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.videos"):
    spark.sql(stmt)  # noqa: F821

# COMMAND ----------
# 1) metadata only (small), no videos yet
local = snapshot_download(DATASET, repo_type="dataset", allow_patterns=["*.parquet", "*.jsonl", "*.csv"],
                          local_dir="/tmp/shofo_meta")
files = glob.glob(f"{local}/**/*.parquet", recursive=True)
meta = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
meta = meta.drop(columns=[c for c in ("video",) if c in meta.columns])  # drop raw bytes if embedded
meta["video_id"] = meta["video_id"].astype(str)
df = spark.createDataFrame(meta.astype({c: str for c in meta.columns if meta[c].dtype == object}))  # noqa: F821
df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.tiktok_meta_all")  # all ~58k rows (used by dashboard)
print("rows:", df.count())

# COMMAND ----------
# 2) filter + creator cap
from brainrot.labeling.rubric import SLANG_REGEX as SLANG_RE  # noqa: E402  (sampling only, never a feature)

pool = (df.where(F.col("duration_ms").between(3_000, 180_000))
          .where(F.col("language").contains('"en'))
          .withColumn("dur_bucket", F.when(F.col("duration_ms") < 15_000, "short")
                                     .when(F.col("duration_ms") < 60_000, "mid").otherwise("long"))
          .withColumn("suspect", F.concat_ws(" ", "description", "hashtags", "sticker_text").rlike(SLANG_RE))
          .withColumn("strata", F.concat_ws("_", F.col("is_ai_generated").cast("string"), "dur_bucket"))
          .withColumn("rn", F.row_number().over(Window.partitionBy("creator").orderBy(F.rand(SEED))))
          .where(F.col("rn") <= PER_CREATOR))

# 3) suspected-rot slice + stratified random slice
n_suspect = int(N_SAMPLE * SUSPECT_FRACTION)
suspects = pool.where("suspect").orderBy(F.rand(SEED)).limit(n_suspect)
rest = pool.join(suspects.select("video_id"), "video_id", "left_anti")
counts = {r["strata"]: r["count"] for r in rest.groupBy("strata").count().collect()}
total = sum(counts.values())
fractions = {k: min(1.0, (N_SAMPLE - n_suspect) / total * 1.1) for k in counts}  # proportional, +10% slack
strat = rest.sampleBy("strata", fractions, seed=SEED).limit(N_SAMPLE - n_suspect)
sample = suspects.unionByName(strat)
sample.groupBy("strata", "suspect").count().show()

# COMMAND ----------
# 4) download only the sampled MP4s into the Volume (repo layout: videos/<NN>/<video_id>.mp4)
ids = [r.video_id for r in sample.select("video_id").collect()]
snapshot_download(DATASET, repo_type="dataset", local_dir=VOLUME,
                  allow_patterns=[f"videos/*/{vid}.mp4" for vid in ids], max_workers=16)
paths = {os.path.basename(p)[:-4]: p for p in glob.glob(f"{VOLUME}/videos/*/*.mp4")}
path_df = spark.createDataFrame(list(paths.items()), "video_id string, video_path string")  # noqa: F821

(sample.join(path_df, "video_id", "left").withColumn("has_video", F.col("video_path").isNotNull())
       .drop("rn").write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.tiktok_raw"))
print("downloaded videos:", len(paths))
