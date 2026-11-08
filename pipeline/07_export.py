# Databricks notebook source
# MAGIC %md # 07 · Export the student for the API + load the labelled dataset into Tiger Data
# MAGIC Copy `student.joblib` to `artifacts/student.joblib` on the API server.

# COMMAND ----------
import json

import mlflow
import psycopg

model = mlflow.sklearn.load_model("models:/brainrot-student/latest")
model.save("/Volumes/brainrot/bronze/videos/student.joblib")
print("threshold:", model.threshold, "| typical values:", model.reference)

# COMMAND ----------
# MAGIC %md ### Labelled dataset → Tiger Data `dataset_videos` (time = date_posted)
# MAGIC Powers the dashboard's "brain rot vs normal engagement" and "brain rot over the years" panels.

# COMMAND ----------
meta = spark.table("brainrot.bronze.tiktok_meta_all").toPandas()  # noqa: F821
labels = spark.table("brainrot.silver.teacher_labels").toPandas()[["video_id", "label", "score"]]  # noqa: F821
meta = meta.merge(labels, on="video_id", how="inner")


def eng(s, *keys):
    d = json.loads(s) if isinstance(s, str) else (s or {})
    return next((d[k] for k in keys if k in d), None)


rows = [(r.date_posted, r.video_id, r.creator, str(r.is_ai_generated).lower() == "true", int(r.duration_ms or 0),
         eng(r.engagement_metrics, "play_count", "plays"), eng(r.engagement_metrics, "like_count", "likes"),
         eng(r.engagement_metrics, "comment_count", "comments"), eng(r.engagement_metrics, "share_count", "shares"),
         int(r.label), int(r.score)) for r in meta.itertuples()]

with psycopg.connect(dbutils.secrets.get("brainrot", "tiger_url")) as conn, conn.cursor() as cur:  # noqa: F821
    cur.executemany(
        "INSERT INTO dataset_videos (time, video_id, creator, is_ai_generated, duration_ms, plays, likes, "
        "comments, shares, brainrot, score) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING", rows)
print("loaded", len(rows))
