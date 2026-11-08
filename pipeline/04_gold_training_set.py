# Databricks notebook source
# MAGIC %md # 04 · GOLD — training set with fixed splits
# MAGIC Joins pixel features + teacher labels and assigns each **creator** to one split, so the student can't
# MAGIC cheat by recognising an account. The 400 human-labelled videos are held out completely.
# MAGIC
# MAGIC | split | share | used for |
# MAGIC |---|---|---|
# MAGIC | train | 70% of creators | fitting + cross-validated tuning |
# MAGIC | val | 15% | threshold, model comparison |
# MAGIC | test | 15% | one final score vs the teacher, looked at ONCE |
# MAGIC | human | 400 videos | the headline numbers (notebook 06) |

# COMMAND ----------
from pyspark.sql import functions as F

SEED = 42
feats = spark.table("brainrot.silver.pixel_features")  # noqa: F821
labels = spark.table("brainrot.silver.teacher_labels").select("video_id", "label", "score")  # noqa: F821
human_ids = (spark.table("brainrot.silver.human_labels").select("video_id").distinct()  # noqa: F821
             if spark.catalog.tableExists("brainrot.silver.human_labels") else None)  # noqa: F821

df = feats.join(labels, "video_id").withColumn("creator", F.coalesce("creator", "video_id"))
# deterministic creator -> split assignment via a hash (stable across reruns)
bucket = F.abs(F.xxhash64(F.col("creator"), F.lit(SEED))) % 100
df = df.withColumn("split", F.when(bucket < 70, "train").when(bucket < 85, "val").otherwise("test"))
if human_ids is not None:
    df = df.join(human_ids.withColumn("is_human", F.lit(True)), "video_id", "left") \
           .withColumn("split", F.when(F.col("is_human"), "human").otherwise(F.col("split"))).drop("is_human")

df.write.mode("overwrite").saveAsTable("brainrot.gold.training_set")
df.groupBy("split", "view").agg(F.count("*").alias("n"), F.avg("label").alias("rot_share")).orderBy("split").show()
