# Databricks notebook source
# MAGIC %md # 05 · Train the STUDENT (pixels only) + ablation, tracked in MLflow
# MAGIC
# MAGIC For each feature group (**motion-only**, **CLIP-only**, **motion + CLIP**):
# MAGIC 1. Tune `C` (regularisation) with 3-fold cross-validation inside `train`, grouped by creator, scoring
# MAGIC    PR-AUC on the **live** view (that's what the demo sees).
# MAGIC 2. Refit on all of `train` (full + live rows) with isotonic calibration.
# MAGIC 3. Pick the 🚩 threshold on `val`.
# MAGIC Then one LightGBM row asks "does a non-linear model beat logistic regression?".
# MAGIC The best motion+CLIP logistic regression is registered as `brainrot-student`.

# COMMAND ----------
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from brainrot.models.student import StudentModel, rows_from_frame

SEED = 42
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
df = spark.table("brainrot.gold.training_set").toPandas()  # noqa: F821
train, val, test = (df[df.split == s] for s in ("train", "val", "test"))
live = lambda d: d[d.view == "live"]  # noqa: E731


def metrics(model, frame) -> dict:
    p = model.predict_proba(rows_from_frame(frame))
    return {"roc_auc": roc_auc_score(frame.label, p), "pr_auc": average_precision_score(frame.label, p),
            "f1": f1_score(frame.label, p >= model.threshold)}


def cv_score(groups: str, params: dict) -> float:
    vids = train.drop_duplicates("video_id")
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    scores = []
    for tr, va in cv.split(vids, vids.label, groups=vids.creator):
        tr_ids, va_ids = set(vids.iloc[tr].video_id), set(vids.iloc[va].video_id)
        f_tr, f_va = train[train.video_id.isin(tr_ids)], live(train[train.video_id.isin(va_ids)])
        m = StudentModel(groups, params).fit(rows_from_frame(f_tr), f_tr.label.values)
        scores.append(average_precision_score(f_va.label, m.predict_proba(rows_from_frame(f_va))))
    return float(np.mean(scores))


mlflow.set_experiment("/Shared/brainrot")
results = []
candidates = [(g, {"model": "logreg", "C": c}) for g in ("motion", "clip", "motion+clip") for c in C_GRID]
candidates += [("motion+clip", {"model": "lgbm"})]

with mlflow.start_run(run_name="student-search"):
    for groups, params in candidates:
        with mlflow.start_run(run_name=f"{groups}-{params}", nested=True):
            s = cv_score(groups, params)
            mlflow.log_params({"groups": groups, **params})
            mlflow.log_metric("cv_pr_auc_live", s)
            results.append({"groups": groups, **params, "cv_pr_auc_live": s})
table = pd.DataFrame(results).sort_values("cv_pr_auc_live", ascending=False)
table

# COMMAND ----------
# MAGIC %md ### Refit the best setting per group → ablation table (val, live view)

# COMMAND ----------
ablation, best_models = [], {}
for groups in ("motion", "clip", "motion+clip"):
    best = table[(table.groups == groups) & (table.model == "logreg")].iloc[0]
    m = StudentModel(groups, {"model": "logreg", "C": float(best.C)}, version=f"student-{groups}")
    m.fit(rows_from_frame(train), train.label.values)
    m.tune_threshold(rows_from_frame(live(val)), live(val).label.values)
    best_models[groups] = m
    ablation.append({"features": groups, "C": best.C, **{f"val_live_{k}": v for k, v in metrics(m, live(val)).items()},
                     **{f"val_full_{k}": v for k, v in metrics(m, val[val.view == "full"]).items()}})
lgbm = StudentModel("motion+clip", {"model": "lgbm"}, version="student-lgbm").fit(rows_from_frame(train),
                                                                                  train.label.values)
lgbm.tune_threshold(rows_from_frame(live(val)), live(val).label.values)
ablation.append({"features": "motion+clip (LightGBM)", **{f"val_live_{k}": v for k, v in metrics(lgbm, live(val)).items()}})
ablation = pd.DataFrame(ablation)
ablation  # <- slide: "what makes brain rot visually recognizable"

# COMMAND ----------
final = best_models["motion+clip"]
print("TEST vs teacher (live view):", metrics(final, live(test)))  # looked at once
print("Most important named features:", final.global_importance())
with mlflow.start_run(run_name="student-final"):
    mlflow.log_params({"groups": "motion+clip", **final.params, "threshold": final.threshold})
    mlflow.log_metrics({f"test_live_{k}": v for k, v in metrics(final, live(test)).items()})
    mlflow.log_table(ablation, "ablation.json")
    mlflow.log_table(pd.DataFrame(final.global_importance(), columns=["feature", "coef"]), "importance.json")
    mlflow.sklearn.log_model(final, "model", registered_model_name="brainrot-student")
