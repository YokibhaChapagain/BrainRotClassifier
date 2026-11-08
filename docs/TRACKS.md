# Prize tracks

| Track | What makes it count | Where |
|---|---|---|
| **Best AI/ML** (primary) | Cross-modal teacher → student distillation (text → pixels), CLIP concept features, motion vs CLIP ablation, creator-grouped splits, calibration, explainable reasons, human-labelled evaluation incl. no-caption videos | `pipeline/`, `brainrot/models/student.py`, `brainrot/explain.py` |
| **Best Use of Gemini API** (MLH) | Gemini is the **teacher** that labels the training set against our rubric, and narrates flags live in the overlay | `brainrot/labeling/gemini_labeler.py`, `/explain` |
| **Best Use of Tiger Data** (MLH) | Hypertables, real-time hierarchical continuous aggregates (hourly → daily), a personal brain rot exposure dashboard, dataset trend by year | `db/schema.sql`, `api/db.py`, `/dashboard` |

Possible bonuses with no extra work: Best General Hack and Best UI/UX (overlay + dashboard).
Hosting on Vultr + a GoDaddy domain is fine but not a target. ElevenLabs was dropped.

Rule of thumb: every sponsor technology must be visible in the 2-minute demo.
