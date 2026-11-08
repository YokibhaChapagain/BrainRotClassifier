# Architecture — BrainRot Meter (UB Hacking 2026)

**Tracks:** Best AI/ML (primary) · Best Use of Gemini API · Best Use of Tiger Data
**Core idea:** a **teacher** (Gemini) labels videos from their *text*; a **student** learns to predict those
labels from *pixels only*. Short brief: [`BRIEF.md`](BRIEF.md). Training details: [`TRAINING.md`](TRAINING.md).

```
 ┌──────────────────────────────── OFFLINE · Databricks ────────────────────────────────┐
 │ BRONZE                 SILVER                                  GOLD                  │
 │ 01 sample ~4k     ─►   02 TEACHER: Whisper fills transcripts   04 training_set       │
 │ TikToks (creator       → Gemini reads TEXT ONLY → 0-10      ─► (features + labels    │
 │ cap, stratified)       03 STUDENT FEATURES: PIXELS ONLY         + creator splits)    │
 │ MP4s → UC Volume       motion (cuts, split screen, colour)      │                    │
 │                        + CLIP (embedding + concepts)            ▼                    │
 │                                                     05 train student + ablation      │
 │                                                     06 evaluate vs 400 human labels  │
 │                                                     07 export → student.joblib       │
 └──────────────────────────────────────────────────────────────────┬───────────────────┘
                                                                    │ MLflow registry
 ┌──────────────────────────────── ONLINE · FastAPI (Docker) ───────▼───────────────────┐
 │ Chrome extension ── 8 frames + in-page cut rate ──► POST /predict ─► student          │
 │   meter + 🚩 + "0.87 · looks like Minecraft gameplay + split screen · 3× cut rate"    │
 │   POST /explain ─► Gemini turns the reasons + 2 frames into a one-liner               │
 │   POST /events/view ─► Tiger Data (watch time, skips, loops, score)                   │
 │ Dashboard  ◄── GET /stats/* ◄── continuous aggregates in Tiger Data                   │
 └───────────────────────────────────────────────────────────────────────────────────────┘
```

Hosting: Docker on a Vultr VM behind Caddy with a GoDaddy domain (plain hosting, not prize targets).

## What came from which design

| Kept from the "pixels-only" plan | Kept from the original plan |
|---|---|
| Teacher reads text, student sees pixels only (cross-modal distillation) | **Gemini** as the teacher (Gemini track) |
| Motion features + **colour intensity** + CLIP content | Chrome extension with a live-updating meter |
| **CLIP concept bank**, so explanations name the content | **Tiger Data** dashboard + per-view logging |
| Explanations like "3× the typical cut rate" | 400 **human-labelled** test videos |
| Bronze / silver / **gold** Delta tables, logistic regression | Creator-grouped splits, calibration, tuned threshold |
| Ablation: motion-only vs motion + CLIP | "Live view" training (score on exactly what the extension sees) |

Added to fix weak spots:
- **Rich-text filter + Whisper.** The teacher only labels videos with real text (transcript or on-screen
  text), and Whisper fills missing transcripts. This avoids labels from `#fyp #viral` alone.
- **"Not enough info" option.** Gemini can say it can't judge, and those videos are dropped.
- **No-caption test.** The headline evaluation shows the student working where text-based approaches can't.

Dropped: audio features (Whisper is only used for the teacher's transcripts), text features, the slang-score
dataset, metadata features, and the AI-generated flag. The student is **pixels only**.

## Pipeline

| Step | Notebook | Layer | What happens |
|---|---|---|---|
| 0 | `docs/LABELING_RUBRIC.md` | — | Team agrees what "brain rot" means (0–10, ≥ 6 = rot) |
| 1 | `01_bronze_ingest.py` | bronze | Filter, max 3 per creator, stratified sample + 25% "suspected rot", download MP4s |
| 2 | `02_silver_teacher_labels.py` | silver | Whisper fills transcripts → rich-text filter → Gemini (text only) → `teacher_labels`; export 400 for humans |
| 3 | `03_silver_pixel_features.py` | silver | Motion + colour + CLIP for a `full` and a `live` (first 4 s, 8 frames) view |
| 4 | `04_gold_training_set.py` | gold | Join + hash creators into train / val / test; human-labelled videos → `human` |
| 5 | `05_train_student.py` | — | Tune C per feature group, ablation table, LightGBM comparison, register `brainrot-student` |
| 6 | `06_evaluate_human.py` | — | Student & teacher vs humans, no-caption subset, calibration |
| 7 | `07_export.py` | — | `student.joblib` for the API; labelled dataset → Tiger Data |

## Student features (all from pixels)

| Group | Feature | How |
|---|---|---|
| motion | `cuts_per_sec` | HSV-histogram jumps at 10 fps (offline: OpenCV; live: measured in-page by the extension) |
| motion | `motion_energy` | mean pixel change between frames |
| motion | `split_screen` | top and bottom halves move independently, or one is still while the other moves |
| motion | `saturation`, `brightness`, `colorfulness` | colour intensity (Hasler–Süsstrunk colourfulness) |
| motion | `flicker` | brightness variation across frames |
| clip | 16 concept probabilities | frames vs prompts like "Minecraft parkour gameplay", "a person cooking" (`brainrot/features/concepts.py`) |
| clip | 512-d embedding | mean CLIP ViT-B/32 image embedding (sentence-transformers) |

## API output (`POST /predict`)

```json
{
  "prob": 0.87, "flagged": true, "label": "brainrot", "threshold": 0.6,
  "reasons": ["looks like Minecraft gameplay", "split screen", "3× the typical cut rate"],
  "summary": "0.87 · looks like Minecraft gameplay + split screen · 3× the typical cut rate",
  "top_concepts": [{"name": "Minecraft gameplay", "p": 0.61}, ...],
  "pace": {"cuts_per_sec": 1.9, "cut_rate_vs_typical": 3.1, "split_screen": 0.82, ...},
  "n_frames": 8, "final": true, "model_version": "student-motion+clip"
}
```

Reasons only come from things the student measured, ordered by each feature's logistic-regression contribution.

## Extension (`extension/`)

- **Reads no text.** It only takes a reel ID (for caching and the dashboard), frames, and an in-page cut rate.
- Captures a frame every 0.5 s and calls `/predict` at 4 frames (early estimate) and 8 (final), so the meter animates.
- When a reel is flagged, asks Gemini (`/explain`) to turn the reasons + 2 frames into a playful one-liner.
- On reel change, logs the view: start time, watch time, loops, skipped, score. Sessions reset after 10 min idle.

## Tiger Data + dashboard

| Object | Type | Powers |
|---|---|---|
| `reel_views` | hypertable | every watched reel |
| `views_hourly` → `views_daily` | real-time hierarchical continuous aggregates | timeline, time spent, heatmap, KPIs |
| `dataset_videos` | hypertable (time = date posted) | rot vs normal engagement, rot share by year |

Dashboard panels:
- **KPIs:** % of feed that's rot, minutes on rot, % of watch time, reels watched, longest rot streak, skip rate
- **Charts:** exposure timeline, time spent, weekday × hour heatmap, engagement comparison, trend by year
- **Table:** recent sessions

`scripts/seed_demo_views.py` adds **synthetic** history for user `demo`, for building the UI only.

**Later (not now):** "does brain rot make me scroll more?" The data is already logged (watch time, skips,
sessions), so this needs only new queries.

## Timeline (~24 h, 4 people)

| Hours | A: data | B: ML | C: backend + Tiger | D: extension + dashboard |
|---|---|---|---|---|
| 0–2 | Rubric, Databricks setup, `01` | Gemini key, `02` on 50 videos, check labels | Tiger service, `schema.sql`, API skeleton | Extension skeleton, overlay |
| 2–8 | `02` all labels, export gold CSV | `03` pixel features, tune concept bank | `/events/view`, `/stats/*`, seed demo data | Frame capture, cut detector, view events |
| 8–14 | Human labeling (all 4) | `04` + `05`, ablation table | Deploy API, load student | Dashboard page, popup |
| 14–20 | `06` evaluation | Error analysis, improve weak concepts | `07` dataset → Tiger, `/explain` | Test on real TikTok/IG |
| 20–24 | Slides | Devpost (use `BRIEF.md`) | Freeze & deploy | Demo video |
