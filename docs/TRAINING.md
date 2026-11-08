# How the model is trained (plain English)

## Teacher and student

| | Teacher | Student |
|---|---|---|
| Who | **Gemini 2.5 Flash** | **Logistic regression** (our model) |
| Sees | Only **text**: caption, hashtags, on-screen text, transcript (Whisper fills gaps) | Only **pixels**: 8–30 frames + cut rate |
| Job | Make the training labels (score 0–10, ≥ 6 = brain rot) | Learn to predict those labels without any text |
| When it runs | Once, offline, on ~3,000 videos | Every reel, live, in the extension |

The student has to find visual patterns that match the teacher's judgment. If it can, brain rot is visually
recognizable, and the model works on videos with no caption or misleading hashtags.

## The models used

| Job | Model | Trained by us? |
|---|---|---|
| Labels (teacher) | Gemini 2.5 Flash, prompted with `docs/LABELING_RUBRIC.md` | No (prompted) |
| Missing transcripts (for the teacher) | Whisper small | No |
| What's on screen (student input) | CLIP ViT-B/32 via sentence-transformers: frame embedding + concept similarities | No (frozen) |
| Pace and colour (student input) | OpenCV / NumPy code: cuts/sec, motion, split screen, saturation, brightness, colourfulness, flicker | No (just code) |
| **The brain rot classifier (student)** | **Logistic regression** + isotonic calibration (LightGBM only as a comparison) | **Yes** |

## Step by step

**1. Sample (bronze, `01`).** From 58k TikToks: 3–180 s, English, max 3 per creator, stratified by AI-flag ×
length, plus 25% "suspected rot" (meme hashtags) so positives aren't rare. ~4,000 MP4s go into a UC Volume.

**2. Teacher labels (silver, `02`).**
- Whisper transcribes videos with missing transcripts.
- Keep only videos with **rich text** (transcript or on-screen text ≥ 40 characters).
- Gemini scores each one: `{score, label, reasons, confidence, enough_info}`.
- Drop "not enough info" and confidence < 0.5.
- Export 400 for **human labels**. Humans *watch* the videos, without seeing Gemini's answer.

**3. Student features (silver, `03`).** Pixels only, two views per video:
- `full`: whole video at 1 frame/s
- `live`: first 4 s at 2 frames/s (8 frames), exactly what the extension captures

**4. Gold table (`04`).** Features + teacher labels, with each **creator** hashed into one split:

| Split | Share | Used for |
|---|---|---|
| train | 70% | fitting + cross-validated tuning |
| val | 15% | 🚩 threshold, comparing models |
| test | 15% | one final score vs the teacher, looked at once |
| human | 400 videos | headline numbers |

Splitting by creator stops the student from recognizing an account instead of the content.

**5. Train + tune (`05`).**
- For each feature group (**motion-only**, **CLIP-only**, **motion + CLIP**), try `C` ∈ {0.01 … 3}. `C` is
  the regularisation strength: smaller = simpler model.
- Score each setting with **3-fold cross-validation inside train**, grouped by creator, using **PR-AUC on
  the live view**. PR-AUC is robust when brain rot is the minority.
- Refit the best setting on all of train (full + live rows).
- **Calibrate** with isotonic regression so "0.8" really means ~80%.
- Pick the 🚩 threshold on val (best F1).
- One **LightGBM** run answers "would a non-linear model do better?".
- Everything is logged to **MLflow**. The best motion + CLIP logistic regression becomes `brainrot-student`.

**6. Evaluate on humans (`06`).**
- Student vs humans: ROC-AUC, PR-AUC, F1, calibration error (ECE), Cohen's κ
- Teacher vs humans: can the pixel student match a text-reading LLM?
- **No-caption subset**: videos whose caption is empty or only `#fyp #viral`
- Human vs human agreement (double-labelled videos): the ceiling for everyone

## Why logistic regression?

It's fast, it works well on CLIP features, and it's **explainable**. Each feature's contribution
(coefficient × standardized value) shows how much it pushed the score. That ordering drives the reasons in the
overlay, and the average coefficients become the "what makes brain rot recognizable" slide.

## What the output looks like

```json
{"prob": 0.87, "flagged": true, "label": "brainrot", "threshold": 0.6,
 "reasons": ["looks like Minecraft gameplay", "split screen", "3× the typical cut rate"],
 "summary": "0.87 · looks like Minecraft gameplay + split screen · 3× the typical cut rate",
 "n_frames": 8, "final": true}
```

- `prob`: calibrated probability, 0–1, shown as the meter
- `flagged`: `prob >= threshold`, shown as the 🚩
- "Typical" means the median of the training videos
- Reasons only use features the student actually measured
