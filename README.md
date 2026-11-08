# 🧠🫠 BrainRot Meter

**UB Hacking 2026.** Scores any short video for brain rot **from the pixels alone** and explains why:
**"0.87 · looks like Minecraft gameplay + split screen · 3× the typical cut rate."** It runs live as a Chrome
extension on TikTok and Instagram Reels, with a dashboard of your exposure over time.

**Teacher → student:** Gemini (the teacher) labels ~3,000 TikToks by reading only their text; a pixels-only
logistic regression (the student) learns to predict those labels from frames: pace, colour and CLIP content.

**Tracks:** Best AI/ML · Best Use of Gemini API · Best Use of Tiger Data

- 📝 Brief: [`docs/BRIEF.md`](docs/BRIEF.md)
- 📐 Architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- 🧪 Training: [`docs/TRAINING.md`](docs/TRAINING.md)
- 🏷️ Labeling rubric: [`docs/LABELING_RUBRIC.md`](docs/LABELING_RUBRIC.md)
- 🏆 Tracks: [`docs/TRACKS.md`](docs/TRACKS.md)

## Data

[Shofo/shofo-tiktok-general-small](https://huggingface.co/datasets/Shofo/shofo-tiktok-general-small):
~58k TikToks (MP4 + metadata). We sample ~4k, the teacher labels ~3k of them, and the student trains on those.

## Repo layout

```
brainrot/            shared Python package (used by Databricks jobs AND the API)
  labeling/          rubric + Gemini teacher (text only)
  features/          video.py (pixels: motion, colour, CLIP), concepts.py (CLIP concept bank),
                     audio.py (Whisper transcripts for the teacher only)
  models/student.py  pixels-only logistic regression + calibration + contributions
  explain.py         reasons + "0.87 · ..." summary line
pipeline/            Databricks notebooks 01 → 07 (bronze → silver → gold → train → evaluate → export)
api/                 FastAPI: /predict, /explain, /events/view, /stats/*, /dashboard
extension/           Chrome MV3 extension (pixels only: live meter, reasons, view logging, popup)
db/                  Tiger Data schema (hypertables + continuous aggregates)
scripts/             seed_demo_views.py (synthetic data for building the dashboard)
deploy/              docker-compose + Caddy
tests/               pytest
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt            # API + tests
pip install -r requirements-ml.txt         # CLIP, Whisper, OpenCV (pipeline + CLIP-enabled API)
cp .env.example .env                       # GEMINI_API_KEY, DATABASE_URL (Tiger Data), USE_CLIP
pytest
uvicorn api.main:app --reload              # http://localhost:8000/docs  ·  /dashboard?user=demo
```

Tiger Data: create a service, then run `psql "$DATABASE_URL" -f db/schema.sql`.
Optionally run `python scripts/seed_demo_views.py` for synthetic demo history.

Extension: `chrome://extensions` → Developer mode → **Load unpacked** → `extension/`. Set `API_URL` in
`extension/config.js`.

Until `artifacts/student.joblib` exists, the API uses a simple motion heuristic so the extension can be built in
parallel. Set `USE_CLIP=0` to run without downloading CLIP.

## Team

| Name | Role |
|---|---|
| Prasansha | |
| | |
| | |
| | |
