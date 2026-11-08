const CONFIG = {
  API_URL: "http://localhost:8000", // where `brainrot serve` runs
  MODE: "api", // "api" asks the Gemini model via API_URL, "demo" scores from motion locally
  // api mode only. "clip" records the playing video (with audio) and sends it to /predict_clip,
  // "screenshots" sends MAX_FRAMES screenshots to /predict. Falls back to screenshots if a video can't be recorded.
  SEND: "clip",
  CLIP_SECONDS: 8, // how much of each video to record
  CLIP_MIN_SECONDS: 3, // swiped away sooner than this = not scored
  CLIP_BITRATE: 1500000, // ~1.5 Mbit/s, so 8 s is about 1.5 MB
  EARLY_FRAMES: 4, // demo mode only: quick first score
  MAX_FRAMES: 8,
  FRAME_INTERVAL_MS: 500, // chrome allows ~2 screenshots/sec
  API_TIMEOUT_MS: 90000,
  DEMO_THRESHOLD: 0.6,
  KEEP_DAYS: 7,
};
