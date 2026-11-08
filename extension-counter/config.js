const CONFIG = {
  API_URL: "http://localhost:8000",
  MODE: "demo", // "demo" scores from motion locally, "api" calls API_URL/predict
  EARLY_FRAMES: 4,
  MAX_FRAMES: 8,
  FRAME_INTERVAL_MS: 500, // chrome allows ~2 screenshots/sec
  DEMO_THRESHOLD: 0.6,
  KEEP_DAYS: 7,
};
