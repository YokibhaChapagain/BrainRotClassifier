// Point at local dev or the deployed API.
const BRAINROT_CONFIG = {
  API_URL: "http://localhost:8000", // e.g. "https://api.brainrotmeter.example"
  EARLY_FRAMES: 4,        // first (rough) prediction
  MAX_FRAMES: 8,          // final prediction — matches the training "live view" (4 s at 2 fps)
  FRAME_INTERVAL_MS: 500, // captureVisibleTab allows ~2 calls/sec
  SESSION_GAP_MS: 10 * 60 * 1000, // new scrolling session after 10 min idle
  EXPLAIN: true,          // ask Gemini to turn the reasons into a one-liner when a reel gets flagged
};
