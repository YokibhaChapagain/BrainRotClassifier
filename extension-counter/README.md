# BrainRot Counter (Chrome extension)

Shows a counter on Instagram and TikTok while you scroll: how many reels today were brain rot, out of how many were scored. Turn it on and off from the toolbar icon.

## Install

1. Open `chrome://extensions` and switch on Developer mode.
2. Click "Load unpacked" and pick this folder.
3. Open instagram.com or tiktok.com and play a video. The counter appears in the top-right corner, and you can drag it anywhere.

After editing any file, click the reload icon on the extension's card, then refresh the Instagram or TikTok tab.

## Modes

Set `MODE` in `config.js`.

- `demo` (default): scores each reel from how much its frames change. No server is needed, so you can build and demo the counter now. It is not the trained model.
- `api`: sends the frames to `API_URL/predict` and uses the model's answer. Point `API_URL` at the deployed server for the demo.

The API has to accept this request and return at least `prob` and `flagged`:

```
POST /predict
{"video_id": "C9x...", "platform": "instagram" or "tiktok", "frames_b64": ["<jpeg base64>", ...], "cuts_per_sec": null}

200 {"prob": 0.87, "flagged": true, ...}
```

That matches `api/main.py` in the BrainRotClassifier repo, so no backend change is needed there.

## How counting works

- The biggest playing video on screen is the current reel.
- Its ID comes from the post's link in a feed (Instagram home, TikTok For You) or from the URL on a single-video page: `/reels/<id>`, `/reel/<id>`, `/p/<id>` on Instagram, `/@user/video/<id>` on TikTok.
- Both sites add to the same daily count. IDs are prefixed `ig:` or `tt:` so they never collide.
- A screenshot of the video area is taken every 0.5 s. The reel gets an early score after 4 frames (about 2 s) and a final one after 8 (about 4 s).
- Each score is stored under the reel's ID for today, so a reel counts once even if you scroll back to it. The final score replaces the early one.
- Reels swiped away before 2 s, or scored while the API is down, are not counted.
- If the counter sits on top of the video, it is hidden for each screenshot so it doesn't end up in the frames.
- Counts reset each day (local time) and the last 7 days are kept. The badge on the icon shows today's brain rot count.

## Files

- `manifest.json`: Chrome MV3 manifest
- `config.js`: API address, mode, timings
- `content.js`: runs on instagram.com and tiktok.com, finds the reel, takes frames, draws the counter
- `background.js`: screenshots, API calls, demo scorer, storage, badge
- `popup.html`, `popup.js`: on/off switch, today's numbers, reset
- `overlay.css`: counter style
