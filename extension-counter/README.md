# BrainRot Counter (Chrome extension)

Shows a counter on Instagram and TikTok while you scroll: how many reels today were brain rot, out of how many were scored. Turn it on and off from the toolbar icon.

## Install

1. Open `chrome://extensions` and switch on Developer mode.
2. Click "Load unpacked" and pick this folder.
3. Open instagram.com or tiktok.com and play a video. The counter appears in the top-right corner, and you can drag it anywhere.

After editing any file, click the reload icon on the extension's card, then refresh the Instagram or TikTok tab.

## Run the model first

The extension asks the Gemini classifier in this repo. Start it before you scroll:

```bash
pip install -r requirements.txt
cp .env.example .env        # put your GEMINI_API_KEY in it
brainrot serve              # or: uvicorn brainrot.web.server:app --port 8000
```

The extension's popup says "Model online" when it can reach the server, or tells you if the API key is missing.

## Modes

Set `MODE` in `config.js`.

- `api` (default): asks the Gemini model in this repo, one call per video. What gets sent depends on `SEND`:
  - `SEND: "clip"` (default): records the first `CLIP_SECONDS` (8 s) of the playing video, with sound, straight from the `<video>` element (`captureStream` + `MediaRecorder`), and uploads it to `API_URL/predict_clip`. The server converts it to mp4 and runs the normal `brainrot classify` pipeline, so `BRAINROT_MODE`, `BRAINROT_MAX_FRAMES` etc. apply and Gemini gets real motion, audio and cuts/minute. Recording reuses what the browser already downloaded to show you the video, so it sends **no extra requests to Instagram or TikTok**. If a video can't be recorded (Chrome refuses for some cross-origin sources), that video falls back to screenshots.
  - `SEND: "screenshots"`: after about 4 seconds (8 screenshots, 0.5 s apart) the frames go to `API_URL/predict`. No audio.
  
  The answer usually takes a few seconds (longer for clips); it still counts if you have already swiped to the next video.
- `demo`: scores each reel from how much its frames change, with no server. Use it only to try the counter without a key. It is not the model.

`API_URL` is `http://localhost:8000`. If the server runs somewhere else, change it, and for anyone else using the extension it has to be a deployed address they can reach.

The request and answer look like this:

```
POST /predict
{"video_id": "C9x...", "platform": "instagram" or "tiktok", "frames_b64": ["<jpeg base64>", ...], "cuts_per_sec": null}

200 {"prob": 0.82, "flagged": true, "score": 82, "label": "brain_rot", "reasons": ["Subway Surfers split-screen"], ...}
```

```
POST /predict_clip   (multipart form)
file=<clip.webm>  video_id=7301...  platform=tiktok

200 same fields as /predict, plus "mode" and "signals" (duration, cuts/min, motion, volume)
```

`prob` is the 0 to 100 score divided by 100, and `flagged` means the score reached `BRAINROT_THRESHOLD` (50 by default). The server caches answers by video ID, so scrolling back to a video doesn't cost a second call.

## How counting works

- The biggest playing video on screen is the current reel.
- Its ID comes from the post's link in a feed (Instagram home, TikTok For You) or from the URL on a single-video page: `/reels/<id>`, `/reel/<id>`, `/p/<id>` on Instagram, `/@user/video/<id>` on TikTok.
- Both sites add to the same daily count. IDs are prefixed `ig:` or `tt:` so they never collide.
- Clip mode: the video is recorded for 8 s of actual playback (pauses and background tabs don't count). If you swipe away after at least `CLIP_MIN_SECONDS` (3 s), the shorter clip is sent; before that, nothing is sent.
- Screenshot mode: a screenshot of the video area is taken every 0.5 s. After 8 screenshots (about 4 s) they are sent to the model. In demo mode there is also an early score after 4. If the counter sits on top of the video, it is hidden for each screenshot so it doesn't end up in the frames.
- Each score is stored under the reel's ID for today, so a reel counts once even if you scroll back to it.
- Videos swiped away too early, or scored while the server is down, are not counted.
- Counts reset each day (local time) and the last 7 days are kept. The badge on the icon shows today's brain rot count.

## Files

- `manifest.json`: Chrome MV3 manifest
- `config.js`: API address, mode, timings
- `content.js`: runs on instagram.com and tiktok.com, finds the reel, records the clip (or takes frames), draws the counter
- `background.js`: clip upload, screenshots, model calls, demo scorer, storage, badge
- `popup.html`, `popup.js`: on/off switch, today's numbers, reset
- `overlay.css`: counter style
