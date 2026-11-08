# brainrot-classifier-prototype

Classifies short videos as brain rot or not with Gemini, and a Chrome extension (`extension-counter/`) that counts brain rot videos while you scroll Instagram or TikTok.

## BrainRot Counter setup

The extension sends each video to the Gemini model running on your own computer, so there are two parts: start the model, then load the extension.

### 1. Get the code

Clone the repo, or download it as a zip and unzip it.

### 2. Start the model

You need Python 3.10 or newer. In the repo folder:

```
pip install -r requirements.txt
```

Copy `.env.example` to `.env` (`copy .env.example .env` on Windows, `cp .env.example .env` on Mac or Linux). Open `.env` and replace `your-key-here` with your Gemini API key from https://aistudio.google.com/apikey.

Start the server and leave the terminal open while you use the extension:

```
python -m uvicorn brainrot.web.server:app --port 8000
```

Check it at http://localhost:8000/health. You should see `"ok": true` and `"has_api_key": true`.

### 3. Load the extension in Chrome

1. Go to `chrome://extensions`.
2. Turn on Developer mode (top right).
3. Click Load unpacked and select the `extension-counter` folder inside the repo. If you get "Manifest file is missing", you picked the folder above it.
4. Click the puzzle-piece icon in the toolbar and pin BrainRot Counter.

### 4. Use it

- Open instagram.com/reels or tiktok.com and play a video. The counter appears in the top-right corner, and you can drag it anywhere.
- After about 4 seconds it shows "asking the model…", and the video's score arrives a few seconds later. A score of 50 or more counts as brain rot.
- Each video counts once, even if you scroll back to it. Videos you swipe past in under 4 seconds aren't counted.
- Click the toolbar icon to turn the counter on or off, see today's numbers, or reset them. The badge on the icon shows today's brain rot count.

### If something's wrong

| What you see | What to do |
| --- | --- |
| "model offline" | The server from step 2 isn't running, or isn't on port 8000. |
| Popup says the API key isn't set | Check `.env`, then restart the server. |
| No counter on the page | Refresh the Instagram or TikTok tab. The extension only attaches to pages opened after it was installed. |
| "model error" | Read the error in the server terminal. |
| You pulled new code | Click the reload arrow on the BrainRot Counter card in `chrome://extensions`, then refresh the tab. |

Everyone runs the model locally with their own key, because the extension looks for it at `localhost:8000`. To use a shared, deployed server instead, set `API_URL` in `extension-counter/config.js` to its address and skip step 2.

More detail on how the extension counts and what `/predict` returns is in [`extension-counter/README.md`](extension-counter/README.md).

## Demo feed (pre-classified)

This is how a platform would use the classifier: every video is scored once, ahead of time, and the stored verdict is read while people scroll. There's no waiting and no model call at view time.

1. Put some videos in `samples/`. The folder is gitignored, so the videos stay out of the repo.
2. Classify them once. Unchanged videos are skipped on later runs, so you only pay for new ones:
   ```
   python -m brainrot.cli index samples
   ```
3. Start the server (as in step 2 above) and open http://localhost:8000/feed.

Swipe or use the arrow keys to move between videos. Tap a video to pause it, and press `m` or the speaker button to unmute. The toggle at the top switches what the "platform" does with flagged videos:

- **Label**: show every video with its verdict.
- **Warn**: blur flagged videos behind a "watch anyway" screen.
- **Hide**: drop flagged videos from the feed.

To show a live classification, press **+** (or drop a video file onto the page). The video uploads, Gemini classifies it once (you'll see "Gemini is watching it…" with a timer), and it's added to the top of the feed with an "added" tag. After that its verdict is stored like every other video's, so it's never classified again.

On the free Gemini tier, each model allows only 20 requests per day, which covers `index` runs and uploads together. Plan the demo around that, or enable billing on the key.

The score button opens the full explanation: the reasoning, tropes, the 7 criteria and the locally measured cuts per minute. Verdicts are stored in `samples/feed.json`. Use `--force` to re-classify everything, and set `BRAINROT_FEED_DIR` to use a different folder.

## Command line

```
pip install -e .
brainrot classify video.mp4            # one or more files, folders or URLs
brainrot eval labels.csv               # accuracy against a labelled CSV (source,label[,context])
brainrot index samples/                # pre-classify a folder for the demo feed
brainrot serve                         # web UI, /feed demo and the extension's API on port 8000
```
