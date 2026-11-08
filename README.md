# Brain Rot Feed

Brain Rot Feed uses Google Gemini to decide whether a short video is "brain rot": low-value, overstimulating content made only to hold your attention, such as Skibidi edits, Subway Surfers split-screens, AI slop and TTS Reddit stories. Each video gets a 0 to 100 score, seven criteria scores and a written explanation.

The repo has three parts:

- **The classifier** (`brainrot/`): local video analysis plus one Gemini call per video.
- **The demo feed** (`/feed`): a vertical, swipeable video feed where every video was classified ahead of time. This is how a platform would use it.
- **A Chrome extension** (`extension-counter/`): counts brain rot videos while you scroll Instagram or TikTok.

## How the app works

A platform doesn't need to classify videos while people watch. It classifies each video **once, when it's uploaded**, stores the verdict, and reads that stored verdict every time someone scrolls past.

```
 AT UPLOAD (once per video, ~10 s)                         WHILE SCROLLING (every view, instant)

 video file ──► local analysis ──► Gemini ──► verdict  ═══► feed loads ──► policy ──► viewer sees
                (OpenCV + ffmpeg)  (rubric,    stored in      the stored     Label /     score + "why"
                keyframes, audio,   JSON out)  feed.json      verdicts       Warn / Hide  panel
                cuts per minute
```

1. **A video arrives**, through the **+** button in the feed or the `brainrot index` command.
2. **Local analysis** (no AI) measures how the video is edited and picks the frames worth sending.
3. **Gemini classifies it.** It gets the keyframes, the audio, the measured pacing and a written rubric, and returns a structured verdict.
4. **The verdict is stored** in `feed.json`, next to the video file.
5. **The feed reads stored verdicts** and never calls Gemini while someone scrolls. That's why scrolling is instant even though one Gemini call takes about 10 seconds.
6. **A policy decides what the viewer sees**: label every video, put flagged videos behind a warning screen, or hide them from the feed.

## Quick start

You need Python 3.10 or newer and ffmpeg (`brew install ffmpeg` on a Mac).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and set `GEMINI_API_KEY` to a key from https://aistudio.google.com/apikey. Then:

```bash
python -m brainrot.cli index samples        # classify every video in samples/ once
python -m uvicorn brainrot.web.server:app --port 8000
```

Open http://localhost:8000/feed.

## Using the demo feed

- **Add videos:** put `.mp4` files in `samples/`, which is gitignored, and run `python -m brainrot.cli index samples`. Videos that are already classified and unchanged are skipped, so you only pay for new ones. Use `--force` to re-classify everything, and `BRAINROT_FEED_DIR` to use a different folder.
- **Captions:** if `samples/captions.json` exists (`{"file.mp4": {"creator": "...", "caption": "..."}}`), each video shows its creator's handle and original caption. Otherwise it shows the file name.
- **Scroll:** swipe, or use the arrow keys or `j`/`k`. Tap a video to pause it, and press `m` or the speaker button to unmute.
- **The score:** each video has a score circle on the right.
    - For brain rot, the ring fills up to the score as the video starts and keeps pulsing until you tap it.
    - For anything else, it just shows the score.
    - Tapping it opens the explanation: Gemini's reasoning, the tropes it found, redeeming qualities, the 7 criteria and the measured cuts per minute.
- **Policy toggle** (top bar):
    - **Label:** show every video with its score.
    - **Warn:** blur flagged videos behind a "Flagged as brain rot" screen with Skip and Watch anyway buttons.
    - **Hide:** remove flagged videos from the feed. The end card says how many were filtered out.
- **Add a video live:** press **+** or drop a video file onto the page. It uploads, Gemini classifies it ("Gemini is watching it…" with a timer), and it's added to the top of the feed with an "added" tag. From then on its verdict is stored like every other video's.
- **Brain damage meter:** the brain at the bottom goes from pink to grey as you watch brain rot. Scrolling onto a brain rot video costs `score × 0.05` points once. After that, every second you keep watching it costs more, and twice as much after 10 seconds. Watching anything else slowly heals you, and so does time away (2 points a minute). It goes through Fresh, Foggy, Cooked, Fried and Gone. Next to it are the counts for this session (videos scrolled and brain rot) and for all time.
- **Session receipt:** tap the meter, press `r`, or use the button on the end card to see what this session cost you: videos scrolled, the brain rot share, watch time, scene cuts sat through, things learned, the worst offender, and the start, peak and current damage, plus all-time totals. The session is saved in the browser, so a reload keeps it. It ends after 30 minutes without watching or when you press **End session**, and the next time you open the feed you get the last session's receipt.

## Technical details

### Local analysis (`brainrot/signals.py`)

This runs before Gemini, with OpenCV and ffmpeg, and costs nothing:

- **Pacing:** the video is sampled at 8 frames per second and each sample is shrunk to 160 px wide. A **scene cut** is counted when the colour histograms of two consecutive samples correlate below 0.55, or the mean pixel difference is above 40. From this it computes cuts per minute, average in-shot motion, and whether the video is vertical.
- **Audio:** ffprobe and ffmpeg check for a sound track and measure its average loudness.
- **Keyframes:** the first 180 seconds are split into N equal segments (`BRAINROT_MAX_FRAMES`). Each segment contributes its most-changing frame, so the whole video is covered and the frames land on cuts and action. Each frame is resized to at most 768 px and saved as a JPEG.

These numbers go into the prompt. Gemini only sees a handful of frames, so it can't count fast cuts itself; the measured cuts per minute tell it how hyper-edited the video really is.

### The Gemini call (`brainrot/classifier.py`, `brainrot/gemini_client.py`)

- **Frames mode** (`BRAINROT_MODE=frames`) sends the keyframes, each with its timestamp, plus a 32 kbps mono MP3 of the audio, all inline in a single request.
- **Native mode** (`BRAINROT_MODE=native`) uploads the whole video through the Gemini File API instead (trimmed to 180 seconds). Gemini samples it at `BRAINROT_NATIVE_FPS` frames per second, with sound. Public YouTube links are passed to Gemini by URL.
- **The prompt** (`brainrot/prompts/classify.txt`) defines brain rot, lists 7 criteria, explicitly lists what is *not* brain rot (fast but skilled edits, educational content, comedy with a real premise) to avoid false positives, and includes the measured pacing.
- **Structured output:** the call sets `response_mime_type="application/json"` and passes a JSON schema generated from the Pydantic model `GeminiVerdict`, so Gemini must return:

| Field | Meaning |
| --- | --- |
| `brain_rot_score` | 0 to 100. At or above `BRAINROT_THRESHOLD` (default 50), the video is brain rot |
| `criteria` | 0 to 10 each: overstimulation, low informational value, incoherence/absurdism, meme slang density, split-screen/filler gameplay, AI slop/TTS, engagement bait |
| `detected_tropes`, `redeeming_qualities` | Named evidence for and against |
| `content_summary`, `reasoning` | What the video shows and why it got that score |
| `confidence` | 0 to 1, lower when the evidence is thin |

- **Reliability:** temperature is 0 so scores are repeatable. Errors 429, 500, 502, 503 and 504 are retried with exponential backoff (2, 4, 8, 16 s). Running out of the daily quota fails immediately with a clear message, because retrying for seconds can't help.

### Storage (`brainrot/feed.py`)

- `samples/feed.json` maps each file name to its verdict, plus the file's size and modification time. If a video file changes, its verdict no longer matches, and the video is marked pending until it's indexed again.
- Writes are atomic (write to a temporary file, then rename), so the server never reads a half-written file. They also take a file lock and merge only their own entry, so an `index` run and live uploads can write at the same time without losing each other's work.
- Uploads get a cleaned file name: only safe characters, and `-2`, `-3` and so on if the name is taken. A failed upload deletes its file, so the feed never shows a video without a verdict.

### Web server (`brainrot/web/server.py`, FastAPI)

| Endpoint | What it does |
| --- | --- |
| `GET /feed` | The demo feed page |
| `GET /api/feed` | Stored verdicts and captions for every classified video, plus any still pending |
| `GET /media/{name}` | Streams a video from the feed folder, with Range support for seeking. Only files in the folder are served |
| `POST /api/feed/upload` | Saves an uploaded video, classifies it once, stores the verdict, and returns the new feed item |
| `GET /` and `POST /api/classify` | Single-video classifier page: upload a file or paste a URL |
| `POST /predict_clip`, `POST /predict` | Used by the browser extension: a recorded clip, or screenshots. Answers are cached by video ID |
| `GET /health`, `GET /api/config` | Model, threshold and whether an API key is set |

### Feed page (`brainrot/web/templates/feed.html`)

A single HTML file with no framework.

- **Playback:** CSS scroll snapping gives one video per screen. An `IntersectionObserver` plays the video that's at least 60% visible, pauses the others, and preloads the next one.
- **Animations:** the score ring is a CSS `conic-gradient` driven by a custom property that JavaScript counts up from 0. The pulse and the screen-edge flash are CSS animations, and they switch off for people whose system is set to reduce motion.
- **Uploads:** `XMLHttpRequest` reports upload progress, then the card switches to "Gemini is watching it…".

## Browser extension

The extension counts brain rot videos while you scroll Instagram or TikTok. It needs the server from the Quick start running at `localhost:8000`.

1. Go to `chrome://extensions` and turn on Developer mode.
2. Click Load unpacked and select the `extension-counter` folder.
3. Open instagram.com/reels or tiktok.com and play a video. A draggable counter appears in the corner.

By default it records the first 8 seconds of the video you're watching, sound included, straight from the page's video player, and sends that clip to `/predict_clip`. This adds no requests to Instagram or TikTok. If a video can't be recorded, it falls back to 8 screenshots sent to `/predict`. Each video counts once per day. More detail is in [`extension-counter/README.md`](extension-counter/README.md).

## Command line

```bash
python -m brainrot.cli classify video.mp4      # one or more files, folders or URLs; --json for raw output
python -m brainrot.cli index samples           # pre-classify a folder for the demo feed
python -m brainrot.cli eval labels.csv         # accuracy against a labelled CSV (source,label[,context])
python -m brainrot.cli serve                   # web server on port 8000
```

`classify`, `index` and `eval` also accept `--mode`, `--threshold`, `--max-frames`, `--max-seconds` and `--model`.

## Settings (`.env`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `GEMINI_API_KEY` | | Your Gemini API key |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Any Gemini model that accepts video |
| `BRAINROT_MODE` | `native` | `native` (whole video) or `frames` (keyframes + audio) |
| `BRAINROT_THRESHOLD` | `50` | Score at or above which a video is brain rot |
| `BRAINROT_MAX_FRAMES` | `16` | Keyframes sent in frames mode |
| `BRAINROT_MAX_SECONDS` | `180` | Only the first N seconds are analysed |
| `BRAINROT_NATIVE_FPS` | `2` | Frames per second Gemini samples in native mode |
| `BRAINROT_FEED_DIR` | `samples` | Folder the demo feed reads |

## Tests

```bash
python -m pytest
```

The tests create their own small videos with ffmpeg and use a fake Gemini backend, so they need no API key and make no network calls.

## Limits

- On the free Gemini tier, each model allows 20 requests per day, shared by `index` runs and live uploads. Enable billing for more.
- Frames mode sees a handful of frames rather than every frame; the measured pacing makes up for some of that, and native mode sends the whole video.
- A verdict reflects one model's judgement against a written rubric. Check it against videos labelled by people with `brainrot eval` before relying on it.
