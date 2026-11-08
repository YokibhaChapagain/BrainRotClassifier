# Brain Rot Feed

Brain Rot Feed decides whether a short video is **brain rot**: low-value, overstimulating content made only to hold your attention. Think Skibidi edits, Subway Surfers split-screens, AI slop and text-to-speech Reddit stories.

Every video gets:

- a **score from 0 to 100** (50 or more counts as brain rot),
- **seven criteria scores**, such as overstimulation and engagement bait,
- a **written explanation** of why it got that score.

Then it shows you what that content does to you. The demo feed tracks a **brain damage meter** while you scroll. As the damage grows, the whole feed rots in front of you: it washes out to zombie green, drips slime and throbs with a heartbeat. At the end, you get a **receipt** for what your scrolling session cost you.

---

## Contents

1. [What's in the repo](#whats-in-the-repo)
2. [Quick start](#quick-start)
3. [The full workflow](#the-full-workflow)
4. [Using the feed](#using-the-feed)
5. [Brain damage, rot effects and the receipt](#brain-damage-rot-effects-and-the-receipt)
6. [Technical architecture](#technical-architecture)
7. [Browser extension](#browser-extension)
8. [Command line](#command-line)
9. [Settings](#settings)
10. [Tests](#tests)
11. [Limits](#limits)

---

## What's in the repo

The project has three parts:

| Part | Where | What it does |
| --- | --- | --- |
| **Classifier** | `brainrot/` | Measures a video locally, then asks Gemini to score it against a written rubric |
| **Demo feed** | `http://localhost:8000/feed` | A vertical, swipeable feed of videos that were classified ahead of time, with the brain damage system on top |
| **Chrome extension** | `extension-counter/` | Counts brain rot videos while you scroll Instagram or TikTok |

```
brainrot/
├── cli.py               command line: classify, index, eval, serve
├── config.py            settings from .env
├── signals.py           local video analysis (OpenCV + ffmpeg, no AI)
├── classifier.py        runs the analysis, builds the prompt and calls the model
├── gemini_client.py     talks to the Gemini API
├── schema.py            the shape of a verdict (Pydantic models)
├── prompts/classify.txt the rubric the model scores against
├── feed.py              stores verdicts in feed.json and orders the feed
└── web/
    ├── server.py        FastAPI server: pages, feed API, uploads, video streaming
    └── templates/
        ├── feed.html    the demo feed (one file, no framework)
        └── index.html   single-video classifier page
extension-counter/       the Chrome extension
tests/                   pytest suite (no API key needed)
samples/                 your videos + feed.json (gitignored)
```

---

## Quick start

You need **Python 3.10+** and **ffmpeg** (`brew install ffmpeg` on a Mac).

**1. Install**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

**2. Add your API key.** Open `.env` and set `GEMINI_API_KEY` to a key from https://aistudio.google.com/apikey.

**3. Add some videos.** Put `.mp4` files in a `samples/` folder. You can also add `samples/captions.json` to show each video's creator and caption (see [Using the feed](#using-the-feed)).

**4. Classify them once**

```bash
python -m brainrot.cli index samples
```

**5. Start the server**

```bash
python -m uvicorn brainrot.web.server:app --port 8000
```

**6. Open the feed** at http://localhost:8000/feed

---

## The full workflow

The key idea: **a platform doesn't classify videos while people watch.** It classifies each video **once, when it arrives**, saves the verdict, and reads that saved verdict every time someone scrolls past. That's why scrolling is instant even though classifying one video takes about 10 seconds.

```
 ① ARRIVE            ② ANALYSE (local)        ③ CLASSIFY (Gemini)       ④ STORE
 ─────────────       ──────────────────       ───────────────────       ─────────────
 + button  ─┐        cuts per minute          rubric + pacing           feed.json
 drop file ─┼──────► motion, loudness ──────► + video or keyframes ───► next to the
 `index`   ─┘        keyframes, audio         → JSON verdict            video file
                                                                             │
 ════════════════════════ once per video, ~10 s ══════════════════════════════╪═══
                                                                             │
 ⑤ SERVE              ⑥ ORDER                  ⑦ POLICY                 ⑧ WATCH
 ─────────────       ──────────────────       ───────────────────      ─────────────
 GET /api/feed ◄──── uploads first,  ◄─────── Label / Warn / Hide ◄─── score ring,
 reads saved         2 clean, then            decides what the         "why" panel,
 verdicts only       worst rot alternating    viewer sees              brain damage,
                                                                       rot effects,
 ═══════════════════════ every view, instant, no AI calls ═════════    receipt
```

**Step by step:**

1. **A video arrives.** It comes from the `index` command (a whole folder), the **+** button, or a file dropped onto the feed page.
2. **Local analysis** measures how the video is edited, with no AI involved: scene cuts per minute, motion, whether it has sound and how loud it is. It also picks the keyframes worth sending.
3. **Gemini classifies it.** It gets the video (or the keyframes and audio), the measured pacing and the written rubric. It returns a structured JSON verdict.
4. **The verdict is saved** to `feed.json` in the same folder as the video. If the video file changes later, its verdict no longer matches and the video goes back to "pending".
5. **The feed loads saved verdicts** from `/api/feed`. No model is called while you scroll.
6. **The feed is ordered** to lure you in. Uploads come first, then 2 clean videos, then the worst brain rot alternating with clean ones (see [Ordering](#ordering-brainrotfeedpy)).
7. **A policy decides what you see**:
    - **Label** every video with its score.
    - **Warn** puts flagged videos behind a blurred warning screen.
    - **Hide** removes them from the feed.
8. **You watch.** Each video shows its score. Brain rot drains your brain damage meter and the feed degrades around you. The receipt totals it all up.

---

## Using the feed

### Controls

| Action | How |
| --- | --- |
| Next / previous video | Swipe, scroll, arrow keys, or `j` / `k` |
| Pause / play | Tap the video |
| Sound on / off | Speaker button or `m` |
| See why a video got its score | Tap the score circle |
| Session receipt | Tap the brain meter, press `r`, or use the button on the end card |
| Rot effects on / off | TV button in the top bar, or `d` |
| Close a panel | `Esc` or tap outside it |
| Add a video | **+** button, or drop a file onto the page |

### What you see

- **Score circle** (right side):
    - For brain rot, it's red, counts up to the score and pulses until you tap it.
    - For anything else, it's green and still.
    - Tapping it opens the **"why" panel**: Gemini's reasoning, the tropes it found, redeeming qualities, the 7 criteria and the measured cuts per minute.
- **Screen-edge flash:** red when a brain rot video comes on screen, green otherwise.
- **Policy toggle** (top bar):
    - **Label:** shows everything with its score.
    - **Warn:** blurs flagged videos behind "Flagged as brain rot", with **Skip** and **Watch anyway** buttons.
    - **Hide:** removes flagged videos from the feed. The end card says how many were filtered out.
- **Adding a video live:** the file uploads, a card shows "Gemini is watching it…" with a timer, and the video joins the top of the feed with an **added** tag. From then on, its verdict is stored like any other.

### Captions

If `samples/captions.json` exists, each video shows its creator's handle and original caption. Otherwise it shows the file name.

```json
{ "clip.mp4": { "creator": "someone", "caption": "wait for the end 💀 #brainrot" } }
```

### Using a different folder

Set `BRAINROT_FEED_DIR` to any folder of videos, then run `index` on it. Videos that are already classified and unchanged are skipped, so you only pay for new ones. Use `--force` to re-classify everything.

---

## Brain damage, rot effects and the receipt

All of this runs in the browser and lives **in memory only**. Refreshing the page gives you a fresh brain.

### How damage builds

Damage goes from 0 to 100. Brain rot hurts you, and **watching anything else slowly heals you**: the feed creeps back to normal until the next brain rot sets it off again. Time away and paused videos don't heal you.

| What you do | Damage | Example: video scored 72 |
| --- | --- | --- |
| Scroll onto a brain rot video | `score × 0.15` | −10.8 |
| Keep watching it | `score ÷ 100 × 2.5` per second, going up by another 1× every 5 seconds with no cap | ~75 points after 16 s on one video |
| Let it loop back to the start | `score × 0.1 × loop number` (each loop costs more) | −7.2, then −14.4, then −21.6 |
| Scroll back to one you already saw ("relapse") | `score × 0.08` | −5.8 |
| A flagged video waiting behind the Warn screen | nothing, until you press **Watch anyway** | 0 |
| Watch a video that isn't brain rot | **heals 2 points a second** while it plays (shows "Recovering…") | 20 s of clean content takes back 40 |

In practice, about 16 seconds on a 72 takes you from Fresh to Fried.

### Brain levels

| Damage | Level | What changes |
| --- | --- | --- |
| 0–19 | **Fresh** | A pink brain. The effects below start fading in as soon as damage rises |
| 20–44 | **Foggy** | Slime sags from the top and starts creeping in from the sides |
| 45–69 | **Cooked** | Keyboard scrolling starts to lag, up to about 0.7 s |
| 70–89 | **Fried** | A slime pool rises along the bottom. The feed tilts, glitches every few seconds, and the title misspells itself ("brian rott fed") |
| 90–100 | **Zombie** | The feed sways, the score rail jitters, swipes stop snapping cleanly to a video, and the heart slows to a dead thud |

### Rot effects

These grow with the square root of your damage, so at 25 damage the feed already looks half rotted:

- **Zombie green:** colours drain and darken, and everything takes on a dull, sickly green cast.
- **Ghosted text:** captions, the score rail and the title blur and split into green and purple ghosts.
- **Old TV:** scanlines, flickering film grain and a dark vignette creep in.
- **Slime:** covers the whole phone screen. Green ooze sags from the top, drips stretch and fall all the way down, goo creeps in from both sides, and a pool rises along the bottom. It gets thicker with damage, and more of it appears at each level.
- **Heartbeat:** the brain icon thumps lub-dub, and the screen edges throb dark red in time. The heart races from about 55 to 125 bpm as damage climbs, then drops to about 25 bpm as a zombie.

Healing walks every effect back the same way it came. The receipt and the "why" panel never rot, so you can always read them. The effects switch off with the TV button or `d`, and the motion stops if your system is set to reduce motion.

### The meter

The pill at the bottom of the screen shows:

- the brain icon, coloured from pink to zombie green,
- your level and remaining brain (for example "Cooked · 41%"),
- videos scrolled and brain rot seen, for this session and for all time since the page loaded.

### The session receipt

A printed-style receipt of what this session cost you:

- videos scrolled, the brain rot count and its share
- time watching, and how much of it was brain rot
- scene cuts sat through, worked out from each video's cuts per minute
- things learned: redeeming qualities of non-brain-rot videos you watched for more than 3 seconds
- the **worst offender**: the video that cost you the most brain
- damage at the start, at its peak and now
- all-time totals
- a verdict line, such as *"No refunds. Your attention has already been spent."*

A session ends after **30 minutes without watching**, or when you press **End session**. Your damage carries into the next session. If you come back to the open tab after a long break, the last session's receipt pops up.

---

## Technical architecture

```
┌──────────────────────────────── Browser ─────────────────────────────────┐
│  feed.html                                       Chrome extension          │
│  ├─ feed + playback (IntersectionObserver)       (Instagram / TikTok)      │
│  ├─ brain damage engine (250 ms tick)                  │                   │
│  ├─ rot effects (CSS driven by --k, --beat)            │ 8 s clip or       │
│  └─ receipt                                            │ screenshots       │
└──────┬────────────────────┬──────────────────────────┬─────────────────────┘
       │ GET /api/feed      │ POST /api/feed/upload    │ POST /predict_clip
       │ GET /media/{name}  │                          │ POST /predict
┌──────▼────────────────────▼──────────────────────────▼─────────────────────┐
│  FastAPI server (web/server.py)                                            │
└──────┬────────────────────┬────────────────────────────────────────────────┘
       │ read               │ classify + write
┌──────▼──────────┐   ┌─────▼──────────────────────────────────────────────┐
│ feed.py         │   │ classifier.py                                      │
│ feed.json       │◄──│  1. signals.py ── OpenCV + ffmpeg: cuts, motion,   │
│ (verdicts),     │   │                   audio, keyframes                 │
│ ordering,       │   │  2. prompt     ── prompts/classify.txt + signals   │
│ file locking    │   │  3. model      ── gemini_client.py → JSON verdict  │
└─────────────────┘   └────────────────────────────────────────────────────┘
```

### 1. Local analysis (`brainrot/signals.py`)

This runs before any AI call, with OpenCV and ffmpeg, and costs nothing:

- **Pacing:** samples the video at 8 frames per second and shrinks each sample to 160 px wide. It counts a **scene cut** when two consecutive samples' colour histograms correlate below 0.55, or their mean pixel difference is above 40. From that it works out cuts per minute, average motion within shots, and whether the video is vertical.
- **Audio:** ffprobe and ffmpeg check for a sound track and measure its average loudness.
- **Keyframes:** splits the first 180 seconds into N equal segments (`BRAINROT_MAX_FRAMES`) and takes the most-changing frame from each one. That way the frames cover the whole video and land on cuts and action. Each frame is resized to at most 768 px and saved as a JPEG.

These numbers go into the prompt. In frames mode, the model only sees a handful of frames, so it can't count fast cuts itself. The measured cuts per minute tell it how hyper-edited the video really is.

### 2. Classification (`brainrot/classifier.py`, `brainrot/gemini_client.py`)

- **Native mode** (`BRAINROT_MODE=native`, the default) uploads the whole video, trimmed to 180 s, through the Gemini File API. Gemini samples it at `BRAINROT_NATIVE_FPS` frames per second, with sound. Public YouTube links are passed to Gemini by URL.
- **Frames mode** (`BRAINROT_MODE=frames`) sends the keyframes, each with its timestamp, plus a 32 kbps mono MP3 of the audio, all inline in a single request.
- **The rubric** (`brainrot/prompts/classify.txt`):
    - defines brain rot and its 7 criteria,
    - lists what is *not* brain rot (fast but skilled edits, educational content, comedy with a real premise) to avoid false positives,
    - lists what *is* brain rot even when it looks calm (meme "core" edits, slang punchlines, #brainrot-tagged clips) to avoid false negatives,
    - includes the measured pacing, plus the post's caption and hashtags when you pass them with `classify --context` or a `context` column in `eval`.
- **Structured output:** the request includes a JSON schema generated from the Pydantic model `GeminiVerdict`, so the model has to return:

| Field | Meaning |
| --- | --- |
| `brain_rot_score` | 0–100. At or above `BRAINROT_THRESHOLD` (default 50), the video is brain rot |
| `criteria` | 0–10 each: overstimulation, low informational value, incoherence/absurdism, meme slang density, split-screen/filler gameplay, AI slop/TTS, engagement bait |
| `detected_tropes`, `redeeming_qualities` | Named evidence for and against |
| `content_summary`, `reasoning` | What the video shows, and why it got that score |
| `confidence` | 0–1, lower when the evidence is thin |

- **Reliability:**
    - Temperature is 0, so scores are repeatable.
    - Errors 429, 500, 502, 503 and 504 are retried with exponential backoff (2, 4, 8, 16 s).
    - Running out of the daily quota fails right away with a clear message, because retrying won't help.

### 3. Storage (`brainrot/feed.py`)

- `samples/feed.json` maps each file name to its verdict, plus the file's size and modification time. A changed file no longer matches its verdict and shows up as pending until it's indexed again.
- **Atomic writes:** each write goes to a temporary file that's then renamed into place, so the server never reads a half-written file.
- **File lock:** writers take a lock and merge only their own entry. An `index` run and live uploads can write at the same time without losing each other's work.
- **Safe uploads:** uploads get a cleaned file name (safe characters only, with `-2`, `-3` and so on if the name is taken). A failed upload deletes its file, so the feed never shows a video without a verdict.

### 4. Ordering (`brainrot/feed.py`)

`arrange()` sets the order of the feed:

1. Uploads, newest first.
2. `LEAD_CLEAN` (2) non-brain-rot videos, so the feed starts out looking normal.
3. `TOP_SLOTS` (6) slots alternating brain rot and non brain rot. The brain rot slots take the highest scores first.
4. Everything else, in file-name order.

If one kind runs out, the other fills the remaining slots.

### 5. Web server (`brainrot/web/server.py`, FastAPI)

| Endpoint | What it does |
| --- | --- |
| `GET /feed` | The demo feed page |
| `GET /api/feed` | Saved verdicts and captions for every classified video, in feed order, plus any still pending |
| `GET /media/{name}` | Streams a video from the feed folder, with Range support for seeking. Only files in that folder are served |
| `POST /api/feed/upload` | Saves an uploaded video, classifies it once, saves the verdict and returns the new feed item |
| `GET /` and `POST /api/classify` | Single-video classifier page: upload a file or paste a URL |
| `POST /predict_clip`, `POST /predict` | Used by the extension: a recorded clip, or screenshots. Answers are cached by video ID |
| `GET /health`, `GET /api/config` | Model, threshold, and whether an API key is set |

### 6. Feed page (`brainrot/web/templates/feed.html`)

A single HTML file with plain JavaScript and CSS, and no framework.

- **Playback:** CSS scroll snapping shows one video per screen. An `IntersectionObserver` plays the video that's at least 60% visible, pauses the others and preloads the next one. The active video only pauses once another one takes over or it leaves the screen, so the rot sway can't stop it. The phone column clips its content, so the swaying feed and every effect stay inside the phone screen.
- **Score ring:** a CSS `conic-gradient` driven by a custom property that JavaScript counts up from 0.
- **Session state:** a plain JavaScript object kept in memory, holding the start time, damage, peak damage, and each video's watch time, damage dealt and loop count. Ended sessions are added to an all-time total.
- **Damage engine:** a timer runs every 250 ms. While a brain rot video is actually playing, visible and not paused, it adds the per-second damage and checks whether the video looped. That check looks for the playback position jumping backwards. While a non-brain-rot video plays, the same timer heals instead.
- **Rot effects:** JavaScript sets three things on the page:
    - `--k`, the square root of damage ÷ 100,
    - `--beat`, the heartbeat length,
    - `data-tier`, the level from 0 to 4.

  CSS does the rest:

| Layer (bottom to top) | Technique |
| --- | --- |
| Video feed | `filter: saturate() brightness() contrast()` scaled by `--k`, and tilt/sway transforms per level |
| Zombie tint | A full-screen layer with `mix-blend-mode: color` in dull green, opacity scaled by `--k` |
| Scanlines, grain, vignette | Stacked gradients plus an SVG `feTurbulence` noise texture, animated in steps |
| Heartbeat throb | An inset red `box-shadow` pulsing on a lub-dub keyframe timed by `--beat` |
| Slime | An SVG built by script at the phone screen's real pixel size (rebuilt on resize, so blobs stay round), with a "goo" filter (blur + alpha threshold) that merges the blobs. The top band and bottom pool scale with `--k`; drips, side goo and the pool show by level |
| Top bar, meter, receipt | Stay crisp on top |

- **Uploads:** `XMLHttpRequest` reports upload progress, then the card switches to "Gemini is watching it…".
- **Accessibility:** every animation stops when the system asks for reduced motion.

---

## Browser extension

The extension counts brain rot videos while you scroll Instagram or TikTok. It needs the server from the Quick start running at `localhost:8000`.

1. Go to `chrome://extensions` and turn on **Developer mode**.
2. Click **Load unpacked** and select the `extension-counter` folder.
3. Open instagram.com/reels or tiktok.com and play a video. A draggable counter appears in the corner.

It records the first 8 seconds of the video you're watching, sound included, straight from the page's video player, and sends that clip to `/predict_clip`. This adds no requests to Instagram or TikTok. If a video can't be recorded, the extension falls back to 8 screenshots sent to `/predict`. Each video counts once per day. More detail is in [`extension-counter/README.md`](extension-counter/README.md).

---

## Command line

```bash
python -m brainrot.cli classify video.mp4      # files, folders or URLs; --json for raw output
python -m brainrot.cli index samples           # pre-classify a folder for the feed (--force to redo all)
python -m brainrot.cli eval labels.csv         # accuracy against a labelled CSV (source,label[,context])
python -m brainrot.cli serve                   # web server on port 8000
```

`classify`, `index` and `eval` also accept `--mode`, `--threshold`, `--max-frames`, `--max-seconds`, `--fps` and `--model`. `classify` also takes `--context` for the post's caption and hashtags.

---

## Settings

Set these in `.env`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `BRAINROT_BACKEND` | `gemini` | Model provider |
| `GEMINI_API_KEY` | | Your Gemini API key |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Any Gemini model that accepts video |
| `BRAINROT_MODE` | `native` | `native` (whole video) or `frames` (keyframes + audio) |
| `BRAINROT_THRESHOLD` | `50` | Score at or above which a video counts as brain rot |
| `BRAINROT_MAX_FRAMES` | `16` | Keyframes sent in frames mode |
| `BRAINROT_MAX_SECONDS` | `180` | Only the first N seconds are analysed |
| `BRAINROT_NATIVE_FPS` | `2` | Frames per second Gemini samples in native mode |
| `BRAINROT_FEED_DIR` | `samples` | Folder the demo feed reads |

The feed's behaviour is tuned with constants:

- **Ordering:** `LEAD_CLEAN` and `TOP_SLOTS` in `brainrot/feed.py`.
- **Damage rates, levels and session timeout:** near the "scroll session" section of `feed.html`.

---

## Tests

```bash
python -m pytest
```

The tests make their own small videos with ffmpeg and use a fake model backend, so they need no API key and make no network calls. They cover the local analysis, the classifier, storage and file locking, feed ordering, uploads and the web endpoints.

---

## Limits

- **Free-tier quota:** on Gemini's free tier, each model allows 20 requests a day, shared by `index` runs and live uploads. Enable billing for more.
- **Frames mode:** it sees a handful of frames rather than every frame. The measured pacing makes up for some of that, and native mode sends the whole video.
- **One model's judgement:** a verdict is one model scoring against a written rubric. Before relying on it, check it against videos people have labelled, using `brainrot eval`.
- **Not saved:** brain damage and session stats live in one browser tab's memory, and a refresh resets them.
