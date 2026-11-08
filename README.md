# Brain Rot Detector

Classifies a video as **brain rot** or **not brain rot** using Gemini, with a 0–100 score, per-criterion
breakdown, detected tropes and reasoning. Works on local files, folders, YouTube links, and (via yt-dlp)
TikTok / Instagram / other links. Comes with a CLI, a web UI and an evaluation command.

Architecture is adapted from [byjlw/video-analyzer](https://github.com/byjlw/video-analyzer); see
[How it compares](#how-it-compares-to-byjlwvideo-analyzer).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[urls,dev]'          # needs ffmpeg on PATH for audio + trimming
cp .env.example .env                  # then put your key in .env
```

Get an API key at https://aistudio.google.com/apikey and set `GEMINI_API_KEY`.

```bash
# one video
brainrot classify path/to/video.mp4

# a YouTube short (Gemini watches it directly by URL)
brainrot classify "https://www.youtube.com/shorts/XXXX"

# a whole folder, saving results
brainrot classify videos/ -o results.jsonl

# web UI -> http://127.0.0.1:8000
brainrot serve
```

Sample output:

```
BRAIN ROT  score 84/100 (threshold 50, confidence 88%)  [native, gemini-3.8-flash, 23.4s]

Summary: A TTS voice reads an AITA Reddit story over Subway Surfers gameplay ...
Criteria (0-10):
  overstimulation                    ################     8
  low_informational_value            ##################   9
  split_screen_or_filler_gameplay    #################### 10
  ...
Tropes: Subway Surfers split-screen; TTS Reddit story; Word-by-word captions
Signals: 61s, 52.3 cuts/min, motion 18.2, vertical, volume -12.4 dB
```

## Measuring accuracy on your own data

Make a CSV with `source,label[,context]` (label = `brain_rot` / `not_brain_rot`). Paths are relative to the
CSV; see `samples_labels.example.csv`.

```bash
brainrot eval my_labels.csv -o eval_results.json
#  OK  score= 84 pred=brain_rot      truth=brain_rot      samples/subway.mp4
#  XX  score= 55 pred=brain_rot      truth=not_brain_rot  samples/mrbeast.mp4
#  20 videos (0 errors) | accuracy 90.0% | precision 88.9% | recall 88.9% | F1 0.89
```

Use the scores in `eval_results.json` to pick the best `--threshold` for your definition of brain rot.

## How it works

```
video ──► local analysis (OpenCV + ffmpeg) ──► objective signals: cuts/min, motion, loudness, aspect
   │                                                         │
   ├─ native mode: upload full video (frames+audio) ─────────┤
   │   via Gemini File API, sampled at 2 fps                 ├──► Gemini + rubric prompt ──► JSON verdict
   └─ frames mode: N keyframes + mono audio, inline ─────────┘     (response_json_schema)     │
                                                                                              ▼
                                               score >= threshold ? brain_rot : not_brain_rot
```

1. **Local signals** (`brainrot/signals.py`): every video is scanned at 8 fps with OpenCV for hard
   scene cuts (HSV histogram correlation + pixel difference), in-shot motion, and ffmpeg loudness. Gemini
   only samples 1–2 fps, so it can't count fast cuts itself. These numbers are passed to it as evidence.
2. **Gemini sees the content**:
   - `native` (default): the whole video including audio is uploaded through the File API. Best accuracy.
     Videos longer than `BRAINROT_MAX_SECONDS` (default 180s) are trimmed first to bound cost.
   - `frames`: picks `BRAINROT_MAX_FRAMES` keyframes (the most changed frame in each equal time segment,
     a coverage-safe version of the reference repo's difference-based selection) plus a 32 kbps mono audio
     track, sent inline. Cheaper, and nothing is stored on Google's side.
   - YouTube URLs are passed straight to Gemini (no download, but no local signals). Use `--download`
     to fetch with yt-dlp instead.
3. **Rubric prompt** (`brainrot/prompts/classify.txt`) defines brain rot through 7 criteria:
   overstimulation, low informational value, incoherence/absurdism, meme-slang density, split-screen
   filler gameplay, AI slop/TTS, and engagement bait. It also lists explicit **non-examples** (fast but
   skilled edits, meme-flavoured education, comedy with a premise) to keep false positives down. Edit this
   file to change the definition.
4. **Structured output**: Gemini must return JSON matching `GeminiVerdict` (`brainrot/schema.py`), so
   results are always parseable. The final label is applied in code from the score and your threshold.

## Configuration

Env vars (or `.env`), each overridable per command with a flag:

| Variable | Default | Flag | Meaning |
|---|---|---|---|
| `GEMINI_API_KEY` | – | | required |
| `GEMINI_MODEL` | `gemini-3.8-flash` | `--model` | any video-capable Gemini model; use a Pro model for harder cases |
| `BRAINROT_MODE` | `native` | `--mode` | `native` or `frames` |
| `BRAINROT_THRESHOLD` | `50` | `--threshold` | score ≥ this ⇒ brain rot |
| `BRAINROT_NATIVE_FPS` | `2` | `--fps` | frames/sec Gemini samples in native mode (0–24] |
| `BRAINROT_MAX_FRAMES` | `16` | `--max-frames` | keyframes in frames mode |
| `BRAINROT_MAX_SECONDS` | `180` | `--max-seconds` | analyse only the first N seconds (0 = all) |

Extra context such as the caption or hashtags can help: `brainrot classify v.mp4 --context "#skibidi #fyp"`.

## Python API

```python
from brainrot import BrainRotClassifier
from brainrot.config import Settings

clf = BrainRotClassifier(Settings())          # reads GEMINI_API_KEY
r = clf.classify("video.mp4")
print(r.label, r.brain_rot_score, r.verdict.detected_tropes)
print(r.model_dump_json(indent=2))
```

## Tests

```bash
pytest -q
```

The 23 tests need no API key. They generate synthetic videos with ffmpeg (a calm single shot vs. a
vertical video that hard-cuts every 0.25s), check the cut detector tells them apart, and run the full
pipeline, CLI and web API against a fake Gemini backend. The real `GeminiBackend` is checked against
the real SDK types with the network call mocked, including retry on 429/5xx.

## How it compares to byjlw/video-analyzer

| | video-analyzer | this project |
|---|---|---|
| Goal | free-text description of a video | binary classification + score |
| LLM | Ollama / OpenAI-compatible, image-only | Gemini, natively watches video **and** audio |
| Calls per video | 1 per frame + 1 reconstruction | **1** |
| Audio | local Whisper transcription | Gemini hears the audio directly (no Whisper) |
| Frames | top-N by frame difference (can cluster) | per-segment max difference (full coverage), or none in native mode |
| Output | prose | schema-enforced JSON |
| Extra | | pacing signals, threshold tuning, eval command, web UI |

The reference's sequential frame-by-frame chaining exists because local image models can't take a video.
Gemini can, so a single multimodal call is cheaper, faster and sees motion and audio the frames would miss.

## Project layout

```
brainrot/
  classifier.py      orchestration: file / URL / YouTube, native vs frames
  gemini_client.py   google-genai wrapper: File API upload, structured output, retries
  signals.py         OpenCV/ffmpeg: cut detection, motion, loudness, keyframes, audio extraction
  schema.py          pydantic models (Gemini response schema + final result)
  prompts/classify.txt   the rubric. Edit this to change what counts as brain rot.
  cli.py             brainrot classify | eval | serve
  web/               FastAPI server + single-page UI
tests/               pytest suite (no network needed)
```
