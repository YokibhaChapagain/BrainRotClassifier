# Brain Rot Labeling Rubric (v0.1 — agree on this at hour 0)

Score each reel **0–10**. `score >= 6` → `brainrot = 1`.

Used by the **teacher** (Gemini), which applies it from the video's text only (caption, hashtags, on-screen text,
transcript), and by **humans** labelling the test set, who apply it while watching the video. The code version
of this rubric is `brainrot/labeling/rubric.py`; keep them in sync.

## Signals that push the score UP
| Signal | Examples | Weight |
|---|---|---|
| Gen-Alpha slang / meme vocabulary | skibidi, rizz, gyatt, Ohio, sigma, fanum tax, mewing, "6-7", Italian-brainrot characters (tralalero, tung tung sahur) | high |
| Split-screen attention bait | Subway Surfers / Minecraft parkour / slime under unrelated narration | high |
| AI slop | AI-generated video/voice with no informational content (`is_ai_generated`) | high |
| TTS / robotic narration of Reddit stories | "AITA…" over gameplay | medium |
| Hyper-fast cuts, flashing, loud meme SFX, bass-boosted audio | > 2 cuts/sec, vine boom | medium |
| Zero informational or narrative value | random repetition, nonsensical loops | medium |
| Engagement bait | "wait for it", "part 47", fake cliffhangers | low |

## Signals that push the score DOWN
- Teaches something (tutorial, explainer, news, recipe, how-to)
- Coherent narrative or genuine creative skill (music performance, art process, sport)
- Personal vlog with real content, calm pacing

## Anchors
- **0–2**: cooking tutorial, science explainer, news clip
- **3–5**: generic dance trend, comedy skit with a real joke, vlog
- **6–7**: meme compilation, Reddit TTS over gameplay
- **8–10**: AI-generated Italian-brainrot character with skibidi audio and 3 cuts/sec

Edge cases: humour ≠ brainrot. Judge the *content*, not the creator. When unsure, score 5 and add a note.
