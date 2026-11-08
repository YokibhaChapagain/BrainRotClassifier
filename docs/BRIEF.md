# BrainRot Meter — project brief

**The problem.** Short-form feeds reward "brain rot": fast-cut, hyper-stimulating videos built to hold attention
rather than inform or connect. Because it performs well on engagement, algorithms push more of it, and nobody
(users, parents, researchers) can measure how much they're getting. Captions don't help: they're often missing,
generic (`#fyp #viral`) or misleading.

**What we built.** A system that scores any short video for brain rot **from the pixels alone** and explains why:

> **0.87** · looks like Minecraft gameplay + split screen · 3× the typical cut rate

It runs live as a **Chrome extension** on TikTok and Instagram Reels (a meter + 🚩 on each reel), and a
**dashboard** shows your personal exposure over time: how much of your feed and your day was brain rot,
when it peaks, and how brain rot content performs on engagement.

**How it learns: teacher and student.**
- **Teacher = Gemini.** For ~3,000 TikToks from the Shofo dataset, Gemini reads only the *text* (caption,
  hashtags, on-screen text and transcript, with Whisper filling in missing transcripts) and scores each video
  0–10 against our written rubric. It never sees the video.
- **Student = our pixels-only classifier** (logistic regression). It never reads text. It learns to reproduce
  the teacher's labels from what's on screen:
  - **pace**: cut rate, motion, split-screen layout, colour intensity, flicker (OpenCV)
  - **content**: CLIP compares frames against plain-English concepts like "Minecraft gameplay",
    "cartoon clip" or "someone cooking"

  At demo time only the student runs: fast, cheap, and it works on videos with no caption at all.

This is called **cross-modal distillation**: the label comes from one kind of information (text), and the
model has to find it in a completely different one (pixels). If the student succeeds, brain rot is
*visually* recognizable, and it can't be fooled by misleading hashtags.

**How we prove it.**
- **Ablation**: motion-only vs CLIP-only vs motion + CLIP shows what makes brain rot visually recognizable.
- **400 human-labelled videos**: humans watch the video. We measure the student vs humans, the teacher vs
  humans, and the student on videos with no useful caption, where a text-based classifier has nothing.
- Splits by creator, calibrated probabilities, and the model is scored on exactly the 8 frames the extension sees.

**Built with.**
- **Databricks**: Delta tables in a bronze/silver/gold pipeline, Unity Catalog Volumes, MLflow
- **Gemini API**: the teacher, plus the playful one-line explanations in the overlay
- **Tiger Data**: hypertables and continuous aggregates powering the exposure dashboard
- Hugging Face (dataset), OpenCV, CLIP via sentence-transformers, Whisper, scikit-learn, FastAPI, Chrome MV3

**Tracks:** Best AI/ML · Best Use of Gemini API · Best Use of Tiger Data
