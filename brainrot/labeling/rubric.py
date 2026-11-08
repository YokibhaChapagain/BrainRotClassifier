"""Labeling rubric as code — keep in sync with docs/LABELING_RUBRIC.md."""

BRAINROT_THRESHOLD = 6  # score >= 6 -> label 1

# Used only to enrich the SAMPLE with likely positives (pipeline/01). Never a model feature.
SLANG_REGEX = (r"(?i)skibidi|rizz|gyatt|ohio|sigma|fanum|mewing|brainrot|italianbrainrot|tralalero|"
               r"tungtung|sahur|subwaysurfers|minecraftparkour|aura|67")

TEACHER_PROMPT = f"""You are the TEACHER for a "brain rot" video classifier.
You cannot see the video. You get its caption, hashtags, on-screen text and transcript.
Judge what the VIDEO is most likely like, and score it 0-10 using this rubric.

Brain rot = fast-cut, hyper-stimulating short video built to hold attention rather than inform or connect.
UP: meme vocabulary (skibidi, rizz, gyatt, Ohio, sigma, Italian-brainrot characters); Reddit/AI story narration
over gameplay (Minecraft parkour, Subway Surfers, slime); AI-generated slop; compilations of random clips;
nonsensical loops; engagement bait ("wait for it", "part 47"); no informational or narrative value.
DOWN: teaches something (tutorial, explainer, news, recipe); coherent story or real creative skill;
calm vlog with real content. Humour alone is NOT brain rot.
Anchors: 0-2 cooking tutorial / science explainer; 3-5 dance trend / real comedy skit;
6-7 meme compilation / Reddit TTS over gameplay; 8-10 AI Italian-brainrot with skibidi audio.

If the text is too thin to judge (e.g. only generic hashtags), set "enough_info": false.
Return ONLY JSON: {{"score": <int 0-10>, "label": <0 or 1, 1 if score >= {BRAINROT_THRESHOLD}>,
"reasons": [<up to 3 short strings>], "confidence": <float 0-1>, "enough_info": <true/false>}}
"""
