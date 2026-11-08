"""CLIP concept bank: plain-English descriptions we compare each frame against.

These give the student human-readable content features ("looks like Minecraft gameplay") and power the
explanations. Edit freely during the hackathon: add concepts you keep seeing, and remove ones that never fire.
"""

# (short name shown to users, CLIP prompt)
CONCEPTS: list[tuple[str, str]] = [
    ("Minecraft gameplay", "a screenshot of Minecraft parkour gameplay"),
    ("Subway Surfers gameplay", "a screenshot of the mobile game Subway Surfers"),
    ("mobile game footage", "a screen recording of a colorful mobile video game"),
    ("slime / satisfying clip", "an oddly satisfying video of slime, soap cutting or kinetic sand"),
    ("cartoon clip", "a frame from a cartoon or animated show"),
    ("AI-generated character", "a surreal AI-generated 3D character with an animal body"),
    ("meme compilation", "a meme with bold impact font text over a random image"),
    ("text-to-speech story", "a reddit post screenshot over background gameplay"),
    ("reaction face", "a close-up of a person making an exaggerated shocked face"),
    ("someone cooking", "a person cooking food in a kitchen"),
    ("tutorial / explainer", "a person explaining something with a whiteboard, slides or diagrams"),
    ("talking to camera", "a person calmly talking to the camera in a room"),
    ("music performance", "a musician playing an instrument or singing"),
    ("sports / workout", "a person playing sports or exercising"),
    ("nature / travel", "a scenic outdoor landscape or travel footage"),
    ("dance trend", "a person dancing to music in their room"),
]
CONCEPT_NAMES = [c[0] for c in CONCEPTS]
CONCEPT_PROMPTS = [c[1] for c in CONCEPTS]
