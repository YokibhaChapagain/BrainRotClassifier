"""Feature extractors. Each returns a dict of named scalars and/or an embedding vector.

Heavy models (sentence-transformers, CLIP, Whisper) are loaded lazily and cached, so the API
and tests can import this package without torch installed.
"""
