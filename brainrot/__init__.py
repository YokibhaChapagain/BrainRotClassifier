"""Brain rot video classifier powered by Gemini."""

from .classifier import BrainRotClassifier
from .schema import ClassificationResult

__all__ = ["BrainRotClassifier", "ClassificationResult"]
__version__ = "0.1.0"
