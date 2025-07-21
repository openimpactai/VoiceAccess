"""
VoiceAccess: Automatic Speech Recognition for Low-Resource Languages

An open-source toolkit for bringing ASR capabilities to endangered and low-resource languages
through transfer learning and data augmentation techniques.
"""

__version__ = "0.1.0"
__author__ = "OpenImpactAI"
__license__ = "MIT"

from .core.asr_engine import ASREngine
from .core.config import Config
from .models.base_model import BaseASRModel

__all__ = ["ASREngine", "Config", "BaseASRModel"]