"""Shared audio preprocessing interface for training and inference."""

from typing import Any

from src.config import AudioPreprocessingConfig


def preprocess_audio(audio: Any, sample_rate: int, config: AudioPreprocessingConfig) -> Any:
    """Preprocess audio according to the shared configuration in future work."""
    raise NotImplementedError("Audio preprocessing is not implemented in Phase 1.")
