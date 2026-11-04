"""Waveform and spectrogram visualization interface."""

from typing import Any


def plot_waveform(audio: Any, sample_rate: int) -> Any:
    """Create a waveform plot when visualization is implemented."""
    raise NotImplementedError("Waveform plotting is not implemented in Phase 1.")


def plot_spectrogram(audio: Any, sample_rate: int) -> Any:
    """Create a spectrogram plot when visualization is implemented."""
    raise NotImplementedError("Spectrogram plotting is not implemented in Phase 1.")
