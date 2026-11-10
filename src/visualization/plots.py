"""Waveform and spectrogram visualizations for validated WAV recordings."""

from __future__ import annotations

import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


_BACKGROUND = "#0e1117"
_FOREGROUND = "#fafafa"
_GRID = "#3b4252"


def _as_mono(audio: np.ndarray) -> np.ndarray:
    """Return finite one-dimensional samples for display without altering inference."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 2:
        samples = np.mean(samples, axis=1, dtype=np.float32)
    if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("Audio visualization requires non-empty finite mono or stereo samples.")
    return samples


def _dark_figure() -> tuple[Figure, plt.Axes]:
    """Create a consistently styled figure for the dark application theme."""
    figure, axis = plt.subplots(figsize=(10, 3.25), facecolor=_BACKGROUND)
    axis.set_facecolor(_BACKGROUND)
    axis.tick_params(colors=_FOREGROUND)
    for spine in axis.spines.values():
        spine.set_color(_GRID)
    return figure, axis


def plot_waveform(audio: np.ndarray, sample_rate: int) -> Figure:
    """Create a dark-theme waveform figure from source WAV samples."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    samples = _as_mono(audio)
    figure, axis = _dark_figure()
    librosa.display.waveshow(samples, sr=sample_rate, ax=axis, color="#4cc9f0", alpha=0.9)
    axis.set(title="Waveform", xlabel="Time (seconds)", ylabel="Amplitude")
    axis.title.set_color(_FOREGROUND)
    axis.xaxis.label.set_color(_FOREGROUND)
    axis.yaxis.label.set_color(_FOREGROUND)
    axis.grid(alpha=0.25, color=_GRID)
    figure.tight_layout()
    return figure


def plot_spectrogram(audio: np.ndarray, sample_rate: int) -> Figure:
    """Create a dark-theme log-power spectrogram figure from WAV samples."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    samples = _as_mono(audio)
    figure, axis = _dark_figure()
    spectrum = np.abs(librosa.stft(samples, n_fft=2_048, hop_length=512)) ** 2
    power_db = librosa.power_to_db(spectrum, ref=np.max)
    image = librosa.display.specshow(power_db, sr=sample_rate, hop_length=512, x_axis="time", y_axis="hz", ax=axis, cmap="magma")
    colorbar = figure.colorbar(image, ax=axis, format="%+2.0f dB")
    colorbar.ax.tick_params(colors=_FOREGROUND)
    colorbar.set_label("Power (dB)", color=_FOREGROUND)
    axis.set(title="Spectrogram", xlabel="Time (seconds)", ylabel="Frequency (Hz)")
    axis.title.set_color(_FOREGROUND)
    axis.xaxis.label.set_color(_FOREGROUND)
    axis.yaxis.label.set_color(_FOREGROUND)
    figure.tight_layout()
    return figure
