"""Tests for audio visualizations used by the Streamlit interface."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.visualization.plots import plot_spectrogram, plot_waveform


def test_waveform_and_spectrogram_render_valid_audio() -> None:
    """Valid mono and stereo arrays produce labelled Matplotlib figures."""
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    mono = np.sin(2 * np.pi * 220 * time).astype(np.float32)
    stereo = np.column_stack((mono, -mono))

    waveform = plot_waveform(stereo, sample_rate)
    spectrogram = plot_spectrogram(mono, sample_rate)

    assert waveform.axes[0].get_title() == "Waveform"
    assert waveform.axes[0].get_xlabel() == "Time (seconds)"
    assert spectrogram.axes[0].get_title() == "Spectrogram"
    assert spectrogram.axes[0].get_ylabel() == "Frequency (Hz)"
    plt.close(waveform)
    plt.close(spectrogram)


def test_visualizations_reject_invalid_audio() -> None:
    """Visualizations report invalid arrays rather than rendering misleading output."""
    with pytest.raises(ValueError, match="non-empty finite"):
        plot_waveform(np.array([], dtype=np.float32), 16_000)
    with pytest.raises(ValueError, match="sample_rate"):
        plot_spectrogram(np.ones(100, dtype=np.float32), 0)
