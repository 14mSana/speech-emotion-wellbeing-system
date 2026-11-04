"""WAV input validation and loading utilities."""

from pathlib import Path

import numpy as np
import soundfile as sf


def validate_wav_path(audio_path: Path) -> Path:
    """Validate and return an existing WAV file path.

    Args:
        audio_path: Path to an input WAV file.

    Raises:
        TypeError: If ``audio_path`` is not path-like.
        ValueError: If the path does not use the WAV extension.
        FileNotFoundError: If the file does not exist.
    """
    if not isinstance(audio_path, Path):
        raise TypeError("audio_path must be a pathlib.Path instance.")
    if audio_path.suffix.lower() != ".wav":
        raise ValueError("Only WAV audio files are supported.")
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file was not found: {audio_path}")
    return audio_path


def load_wav(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load a non-empty, finite WAV recording without resampling it.

    The returned samples use SoundFile's ``(frames, channels)`` layout for
    multi-channel files and ``float32`` values. Resampling and mono conversion
    belong to :func:`src.audio.preprocessing.preprocess_audio`.

    Args:
        audio_path: Existing WAV file to load.

    Returns:
        Audio samples and their source sample rate.

    Raises:
        ValueError: If the WAV is corrupted, empty, or contains non-finite data.
    """
    validated_path = validate_wav_path(audio_path)
    try:
        audio, sample_rate = sf.read(validated_path, dtype="float32", always_2d=False)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Could not read WAV audio file: {validated_path}") from error

    samples = np.asarray(audio, dtype=np.float32)
    if samples.size == 0:
        raise ValueError(f"Audio file is empty: {validated_path}")
    if sample_rate <= 0:
        raise ValueError(f"Audio file has an invalid sample rate: {validated_path}")
    if not np.isfinite(samples).all():
        raise ValueError(f"Audio file contains NaN or infinite samples: {validated_path}")
    return samples, int(sample_rate)
