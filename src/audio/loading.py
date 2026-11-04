"""WAV input validation and audio-loading interface.

Implementation will use Librosa and return audio samples with the configured rate.
"""

from pathlib import Path


def validate_wav_path(audio_path: Path) -> None:
    """Validate that an existing input path has a WAV extension."""
    if audio_path.suffix.lower() != ".wav":
        raise ValueError("Only WAV audio files are supported.")
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file was not found: {audio_path}")
