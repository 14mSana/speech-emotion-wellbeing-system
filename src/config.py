"""Shared, stable configuration contracts for training and inference."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

EMOTION_LABELS: tuple[str, ...] = ("neutral", "calm", "happy", "sad", "angry")
N_MFCC = 13
FEATURE_NAMES: tuple[str, ...] = (
    *(f"mfcc_{index:02d}_mean" for index in range(1, N_MFCC + 1)),
    *(f"mfcc_{index:02d}_std" for index in range(1, N_MFCC + 1)),
    "pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean", "zcr_std",
)


@dataclass(frozen=True)
class AudioPreprocessingConfig:
    """Parameters that must be identical during training and inference."""

    sample_rate: int = 22_050
    mono: bool = True
    trim_silence: bool = True
    trim_top_db: float = 30.0


if len(FEATURE_NAMES) != 32:
    raise RuntimeError("The baseline feature schema must contain exactly 32 features.")
