"""Prepare the approved five-class RAVDESS audio-speech dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from src.audio.features import extract_features
from src.audio.loading import load_wav
from src.config import AudioPreprocessingConfig, FEATURE_NAMES, PROCESSED_DATA_DIR, RAW_DATA_DIR


RAVDESS_DATASET_DIRNAME = "Audio_Speech_Actors_01-24"
RAVDESS_EMOTION_CODES: dict[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
}
TRAIN_ACTORS: tuple[str, ...] = tuple(f"{actor:02d}" for actor in range(1, 17))
VALIDATION_ACTORS: tuple[str, ...] = tuple(f"{actor:02d}" for actor in range(17, 21))
TEST_ACTORS: tuple[str, ...] = tuple(f"{actor:02d}" for actor in range(21, 25))
SPLIT_ACTORS: dict[str, tuple[str, ...]] = {
    "train": TRAIN_ACTORS,
    "validation": VALIDATION_ACTORS,
    "test": TEST_ACTORS,
}
EXPECTED_SPLIT_COUNTS: dict[str, int] = {"train": 576, "validation": 144, "test": 144}

_FILENAME_PATTERN = re.compile(
    r"^(?P<modality>\d{2})-(?P<vocal_channel>\d{2})-"
    r"(?P<emotion_code>\d{2})-(?P<intensity>\d{2})-"
    r"(?P<statement>\d{2})-(?P<repetition>\d{2})-(?P<actor_id>\d{2})$"
)


@dataclass(frozen=True)
class RavdessRecord:
    """Metadata parsed from one approved RAVDESS WAV recording."""

    relative_path: str
    filename: str
    modality: str
    vocal_channel: str
    emotion_code: str
    emotion_label: str
    intensity: str
    statement: str
    repetition: str
    actor_id: str


@dataclass(frozen=True)
class SkippedFile:
    """A discovered WAV that was excluded with a clear reason."""

    relative_path: str
    reason: str


@dataclass(frozen=True)
class ManifestResult:
    """Selected records and exclusions produced during discovery."""

    records: tuple[RavdessRecord, ...]
    skipped: tuple[SkippedFile, ...]


@dataclass(frozen=True)
class FeaturePreparationResult:
    """Paths and counts emitted while building the feature dataset."""

    feature_path: Path
    failed_path: Path
    metadata_path: Path
    extracted_count: int
    failed_count: int
    split_counts: dict[str, int]


def parse_ravdess_filename(filename: str) -> dict[str, str]:
    """Parse a RAVDESS seven-part WAV filename or raise ``ValueError``."""
    path = Path(filename)
    if path.suffix.lower() != ".wav":
        raise ValueError(f"RAVDESS recording must use a .wav extension: {filename}")
    match = _FILENAME_PATTERN.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Malformed RAVDESS filename: {filename}")
    return match.groupdict()


def discover_ravdess_records(dataset_root: Path) -> ManifestResult:
    """Recursively discover approved RAVDESS audio-speech recordings.

    Every WAV outside the modality/channel/five-label schema is retained in the
    skip report rather than silently ignored.
    """
    if not isinstance(dataset_root, Path):
        raise TypeError("dataset_root must be a pathlib.Path instance.")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"RAVDESS dataset directory was not found: {dataset_root}")

    records: list[RavdessRecord] = []
    skipped: list[SkippedFile] = []
    for audio_path in sorted(dataset_root.rglob("*.wav")):
        relative_path = audio_path.relative_to(dataset_root)
        try:
            parsed = parse_ravdess_filename(audio_path.name)
        except ValueError as error:
            skipped.append(SkippedFile(str(relative_path), str(error)))
            continue

        if parsed["modality"] != "03":
            skipped.append(SkippedFile(str(relative_path), "Excluded modality; expected 03 (audio-only)."))
            continue
        if parsed["vocal_channel"] != "01":
            skipped.append(SkippedFile(str(relative_path), "Excluded vocal channel; expected 01 (speech)."))
            continue
        emotion_label = RAVDESS_EMOTION_CODES.get(parsed["emotion_code"])
        if emotion_label is None:
            skipped.append(
                SkippedFile(
                    str(relative_path),
                    f"Excluded emotion code {parsed['emotion_code']}; only 01 through 05 are supported.",
                )
            )
            continue

        expected_folder = f"Actor_{parsed['actor_id']}"
        if audio_path.parent.name != expected_folder:
            skipped.append(
                SkippedFile(
                    str(relative_path),
                    f"Actor folder mismatch; expected {expected_folder} from filename.",
                )
            )
            continue
        records.append(
            RavdessRecord(
                relative_path=str(relative_path),
                filename=audio_path.name,
                emotion_label=emotion_label,
                **parsed,
            )
        )
    return ManifestResult(tuple(records), tuple(skipped))


def write_manifest(result: ManifestResult, output_dir: Path = PROCESSED_DATA_DIR) -> tuple[Path, Path]:
    """Write selected manifest and skipped-file report as CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "ravdess_manifest.csv"
    skipped_path = output_dir / "ravdess_skipped_files.csv"
    pd.DataFrame([asdict(record) for record in result.records]).to_csv(manifest_path, index=False)
    pd.DataFrame([asdict(record) for record in result.skipped], columns=("relative_path", "reason")).to_csv(
        skipped_path, index=False
    )
    return manifest_path, skipped_path


def split_for_actor(actor_id: str) -> str:
    """Return the fixed speaker-independent partition for one actor ID."""
    for split_name, actor_ids in SPLIT_ACTORS.items():
        if actor_id in actor_ids:
            return split_name
    raise ValueError(f"Actor {actor_id} is not assigned to a dataset split.")


def add_speaker_split(records: Iterable[RavdessRecord]) -> pd.DataFrame:
    """Create manifest rows with fixed actor-isolated split assignments."""
    rows = [{**asdict(record), "split": split_for_actor(record.actor_id)} for record in records]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No approved RAVDESS recordings were discovered.")
    _validate_split_isolation(frame)
    return frame


def _validate_split_isolation(frame: pd.DataFrame) -> None:
    actor_splits = frame.groupby("actor_id")["split"].nunique()
    overlapping_actors = actor_splits[actor_splits != 1]
    if not overlapping_actors.empty:
        raise ValueError(f"Actors appear in multiple splits: {list(overlapping_actors.index)}")


def build_feature_dataset(
    dataset_root: Path,
    records: Iterable[RavdessRecord],
    output_dir: Path = PROCESSED_DATA_DIR,
    config: AudioPreprocessingConfig | None = None,
    workers: int = 4,
) -> FeaturePreparationResult:
    """Extract the canonical feature vectors and write local processed CSV files."""
    if not isinstance(dataset_root, Path):
        raise TypeError("dataset_root must be a pathlib.Path instance.")
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer.")
    preprocessing_config = config or AudioPreprocessingConfig()
    manifest = add_speaker_split(records)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    manifest_rows = manifest.to_dict(orient="records")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        extracted_rows = executor.map(
            _extract_manifest_row,
            ((dataset_root, row, preprocessing_config) for row in manifest_rows),
        )
        for manifest_row, features, failure in extracted_rows:
            if failure is not None:
                failures.append(failure)
                continue
            if features is None:
                raise RuntimeError("Feature extraction did not return features or a failure.")
            rows.append({**manifest_row, **dict(zip(FEATURE_NAMES, features.tolist(), strict=True))})

    feature_frame = pd.DataFrame(rows, columns=[*manifest.columns, *FEATURE_NAMES])
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_path = output_dir / "ravdess_feature_failures.csv"
    pd.DataFrame(failures, columns=("relative_path", "reason")).to_csv(failed_path, index=False)
    _validate_feature_frame(feature_frame)
    split_counts = {split: int((feature_frame["split"] == split).sum()) for split in SPLIT_ACTORS}
    if failures:
        raise RuntimeError(
            f"Feature extraction failed for {len(failures)} file(s); no incomplete dataset was accepted. "
            "See ravdess_feature_failures.csv for details."
        )
    if split_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Unexpected split counts: {split_counts}; expected {EXPECTED_SPLIT_COUNTS}.")

    feature_path = output_dir / "ravdess_features.csv"
    metadata_path = output_dir / "ravdess_feature_metadata.json"
    feature_frame.to_csv(feature_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "feature_names": list(FEATURE_NAMES),
                "preprocessing_config": asdict(preprocessing_config),
                "emotion_mapping": RAVDESS_EMOTION_CODES,
                "split_actors": SPLIT_ACTORS,
                "split_counts": split_counts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return FeaturePreparationResult(feature_path, failed_path, metadata_path, len(rows), len(failures), split_counts)


def _extract_manifest_row(
    task: tuple[Path, dict[str, object], AudioPreprocessingConfig],
) -> tuple[dict[str, object], np.ndarray | None, dict[str, str] | None]:
    """Extract one row while returning failures for the persisted report."""
    dataset_root, manifest_row, preprocessing_config = task
    relative_path = Path(str(manifest_row["relative_path"]))
    try:
        audio, sample_rate = load_wav(dataset_root / relative_path)
        features = np.asarray(extract_features(audio, sample_rate, preprocessing_config), dtype=np.float64)
        if features.shape != (len(FEATURE_NAMES),):
            raise ValueError(f"Expected {len(FEATURE_NAMES)} features, received shape {features.shape}.")
        if not np.isfinite(features).all():
            raise ValueError("Feature extraction produced non-finite values.")
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return manifest_row, None, {"relative_path": str(relative_path), "reason": str(error)}
    return manifest_row, features, None


def _validate_feature_frame(frame: pd.DataFrame) -> None:
    """Reject incomplete, non-finite, or label-invalid feature data."""
    if frame.empty:
        raise ValueError("Feature dataset is empty.")
    if list(frame.columns[-len(FEATURE_NAMES) :]) != list(FEATURE_NAMES):
        raise ValueError("Feature columns do not match the canonical feature order.")
    if set(frame["emotion_label"]) - set(RAVDESS_EMOTION_CODES.values()):
        raise ValueError("Feature dataset contains unsupported emotion labels.")
    matrix = frame.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    if matrix.shape[1] != len(FEATURE_NAMES) or not np.isfinite(matrix).all():
        raise ValueError("Feature dataset contains invalid feature values.")


def prepare_ravdess_dataset(
    dataset_root: Path,
    output_dir: Path = PROCESSED_DATA_DIR,
    config: AudioPreprocessingConfig | None = None,
    workers: int = 4,
) -> FeaturePreparationResult:
    """Discover, report, persist, and featurize the local RAVDESS dataset."""
    manifest_result = discover_ravdess_records(dataset_root)
    write_manifest(manifest_result, output_dir)
    return build_feature_dataset(dataset_root, manifest_result.records, output_dir, config, workers)


if __name__ == "__main__":
    result = prepare_ravdess_dataset(RAW_DATA_DIR / RAVDESS_DATASET_DIRNAME, workers=1)
    print(f"Extracted feature rows: {result.extracted_count}")
    print(f"Failed feature rows: {result.failed_count}")
    print(f"Split counts: {result.split_counts}")
