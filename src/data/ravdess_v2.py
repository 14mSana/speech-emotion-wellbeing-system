"""Local preparation of the opt-in 40-feature RAVDESS acoustic dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.audio.features import extract_extended_features
from src.audio.loading import load_wav
from src.config import EXTENDED_FEATURE_NAMES, FEATURE_VERSION_V2, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.ravdess import RAVDESS_DATASET_DIRNAME
from src.models.train import FEATURE_METADATA_FILENAME, _load_feature_data, _load_feature_metadata


V2_FEATURE_FILENAME = "ravdess_v2_features.csv"
V2_METADATA_FILENAME = "ravdess_v2_feature_metadata.json"
V2_FAILURES_FILENAME = "ravdess_v2_feature_failures.csv"


@dataclass(frozen=True)
class ExtendedFeaturePreparationResult:
    """Outputs from preparing the independent v2 feature dataset."""

    feature_path: Path
    metadata_path: Path
    failure_path: Path
    extracted_count: int


def _extract_one(
    dataset_root: Path, relative_path: str, preprocessing_config: object
) -> tuple[tuple[float, ...] | None, str | None]:
    """Extract one v2 vector while preserving a failure reason for the report."""
    try:
        audio, sample_rate = load_wav(dataset_root / relative_path)
        return extract_extended_features(audio, sample_rate, preprocessing_config), None
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return None, str(error)


def prepare_extended_feature_dataset(
    source_feature_path: Path = PROCESSED_DATA_DIR / "ravdess_features.csv",
    source_metadata_path: Path | None = None,
    dataset_root: Path = RAW_DATA_DIR / RAVDESS_DATASET_DIRNAME,
    output_dir: Path = PROCESSED_DATA_DIR,
    workers: int = 4,
) -> ExtendedFeaturePreparationResult:
    """Build a separate v2 table from the validated, actor-isolated v1 manifest.

    The source table is used only to retain its validated paths, labels, and
    fixed split assignments. Audio is re-read and processed through the v2
    extractor. Existing v1 outputs are never replaced.
    """
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"RAVDESS dataset directory was not found: {dataset_root}")
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer.")
    metadata_path = source_metadata_path or source_feature_path.parent / FEATURE_METADATA_FILENAME
    source_metadata = _load_feature_metadata(metadata_path)
    source_frame = _load_feature_data(source_feature_path)
    if "relative_path" not in source_frame:
        raise ValueError("The source feature dataset is missing relative_path.")

    relative_paths = source_frame["relative_path"].astype(str).tolist()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        outcomes = list(
            executor.map(
                lambda relative_path: _extract_one(dataset_root, relative_path, source_metadata.preprocessing_config),
                relative_paths,
            )
        )
    failures = [
        {"relative_path": relative_path, "reason": reason}
        for relative_path, (_, reason) in zip(relative_paths, outcomes, strict=True)
        if reason is not None
    ]
    if failures:
        raise RuntimeError(
            f"v2 extraction failed for {len(failures)} recording(s); no incomplete dataset was accepted."
        )
    vectors = [vector for vector, _ in outcomes]
    if any(vector is None for vector in vectors):
        raise RuntimeError("v2 extraction returned an unexpected empty vector.")
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.shape != (len(source_frame), len(EXTENDED_FEATURE_NAMES)) or not np.isfinite(matrix).all():
        raise ValueError("v2 extraction produced invalid feature vectors.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_frame = source_frame.loc[:, [column for column in source_frame.columns if column not in source_metadata.feature_names]].copy()
    for index, name in enumerate(EXTENDED_FEATURE_NAMES):
        output_frame[name] = matrix[:, index]
    feature_path = output_dir / V2_FEATURE_FILENAME
    output_metadata_path = output_dir / V2_METADATA_FILENAME
    failure_path = output_dir / V2_FAILURES_FILENAME
    output_frame.to_csv(feature_path, index=False)
    pd.DataFrame(failures, columns=("relative_path", "reason")).to_csv(failure_path, index=False)
    output_metadata_path.write_text(
        json.dumps(
            {
                "feature_version": FEATURE_VERSION_V2,
                "feature_names": list(EXTENDED_FEATURE_NAMES),
                "preprocessing_config": asdict(source_metadata.preprocessing_config),
                "source_feature_metadata": metadata_path.name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ExtendedFeaturePreparationResult(feature_path, output_metadata_path, failure_path, len(output_frame))
