"""Tests for RAVDESS parsing, preparation, splitting, and baseline training."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.config import AudioPreprocessingConfig, FEATURE_NAMES
from src.data import ravdess
from src.data.ravdess import (
    EXPECTED_SPLIT_COUNTS,
    RAVDESS_EMOTION_CODES,
    RavdessRecord,
    SPLIT_ACTORS,
    add_speaker_split,
    build_feature_dataset,
    discover_ravdess_records,
    parse_ravdess_filename,
)
from src.models.predict import load_saved_preprocessing_config
from src.models.train import train_random_forest


def _record(actor_id: str, emotion_code: str = "01", index: int = 1) -> RavdessRecord:
    """Create deterministic metadata for one approved synthetic record."""
    return RavdessRecord(
        relative_path=f"Actor_{actor_id}/03-01-{emotion_code}-01-01-01-{actor_id}.wav",
        filename=f"03-01-{emotion_code}-01-01-01-{actor_id}.wav",
        modality="03",
        vocal_channel="01",
        emotion_code=emotion_code,
        emotion_label=RAVDESS_EMOTION_CODES[emotion_code],
        intensity="01",
        statement="01",
        repetition=f"{index:02d}",
        actor_id=actor_id,
    )


def _training_rows() -> list[dict[str, object]]:
    """Build a deterministic, complete synthetic five-class feature dataset."""
    rows: list[dict[str, object]] = []
    for actor in range(1, 25):
        actor_id = f"{actor:02d}"
        split = "train" if actor <= 16 else "validation" if actor <= 20 else "test"
        for code_index, (emotion_code, emotion_label) in enumerate(RAVDESS_EMOTION_CODES.items()):
            repetitions = 4 if emotion_code == "01" else 8
            for repetition in range(repetitions):
                features = np.full(len(FEATURE_NAMES), float(code_index), dtype=float)
                features += actor / 1_000 + repetition / 10_000
                rows.append(
                    {
                        "split": split,
                        "actor_id": actor_id,
                        "emotion_label": emotion_label,
                        **dict(zip(FEATURE_NAMES, features, strict=True)),
                    }
                )
    return rows


def _write_feature_metadata(path: Path, config: AudioPreprocessingConfig | None = None) -> None:
    """Write the exact contract emitted by local feature preparation."""
    preprocessing_config = config or AudioPreprocessingConfig()
    path.write_text(
        json.dumps(
            {
                "feature_names": list(FEATURE_NAMES),
                "preprocessing_config": asdict(preprocessing_config),
                "emotion_mapping": RAVDESS_EMOTION_CODES,
                "split_actors": {name: list(actors) for name, actors in SPLIT_ACTORS.items()},
                "split_counts": EXPECTED_SPLIT_COUNTS,
            }
        ),
        encoding="utf-8",
    )


def _write_training_inputs(
    directory: Path, config: AudioPreprocessingConfig | None = None
) -> tuple[Path, Path]:
    """Write a complete valid synthetic dataset and its matching metadata."""
    feature_path = directory / "features.csv"
    metadata_path = directory / "ravdess_feature_metadata.json"
    pd.DataFrame(_training_rows()).to_csv(feature_path, index=False)
    _write_feature_metadata(metadata_path, config)
    return feature_path, metadata_path


def test_parse_filename_and_emotion_mapping() -> None:
    """The seven RAVDESS filename fields and approved label mapping are stable."""
    parsed = parse_ravdess_filename("03-01-05-02-02-01-24.wav")

    assert parsed == {
        "modality": "03",
        "vocal_channel": "01",
        "emotion_code": "05",
        "intensity": "02",
        "statement": "02",
        "repetition": "01",
        "actor_id": "24",
    }
    assert RAVDESS_EMOTION_CODES[parsed["emotion_code"]] == "angry"
    with pytest.raises(ValueError, match="Malformed"):
        parse_ravdess_filename("not-ravdess.wav")


def test_discovery_filters_non_speech_and_unsupported_emotions(tmp_path: Path) -> None:
    """Only audio-only speech records for the agreed five labels are selected."""
    actor_dir = tmp_path / "Actor_01"
    actor_dir.mkdir()
    for name in (
        "03-01-01-01-01-01-01.wav",
        "03-02-01-01-01-01-01.wav",
        "03-01-06-01-01-01-01.wav",
        "malformed.wav",
    ):
        (actor_dir / name).touch()

    result = discover_ravdess_records(tmp_path)

    assert [record.emotion_label for record in result.records] == ["neutral"]
    assert len(result.skipped) == 3
    assert any("vocal channel" in item.reason for item in result.skipped)
    assert any("emotion code 06" in item.reason for item in result.skipped)
    assert any("Malformed" in item.reason for item in result.skipped)


def test_actor_splits_are_isolated_and_have_expected_counts() -> None:
    """Every actor belongs to exactly one fixed split with expected row counts."""
    records: list[RavdessRecord] = []
    for actor in range(1, 25):
        actor_id = f"{actor:02d}"
        records.extend(_record(actor_id, "01", index) for index in range(4))
        for emotion_code in ("02", "03", "04", "05"):
            records.extend(_record(actor_id, emotion_code, index) for index in range(8))

    frame = add_speaker_split(records)

    assert frame.groupby("actor_id")["split"].nunique().eq(1).all()
    assert frame["split"].value_counts().to_dict() == EXPECTED_SPLIT_COUNTS


def test_feature_dataset_has_canonical_finite_matrix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prepared records create one finite row in the established feature order."""
    dataset_root = tmp_path / "raw"
    actor_dir = dataset_root / "Actor_01"
    actor_dir.mkdir(parents=True)
    filename = "03-01-01-01-01-01-01.wav"
    time = np.arange(48_000, dtype=np.float32) / 48_000
    sf.write(actor_dir / filename, 0.1 * np.sin(2 * np.pi * 220 * time), 48_000)
    record = _record("01")
    monkeypatch.setattr(ravdess, "EXPECTED_SPLIT_COUNTS", {"train": 1, "validation": 0, "test": 0})

    result = build_feature_dataset(dataset_root, [record], tmp_path / "processed")
    frame = pd.read_csv(result.feature_path)

    assert frame.loc[:, FEATURE_NAMES].shape == (1, 32)
    assert np.isfinite(frame.loc[:, FEATURE_NAMES].to_numpy(dtype=float)).all()
    assert list(frame.loc[:, FEATURE_NAMES].columns) == list(FEATURE_NAMES)


def test_random_forest_training_and_probability_shape(tmp_path: Path) -> None:
    """Training persists a classifier with one probability column per emotion."""
    feature_path, metadata_path = _write_training_inputs(tmp_path)

    result = train_random_forest(feature_path, tmp_path / "models", metadata_path)
    model = joblib.load(result.model_path)
    probabilities = model.predict_proba(pd.DataFrame(_training_rows()).loc[:2, FEATURE_NAMES].to_numpy())

    assert result.test_evaluation.accuracy == pytest.approx(1.0)
    assert probabilities.shape == (3, len(RAVDESS_EMOTION_CODES))
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_training_rejects_incorrect_actor_assignments_with_valid_counts(tmp_path: Path) -> None:
    """A count-preserving actor swap cannot bypass the fixed split contract."""
    feature_path, metadata_path = _write_training_inputs(tmp_path)
    frame = pd.read_csv(feature_path, dtype={"actor_id": str})
    frame["actor_id"] = frame["actor_id"].replace({"01": "17", "17": "01"})
    frame.to_csv(feature_path, index=False)

    with pytest.raises(ValueError, match="Invalid actors for train split"):
        train_random_forest(feature_path, tmp_path / "models", metadata_path)


def test_custom_preprocessing_config_round_trips_to_model_metadata(tmp_path: Path) -> None:
    """Model metadata retains the exact config used to generate its features."""
    custom_config = AudioPreprocessingConfig(sample_rate=16_000, mono=False, trim_silence=False, trim_top_db=25.0)
    feature_path, metadata_path = _write_training_inputs(tmp_path, custom_config)

    result = train_random_forest(feature_path, tmp_path / "models", metadata_path)
    saved_metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert saved_metadata["feature_names"] == list(FEATURE_NAMES)
    assert saved_metadata["preprocessing_config"] == asdict(custom_config)
    assert load_saved_preprocessing_config(result.metadata_path) == custom_config


@pytest.mark.parametrize("metadata_contents, expected_error", [(None, "Feature metadata was not found"), ({}, "missing required fields")])
def test_training_rejects_missing_or_invalid_feature_metadata(
    tmp_path: Path, metadata_contents: dict[str, object] | None, expected_error: str
) -> None:
    """Training never falls back to defaults when feature metadata is unavailable or invalid."""
    feature_path, metadata_path = _write_training_inputs(tmp_path)
    if metadata_contents is None:
        metadata_path.unlink()
    else:
        metadata_path.write_text(json.dumps(metadata_contents), encoding="utf-8")

    with pytest.raises((FileNotFoundError, ValueError), match=expected_error):
        train_random_forest(feature_path, tmp_path / "models", metadata_path)
