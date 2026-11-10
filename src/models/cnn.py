"""Separate log-Mel CNN experiment for speaker-independent RAVDESS analysis.

TensorFlow is intentionally imported only inside functions so the established
baseline environment stays usable without the optional CNN dependency.  This
module never changes the production Random Forest artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd

from src.audio.loading import load_wav
from src.audio.preprocessing import preprocess_audio
from src.config import AudioPreprocessingConfig, EMOTION_LABELS, MODEL_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.ravdess import EXPECTED_SPLIT_COUNTS, RAVDESS_DATASET_DIRNAME, SPLIT_ACTORS
from src.models.evaluation import EvaluationResult, evaluate_predictions
from src.models.train import FEATURE_METADATA_FILENAME, _load_feature_data, _load_feature_metadata


RANDOM_SEED = 42
CNN_MODEL_FILENAME = "ravdess_logmel_cnn.keras"
CNN_METADATA_FILENAME = "ravdess_logmel_cnn_metadata.json"


@dataclass(frozen=True)
class LogMelConfig:
    """Fixed, data-independent log-Mel representation parameters."""

    n_mels: int = 64
    n_fft: int = 1_024
    hop_length: int = 256
    fmin: float = 20.0
    fmax: float = 8_000.0
    frames: int = 192
    pad_value_db: float = -80.0


@dataclass(frozen=True)
class SpectrogramNormalizer:
    """Per-Mel-bin training-only normalization statistics."""

    mean: tuple[float, ...]
    std: tuple[float, ...]


@dataclass(frozen=True)
class CNNTrainingResult:
    """Artifacts and validation metrics from one fixed CNN experiment run."""

    model_path: Path
    metadata_path: Path
    validation_evaluation: EvaluationResult
    best_epoch: int


def _tensorflow() -> Any:
    """Import TensorFlow only in the dedicated CNN environment."""
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is unavailable. Run CNN commands with .venv-cnn, not the baseline .venv."
        ) from error
    return tf


def extract_logmel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    preprocessing_config: AudioPreprocessingConfig,
    mel_config: LogMelConfig = LogMelConfig(),
) -> np.ndarray:
    """Create a fixed-size finite log-Mel image using shared audio preprocessing."""
    if mel_config.n_mels <= 0 or mel_config.n_fft <= 0 or mel_config.hop_length <= 0 or mel_config.frames <= 0:
        raise ValueError("Log-Mel dimensions must be positive.")
    if mel_config.fmin < 0 or mel_config.fmax <= mel_config.fmin:
        raise ValueError("Log-Mel frequency bounds are invalid.")
    samples = preprocess_audio(audio, sample_rate, preprocessing_config)
    if samples.size < mel_config.n_fft:
        samples = np.pad(samples, (0, mel_config.n_fft - samples.size))
    nyquist = preprocessing_config.sample_rate / 2
    if mel_config.fmax > nyquist:
        raise ValueError("Log-Mel fmax must not exceed the preprocessing sample-rate Nyquist limit.")
    mel_power = librosa.feature.melspectrogram(
        y=samples,
        sr=preprocessing_config.sample_rate,
        n_fft=mel_config.n_fft,
        hop_length=mel_config.hop_length,
        n_mels=mel_config.n_mels,
        fmin=mel_config.fmin,
        fmax=mel_config.fmax,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel_power, ref=1.0)
    if log_mel.shape[1] >= mel_config.frames:
        start = (log_mel.shape[1] - mel_config.frames) // 2
        fixed = log_mel[:, start : start + mel_config.frames]
    else:
        fixed = np.full((mel_config.n_mels, mel_config.frames), mel_config.pad_value_db, dtype=np.float32)
        fixed[:, : log_mel.shape[1]] = log_mel
    fixed = np.asarray(fixed, dtype=np.float32)
    if fixed.shape != (mel_config.n_mels, mel_config.frames) or not np.isfinite(fixed).all():
        raise ValueError("Log-Mel extraction produced an invalid spectrogram.")
    return fixed


def fit_spectrogram_normalizer(train_spectrograms: np.ndarray) -> SpectrogramNormalizer:
    """Fit per-Mel-bin normalization from training spectrograms only."""
    values = np.asarray(train_spectrograms, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("Training spectrograms must be a non-empty finite 3D array.")
    mean = np.mean(values, axis=(0, 2), dtype=np.float64)
    std = np.std(values, axis=(0, 2), dtype=np.float64)
    std = np.where(std > 1e-6, std, 1.0)
    return SpectrogramNormalizer(tuple(float(value) for value in mean), tuple(float(value) for value in std))


def apply_spectrogram_normalizer(
    spectrograms: np.ndarray, normalizer: SpectrogramNormalizer
) -> np.ndarray:
    """Apply previously fitted training statistics without refitting them."""
    values = np.asarray(spectrograms, dtype=np.float32)
    mean = np.asarray(normalizer.mean, dtype=np.float32)
    std = np.asarray(normalizer.std, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] != mean.size or mean.shape != std.shape or np.any(std <= 0):
        raise ValueError("Spectrogram normalization dimensions are invalid.")
    normalized = (values - mean[None, :, None]) / std[None, :, None]
    if not np.isfinite(normalized).all():
        raise ValueError("Spectrogram normalization produced non-finite values.")
    return normalized.astype(np.float32)


def encode_emotion_labels(labels: np.ndarray) -> np.ndarray:
    """Encode approved emotion labels in the canonical output order."""
    mapping = {label: index for index, label in enumerate(EMOTION_LABELS)}
    unknown = sorted(set(str(label) for label in labels) - set(mapping))
    if unknown:
        raise ValueError(f"Unsupported CNN labels: {unknown}")
    return np.asarray([mapping[str(label)] for label in labels], dtype=np.int64)


def _resolve_recording_path(dataset_root: Path, relative_path: str) -> Path:
    """Resolve a manifest path while preventing paths outside the dataset root."""
    root = dataset_root.resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Feature row points outside the dataset root: {relative_path}") from error
    return path


def _extract_partition_spectrograms(
    partition: pd.DataFrame,
    dataset_root: Path,
    preprocessing_config: AudioPreprocessingConfig,
    mel_config: LogMelConfig,
) -> np.ndarray:
    """Extract immutable log-Mel inputs for one already assigned actor partition."""
    images: list[np.ndarray] = []
    for relative_path in partition["relative_path"].astype(str):
        audio_path = _resolve_recording_path(dataset_root, relative_path)
        audio, sample_rate = load_wav(audio_path)
        images.append(extract_logmel_spectrogram(audio, sample_rate, preprocessing_config, mel_config))
    return np.stack(images).astype(np.float32)


def _load_partitions(
    feature_path: Path,
    dataset_root: Path,
    feature_metadata_path: Path,
    mel_config: LogMelConfig,
    split_names: tuple[str, ...] = ("train", "validation"),
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], AudioPreprocessingConfig]:
    """Load validated split rows and create fixed inputs without fitting on them."""
    feature_metadata = _load_feature_metadata(feature_metadata_path)
    frame = _load_feature_data(feature_path)
    if "relative_path" not in frame:
        raise ValueError("Feature dataset is missing relative_path required for CNN audio loading.")
    features: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for split_name in split_names:
        expected_count = EXPECTED_SPLIT_COUNTS[split_name]
        partition = frame.loc[frame["split"] == split_name]
        if len(partition) != expected_count:
            raise ValueError(f"Unexpected {split_name} partition size for CNN: {len(partition)}.")
        features[split_name] = _extract_partition_spectrograms(
            partition, dataset_root, feature_metadata.preprocessing_config, mel_config
        )
        labels[split_name] = partition["emotion_label"].to_numpy(dtype=str)
    return features, labels, feature_metadata.preprocessing_config


def _class_weights(encoded_labels: np.ndarray) -> dict[int, float]:
    """Compute deterministic balanced weights solely from training labels."""
    counts = np.bincount(encoded_labels, minlength=len(EMOTION_LABELS))
    if np.any(counts == 0):
        raise ValueError("Every approved emotion must appear in the CNN training split.")
    total = int(encoded_labels.size)
    return {index: total / (len(EMOTION_LABELS) * int(count)) for index, count in enumerate(counts)}


def _build_model(input_shape: tuple[int, int, int]) -> Any:
    """Build the fixed regularized CNN architecture without training it."""
    tf = _tensorflow()
    regularizer = tf.keras.regularizers.l2(1e-4)
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu", kernel_regularizer=regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=regularizer),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizer),
            tf.keras.layers.Dropout(0.40),
            tf.keras.layers.Dense(len(EMOTION_LABELS), activation="softmax"),
        ]
    )


def train_logmel_cnn(
    feature_path: Path = PROCESSED_DATA_DIR / "ravdess_features.csv",
    dataset_root: Path = RAW_DATA_DIR / RAVDESS_DATASET_DIRNAME,
    model_dir: Path = MODEL_DIR,
    feature_metadata_path: Path | None = None,
    mel_config: LogMelConfig = LogMelConfig(),
    epochs: int = 40,
    batch_size: int = 32,
    patience: int = 8,
) -> CNNTrainingResult:
    """Train a fixed CNN on train actors and select its stopping epoch on validation.

    This deliberately does not extract, fit, or evaluate the test partition.
    Call :func:`evaluate_saved_cnn_test` exactly once after accepting the
    validation result.
    """
    if epochs <= 0 or batch_size <= 0 or patience <= 0:
        raise ValueError("epochs, batch_size, and patience must be positive.")
    metadata_path = feature_metadata_path or feature_path.parent / FEATURE_METADATA_FILENAME
    split_images, split_labels, preprocessing_config = _load_partitions(
        feature_path, dataset_root, metadata_path, mel_config
    )
    train_images = split_images["train"]
    validation_images = split_images["validation"]
    normalizer = fit_spectrogram_normalizer(train_images)
    x_train = apply_spectrogram_normalizer(train_images, normalizer)[..., np.newaxis]
    x_validation = apply_spectrogram_normalizer(validation_images, normalizer)[..., np.newaxis]
    y_train = encode_emotion_labels(split_labels["train"])
    y_validation = encode_emotion_labels(split_labels["validation"])

    tf = _tensorflow()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    model = _build_model(tuple(int(value) for value in x_train.shape[1:]))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=_class_weights(y_train),
        callbacks=callbacks,
        verbose=2,
    )
    validation_indices = np.argmax(model.predict(x_validation, verbose=0), axis=1)
    validation_predictions = np.asarray(EMOTION_LABELS, dtype=str)[validation_indices]
    validation_evaluation = evaluate_predictions(split_labels["validation"], validation_predictions)
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / CNN_MODEL_FILENAME
    model_metadata_path = model_dir / CNN_METADATA_FILENAME
    model.save(model_path)
    model_metadata_path.write_text(
        json.dumps(
            {
                "model_type": "logmel_cnn_experiment",
                "emotion_labels": list(EMOTION_LABELS),
                "random_seed": RANDOM_SEED,
                "actor_splits": SPLIT_ACTORS,
                "split_counts": EXPECTED_SPLIT_COUNTS,
                "preprocessing_config": asdict(preprocessing_config),
                "logmel_config": asdict(mel_config),
                "normalizer": asdict(normalizer),
                "input_shape": list(x_train.shape[1:]),
                "training_parameters": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "early_stopping_patience": patience,
                    "optimizer": "Adam(learning_rate=0.001)",
                    "class_weight": "balanced_from_train_only",
                },
                "best_epoch": best_epoch,
                "validation_evaluation": validation_evaluation.to_dict(),
                "test_evaluation": None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return CNNTrainingResult(model_path, model_metadata_path, validation_evaluation, best_epoch)


def _load_cnn_metadata(metadata_path: Path) -> dict[str, Any]:
    """Load a saved CNN contract and reject incomplete experimental artifacts."""
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"CNN metadata is not valid JSON: {metadata_path}") from error
    required = {"emotion_labels", "preprocessing_config", "logmel_config", "normalizer", "test_evaluation"}
    if not isinstance(metadata, dict) or required - set(metadata):
        raise ValueError("CNN metadata is incomplete.")
    if metadata["emotion_labels"] != list(EMOTION_LABELS):
        raise ValueError("CNN metadata emotion labels do not match the canonical order.")
    return metadata


def evaluate_saved_cnn_test(
    feature_path: Path = PROCESSED_DATA_DIR / "ravdess_features.csv",
    dataset_root: Path = RAW_DATA_DIR / RAVDESS_DATASET_DIRNAME,
    model_path: Path = MODEL_DIR / CNN_MODEL_FILENAME,
    metadata_path: Path = MODEL_DIR / CNN_METADATA_FILENAME,
) -> EvaluationResult:
    """Perform the one final held-out test evaluation for an accepted CNN artifact."""
    metadata = _load_cnn_metadata(metadata_path)
    if metadata["test_evaluation"] is not None:
        raise RuntimeError("This CNN artifact already has a recorded test evaluation.")
    preprocessing_config = AudioPreprocessingConfig(**metadata["preprocessing_config"])
    mel_config = LogMelConfig(**metadata["logmel_config"])
    normalizer = SpectrogramNormalizer(
        tuple(metadata["normalizer"]["mean"]), tuple(metadata["normalizer"]["std"])
    )
    frame = _load_feature_data(feature_path)
    test_partition = frame.loc[frame["split"] == "test"]
    test_images = _extract_partition_spectrograms(test_partition, dataset_root, preprocessing_config, mel_config)
    x_test = apply_spectrogram_normalizer(test_images, normalizer)[..., np.newaxis]
    tf = _tensorflow()
    model = tf.keras.models.load_model(model_path)
    predicted_indices = np.argmax(model.predict(x_test, verbose=0), axis=1)
    predictions = np.asarray(EMOTION_LABELS, dtype=str)[predicted_indices]
    evaluation = evaluate_predictions(test_partition["emotion_label"].to_numpy(dtype=str), predictions)
    metadata["test_evaluation"] = evaluation.to_dict()
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return evaluation


def predict_logmel_wav(model: Any, audio_path: Path, metadata_path: Path) -> tuple[str, dict[str, float]]:
    """Predict one WAV with an accepted CNN artifact in canonical label order."""
    metadata = _load_cnn_metadata(metadata_path)
    preprocessing_config = AudioPreprocessingConfig(**metadata["preprocessing_config"])
    mel_config = LogMelConfig(**metadata["logmel_config"])
    normalizer = SpectrogramNormalizer(
        tuple(metadata["normalizer"]["mean"]), tuple(metadata["normalizer"]["std"])
    )
    audio, sample_rate = load_wav(audio_path)
    image = extract_logmel_spectrogram(audio, sample_rate, preprocessing_config, mel_config)
    inputs = apply_spectrogram_normalizer(image[None, ...], normalizer)[..., np.newaxis]
    probabilities_array = np.asarray(model.predict(inputs, verbose=0), dtype=np.float64)
    if probabilities_array.shape != (1, len(EMOTION_LABELS)) or not np.isfinite(probabilities_array).all():
        raise ValueError("CNN returned an invalid probability shape.")
    probabilities = {label: float(value) for label, value in zip(EMOTION_LABELS, probabilities_array[0], strict=True)}
    label = max(probabilities, key=probabilities.__getitem__)
    return label, probabilities


if __name__ == "__main__":
    result = train_logmel_cnn()
    print(f"Saved experimental CNN: {result.model_path}")
    print(f"Best validation epoch: {result.best_epoch}")
    print(f"Validation accuracy: {result.validation_evaluation.accuracy:.4f}")
    print(f"Validation macro F1: {result.validation_evaluation.macro_f1:.4f}")
