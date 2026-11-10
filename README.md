# Speech Emotion & Wellbeing Signal Recognition System

A university machine-learning project that recognizes five speech emotions from WAV audio: **neutral**, **calm**, **happy**, **sad**, and **angry**. The planned application uses Librosa for audio features, a Random Forest baseline model, and Streamlit for interactive predictions and visualizations.

## Scope

The baseline extracts MFCC, pitch, RMS energy, and zero-crossing-rate features. The RAVDESS Audio Speech Actors dataset has been downloaded locally under `data/raw/` and is prepared locally into an ignored manifest and feature dataset. The baseline is a research/educational classifier of acted speech emotion; it is not a clinical diagnostic tool and does not detect mental-health conditions.

## Dataset and local preparation

This project uses the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**, Audio Speech Actors subset (Livingstone & Russo, 2018). The local dataset is licensed under **CC BY-NC-SA 4.0**. Dataset recordings and derived feature files remain local and are excluded from Git.

The preparation pipeline recursively reads `data/raw/Audio_Speech_Actors_01-24/`, retains audio-only speech (`03-01`) for the five approved labels, and writes ignored local artifacts under `data/processed/`. The artifacts include the manifest, skipped-file report, feature table, failed-feature report, and feature metadata.

Speaker-independent partitions are fixed by actor: Actors 01–16 train, 17–20 validation, and 21–24 test. The baseline trains only on training actors, selects its configuration with validation actors, and evaluates the test actors once.

## Phase 6 experimental models

The Random Forest remains the production model used by default in Streamlit. Local, ignored experimental artifacts may also be created under `models/`:

- RBF-SVM: `StandardScaler` followed by the fixed RBF configuration selected on validation actors only (`C=1`, `gamma="scale"`, `class_weight="balanced"`, `random_state=42`).
- Log-Mel CNN: a separate CPU-only TensorFlow experiment using 64-band fixed-size log-Mel spectrograms, normalization fitted only on training actors, regularization, and validation-loss early stopping.

The CNN uses a dedicated environment and does not change the main `.venv` or the production Random Forest:

```powershell
C:\Program Files\Python313\python.exe -m venv .venv-cnn
.\.venv-cnn\Scripts\python.exe -m pip install -r requirements.txt tensorflow-cpu==2.21.0
.\.venv-cnn\Scripts\python.exe -m src.models.cnn
```

Run the CNN on held-out test actors only after accepting its validation result. Its final-evaluation helper records the test result and rejects a second run for the same artifact. Model files, local data, and `.venv-cnn` are ignored by Git.

The optional Streamlit comparison is explicitly for external synthetic-audio evaluation (for example, a supplied ElevenLabs WAV). It does not add that WAV to RAVDESS, persist it, or use it for training. It classifies acted vocal-expression categories only and must not be interpreted as a mental-health or wellbeing assessment.

## v2 classical research experiments

The opt-in 40-feature schema has also been used for separate Logistic
Regression and XGBoost experiments. Their feature data and model artifacts are
ignored locally and do not replace the production Random Forest, RBF-SVM
experiment, or CNN. Standardization for Logistic Regression is fitted on
training actors only; fixed candidate configurations are selected by validation
macro F1 before one held-out test evaluation. See
[`docs/classical_v2_comparison.md`](docs/classical_v2_comparison.md) for the
protocol, selected settings, per-class metrics, and canonical-order confusion
matrices. Results from the 40-feature experiments should not be presented as a
direct replacement ranking of the historical 32-feature models.

## Architecture

```text
app.py                         Streamlit entry point (placeholder)
src/
  config.py                    Shared labels, paths, and fixed feature schema
  audio/                       WAV validation, preprocessing, feature extraction
  data/                        RAVDESS manifest, split, and feature preparation
  models/                      Random Forest training, inference, and evaluation
  visualization/               Waveform and spectrogram visualizations
data/README.md                 Dataset storage instructions
data/raw/                      Local source audio; ignored by Git
data/processed/                Derived data/features; ignored by Git
models/                        Serialized model artifacts; ignored by Git
tests/                         Unit tests for contracts and future modules
```

## Feature schema

The stable baseline feature vector has 32 values, in this exact order:

1. `mfcc_01_mean` through `mfcc_13_mean`
2. `mfcc_01_std` through `mfcc_13_std`
3. `pitch_mean`, `pitch_std`
4. `rms_mean`, `rms_std`
5. `zcr_mean`, `zcr_std`

Training and inference must use the same preprocessing configuration and this exact schema. The canonical definition is `src.config.FEATURE_NAMES`.

### Acoustic feature versions

`acoustic_v1_32` is the immutable production schema used by the saved Random Forest and RBF-SVM artifacts. Audio is validated, downmixed to mono, resampled to the persisted sample rate, and optionally silence-trimmed before all descriptors use a 2,048-sample analysis frame and 512-sample hop. MFCCs, pitch, RMS energy, and ZCR are summarized as population mean and standard deviation over frames; non-finite pitch frames are excluded before aggregation. No dataset-fitted normalization is performed during v1 feature extraction.

`acoustic_v2_40_spectral` is an opt-in, backward-compatible research schema. It preserves all 32 v1 values in the same order, then adds these eight values:

1. `spectral_centroid_mean`, `spectral_centroid_std`
2. `spectral_bandwidth_mean`, `spectral_bandwidth_std`
3. `spectral_rolloff_mean`, `spectral_rolloff_std` (85% energy rolloff)
4. `spectral_contrast_mean`, `spectral_contrast_std`

The first three spectral descriptors are aggregated over time frames. Librosa produces six spectral-contrast bands; they are averaged per frame before the same population mean/std aggregation. The v2 extractor uses the identical shared `AudioPreprocessingConfig`, sample rate, frame size, and hop size as v1, applies no fitted normalization, and is not connected to historical model artifacts. Future v2 training and inference must persist and validate `EXTENDED_FEATURE_NAMES` with the preprocessing configuration exactly as v1 does for `FEATURE_NAMES`.

## Development roles

- Audio/features: loading, preprocessing, feature extraction, and feature tests.
- Machine learning: labels, speaker-independent splits, Random Forest, and metrics.
- Frontend: Streamlit upload workflow, probabilities, and visualizations.

## Setup and local commands

Use Python 3.13.3 and the existing virtual environment, then install the listed project requirements:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

See `AGENTS.md` for collaboration and Git rules, and `data/README.md` before adding any dataset files.

Prepare the local data and train after installing dependencies:

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; from src.data.ravdess import prepare_ravdess_dataset; prepare_ravdess_dataset(Path('data/raw/Audio_Speech_Actors_01-24'))"
.\.venv\Scripts\python.exe -m src.models.train
```
