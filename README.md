# Speech Emotion & Wellbeing Signal Recognition System

A university machine-learning project that recognizes five speech emotions from WAV audio: **neutral**, **calm**, **happy**, **sad**, and **angry**. The planned application uses Librosa for audio features, a Random Forest baseline model, and Streamlit for interactive predictions and visualizations.

## Scope

The baseline extracts MFCC, pitch, RMS energy, and zero-crossing-rate features. The RAVDESS Audio Speech Actors dataset has been downloaded locally under `data/raw/` and is prepared locally into an ignored manifest and feature dataset. The baseline is a research/educational classifier of acted speech emotion; it is not a clinical diagnostic tool and does not detect mental-health conditions.

## Dataset and local preparation

This project uses the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**, Audio Speech Actors subset (Livingstone & Russo, 2018). The local dataset is licensed under **CC BY-NC-SA 4.0**. Dataset recordings and derived feature files remain local and are excluded from Git.

The preparation pipeline recursively reads `data/raw/Audio_Speech_Actors_01-24/`, retains audio-only speech (`03-01`) for the five approved labels, and writes ignored local artifacts under `data/processed/`. The artifacts include the manifest, skipped-file report, feature table, failed-feature report, and feature metadata.

Speaker-independent partitions are fixed by actor: Actors 01–16 train, 17–20 validation, and 21–24 test. The baseline trains only on training actors, selects its configuration with validation actors, and evaluates the test actors once.

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
