# Speech Emotion & Wellbeing Signal Recognition System

A university machine-learning project that recognizes five speech emotions from WAV audio: **neutral**, **calm**, **happy**, **sad**, and **angry**. The planned application uses Librosa for audio features, a Random Forest baseline model, and Streamlit for interactive predictions and visualizations.

## Scope

The baseline extracts MFCC, pitch, RMS energy, and zero-crossing-rate features. It will display the predicted emotion, class probabilities, waveform, and spectrogram. This repository currently contains only the Phase 1 project foundation; no dataset has been downloaded and no model has been trained.

## Architecture

```text
app.py                         Streamlit entry point (placeholder)
src/
  config.py                    Shared labels, paths, and fixed feature schema
  audio/                       WAV validation, preprocessing, feature extraction
  models/                      Training, inference, and evaluation modules
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

## Setup (when implementation begins)

Use Python 3.13.3 and the existing virtual environment, then install the listed project requirements:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

See `AGENTS.md` for collaboration and Git rules, and `data/README.md` before adding any dataset files.
