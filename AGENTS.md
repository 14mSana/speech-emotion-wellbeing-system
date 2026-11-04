# Project instructions

## Purpose and scope

Build a small, reproducible speech-emotion baseline for five labels: `neutral`, `calm`, `happy`, `sad`, and `angry`. Use Python 3.13.3. Keep the solution suitable for a three-person team and a five-day deadline: Librosa features, a Random Forest baseline, and a Streamlit interface only.

## Code standards

- Put application logic in `src/`; keep `app.py` thin.
- Use type hints, concise docstrings, `pathlib.Path`, and deterministic configuration.
- Keep loading, preprocessing, feature extraction, model work, and visualization in their respective modules.
- Reuse `AudioPreprocessingConfig` and `FEATURE_NAMES` for both training and inference. Never create a different inference feature order.
- The approved feature schema is 13 MFCC means, 13 MFCC standard deviations, then pitch, RMS, and ZCR mean/std pairs (32 values total).
- Validate WAV inputs and fail with clear errors. Do not silently alter labels.
- Keep dependencies limited to `requirements.txt`; avoid deep-learning frameworks.

## Model and evaluation requirements

- Support exactly the five labels in `src.config.EMOTION_LABELS` unless the team explicitly agrees to a schema change.
- Persist preprocessing configuration and feature schema alongside any future model.
- Use speaker-independent train/validation/test grouping. Never allow the same speaker in both train and evaluation partitions.
- Report per-class metrics, a confusion matrix, and probability outputs when model work is implemented.

## Testing

- Add or update tests in `tests/` for every behavioral change.
- Run `python -m pytest` before handing off changes once dependencies are installed.
- Test feature order, expected vector length, label constraints, invalid audio, and prediction-probability shape as relevant.

## Git safety

- Never commit raw or processed datasets, `.venv`, model binaries, secrets, or `.streamlit/secrets.toml`.
- Do not rewrite history, force-push, change remotes, or discard another developer's work without explicit authorization.
- Inspect `git status` before staging. Make small, focused commits only when asked.
- Do not download datasets, train models, or publish artifacts without approval.
