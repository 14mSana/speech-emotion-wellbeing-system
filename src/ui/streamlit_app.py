"""Streamlit presentation for local WAV emotion-category predictions."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.audio.loading import load_wav
from src.config import EMOTION_LABELS, MODEL_DIR
from src.models.predict import load_saved_preprocessing_config, predict_wav
from src.visualization.plots import plot_spectrogram, plot_waveform


MODEL_PATH = MODEL_DIR / "ravdess_random_forest.joblib"
MODEL_METADATA_PATH = MODEL_DIR / "ravdess_random_forest_metadata.json"
SVM_MODEL_PATH = MODEL_DIR / "ravdess_rbf_svm.joblib"
SVM_MODEL_METADATA_PATH = MODEL_DIR / "ravdess_rbf_svm_metadata.json"
CNN_MODEL_PATH = MODEL_DIR / "ravdess_logmel_cnn.keras"
CNN_MODEL_METADATA_PATH = MODEL_DIR / "ravdess_logmel_cnn_metadata.json"
DISPLAY_LABELS: dict[str, str] = {
    "neutral": "Neutral",
    "calm": "Calm",
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
}
PREDICTED_PROBABILITY_COLUMN = "Predicted probability"


@st.cache_resource(show_spinner=False)
def load_prediction_model(model_path: Path = MODEL_PATH, metadata_path: Path = MODEL_METADATA_PATH) -> object:
    """Load the existing trained model once and validate its saved configuration."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model was not found: {model_path}")
    load_saved_preprocessing_config(metadata_path)
    return joblib.load(model_path)


@st.cache_resource(show_spinner=False)
def load_experimental_svm_model(
    model_path: Path = SVM_MODEL_PATH, metadata_path: Path = SVM_MODEL_METADATA_PATH
) -> object:
    """Load the optional RBF-SVM experiment without altering the production model."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Experimental SVM model was not found: {model_path}")
    load_saved_preprocessing_config(metadata_path)
    return joblib.load(model_path)


def _persist_uploaded_wav(uploaded_audio: Any) -> Path:
    """Write one uploaded WAV to a temporary file for the existing path-based pipeline."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temporary_file:
        temporary_file.write(uploaded_audio.getvalue())
        return Path(temporary_file.name)


def _audio_details(audio_path: Path) -> tuple[np.ndarray, int, int, float]:
    """Load source samples and calculate display-only recording properties."""
    audio, sample_rate = load_wav(audio_path)
    channels = 1 if audio.ndim == 1 else int(audio.shape[1])
    duration = float(audio.shape[0] / sample_rate)
    return audio, sample_rate, channels, duration


def _render_header() -> None:
    """Render the project title, scope, and safety disclaimer."""
    st.title("Speech Emotion & Wellbeing Signal Recognition")
    st.caption("A reproducible Random Forest baseline for five acted speech-emotion categories.")
    st.warning(
        "This tool classifies **acted vocal-expression categories** in a WAV recording. "
        "It is not a mental-health diagnosis and is not a reliable assessment of a person's wellbeing."
    )


def build_probability_frame(probabilities: dict[str, float]) -> pd.DataFrame:
    """Return display probabilities in the stable canonical emotion order."""
    return pd.DataFrame(
        {
            "Emotion": [DISPLAY_LABELS[label] for label in EMOTION_LABELS],
            PREDICTED_PROBABILITY_COLUMN: [probabilities[label] for label in EMOTION_LABELS],
        }
    ).set_index("Emotion")


def _render_probability_chart(probabilities: dict[str, float]) -> None:
    """Render predicted probabilities in the canonical project label order."""
    probability_frame = build_probability_frame(probabilities)
    st.bar_chart(probability_frame, color="#4cc9f0", height=280)
    st.dataframe(
        probability_frame.style.format({PREDICTED_PROBABILITY_COLUMN: "{:.1%}"}),
        use_container_width=True,
    )


def _render_prediction(audio_path: Path) -> None:
    """Run the fixed inference pipeline and render its outcome or a clear error."""
    try:
        model = load_prediction_model()
        label, probabilities = predict_wav(model, audio_path, MODEL_METADATA_PATH)
    except (FileNotFoundError, TypeError, ValueError, OSError) as error:
        st.error(f"Prediction could not be completed: {error}")
        return

    predicted_probability = probabilities[label]
    st.subheader("Prediction")
    st.metric(
        "Predicted vocal-expression category",
        DISPLAY_LABELS[label],
        f"{predicted_probability:.1%} predicted probability",
    )
    st.write(
        f"The model found acoustic patterns most similar to the **{DISPLAY_LABELS[label].lower()}** "
        "acted speech category in its training data. This is a model output, not a statement about a speaker's feelings or health."
    )
    st.subheader("Class probabilities")
    _render_probability_chart(probabilities)


def _render_external_synthetic_comparison(audio_path: Path) -> None:
    """Optionally compare saved experiments on an external synthetic WAV only."""
    st.subheader("External synthetic-audio evaluation")
    st.caption(
        "Optional comparison for an external synthetic recording (for example, ElevenLabs). "
        "This recording is not added to RAVDESS or used for training."
    )
    if not (SVM_MODEL_PATH.is_file() and SVM_MODEL_METADATA_PATH.is_file()):
        st.info("The experimental RBF-SVM artifact is not available locally.")
    else:
        try:
            model = load_experimental_svm_model()
            label, probabilities = predict_wav(model, audio_path, SVM_MODEL_METADATA_PATH)
            st.markdown(f"**Experimental RBF-SVM:** {DISPLAY_LABELS[label]}")
            _render_probability_chart(probabilities)
        except (FileNotFoundError, TypeError, ValueError, OSError) as error:
            st.error(f"Experimental SVM comparison could not be completed: {error}")

    if CNN_MODEL_PATH.is_file() and CNN_MODEL_METADATA_PATH.is_file():
        try:
            import tensorflow as tf
            from src.models.cnn import predict_logmel_wav

            cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
            label, probabilities = predict_logmel_wav(cnn_model, audio_path, CNN_MODEL_METADATA_PATH)
            st.markdown(f"**Experimental log-Mel CNN:** {DISPLAY_LABELS[label]}")
            _render_probability_chart(probabilities)
        except ImportError:
            st.info("The CNN artifact is available, but comparison requires launching Streamlit from .venv-cnn.")
        except (FileNotFoundError, TypeError, ValueError, OSError) as error:
            st.error(f"Experimental CNN comparison could not be completed: {error}")


def render_app() -> None:
    """Render the Streamlit application without retraining or altering artifacts."""
    st.set_page_config(page_title="Speech Emotion Baseline", page_icon="🎙️", layout="wide")
    _render_header()
    uploaded_audio = st.file_uploader("Upload a WAV recording", type=["wav"], help="Only valid WAV audio is supported.")
    if uploaded_audio is None:
        st.info("Upload a WAV recording to view its audio analysis and prediction.")
        return

    temporary_path = _persist_uploaded_wav(uploaded_audio)
    try:
        try:
            audio, sample_rate, channels, duration = _audio_details(temporary_path)
        except (FileNotFoundError, TypeError, ValueError, OSError) as error:
            st.error(f"The uploaded file is not a valid WAV recording: {error}")
            return

        st.audio(uploaded_audio.getvalue(), format="audio/wav")
        details = st.columns(3)
        details[0].metric("Duration", f"{duration:.2f} s")
        details[1].metric("Sample rate", f"{sample_rate:,} Hz")
        details[2].metric("Channels", "Mono" if channels == 1 else f"Stereo ({channels})")

        left_column, right_column = st.columns(2)
        with left_column:
            st.subheader("Waveform")
            st.pyplot(plot_waveform(audio, sample_rate), use_container_width=True)
        with right_column:
            st.subheader("Spectrogram")
            st.pyplot(plot_spectrogram(audio, sample_rate), use_container_width=True)
        _render_prediction(temporary_path)
        evaluate_external = st.checkbox(
            "Compare experimental models for external synthetic audio",
            help="Use only for an external synthetic recording; this never trains, saves, or adds the upload to RAVDESS.",
        )
        if evaluate_external:
            _render_external_synthetic_comparison(temporary_path)
    except (TypeError, ValueError, OSError) as error:
        st.error(f"Audio analysis could not be completed: {error}")
    finally:
        temporary_path.unlink(missing_ok=True)
