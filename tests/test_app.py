"""Tests for the Streamlit entry point and canonical UI contracts."""

import app

from src.ui import streamlit_app


def test_main_delegates_to_streamlit_renderer(monkeypatch) -> None:
    """The entry point contains no application logic beyond UI delegation."""
    calls: list[bool] = []
    monkeypatch.setattr(app, "render_app", lambda: calls.append(True))

    app.main()

    assert calls == [True]


def test_probability_chart_uses_canonical_display_order() -> None:
    """Chart and table share Neutral, Calm, Happy, Sad, Angry ordering."""
    probabilities = {"angry": 0.5, "neutral": 0.1, "sad": 0.15, "happy": 0.2, "calm": 0.05}

    frame = streamlit_app.build_probability_frame(probabilities)

    assert list(frame.index) == ["Neutral", "Calm", "Happy", "Sad", "Angry"]
    assert list(frame.columns) == ["Predicted probability"]
    assert frame["Predicted probability"].tolist() == [0.1, 0.05, 0.2, 0.15, 0.5]


def test_prediction_badge_uses_predicted_probability_terminology(monkeypatch, tmp_path) -> None:
    """The prediction badge avoids presenting model output as calibrated confidence."""
    metric_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(streamlit_app, "load_prediction_model", lambda: object())
    monkeypatch.setattr(
        streamlit_app,
        "predict_wav",
        lambda model, path, metadata_path: ("happy", {"neutral": 0.1, "calm": 0.1, "happy": 0.5, "sad": 0.1, "angry": 0.2}),
    )
    monkeypatch.setattr(streamlit_app.st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "write", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "metric", lambda *args, **kwargs: metric_calls.append(args))
    monkeypatch.setattr(streamlit_app, "_render_probability_chart", lambda probabilities: None)

    streamlit_app._render_prediction(tmp_path / "audio.wav")

    assert metric_calls[0] == (
        "Predicted vocal-expression category",
        "Happy",
        "50.0% predicted probability",
    )


def test_external_synthetic_comparison_uses_saved_svm_prediction_contract(monkeypatch, tmp_path) -> None:
    """The opt-in comparison remains external-only and uses canonical probabilities."""
    messages: list[str] = []
    svm_model_path = tmp_path / "svm.joblib"
    svm_metadata_path = tmp_path / "svm_metadata.json"
    svm_model_path.touch()
    svm_metadata_path.touch()
    monkeypatch.setattr(streamlit_app, "SVM_MODEL_PATH", svm_model_path)
    monkeypatch.setattr(streamlit_app, "SVM_MODEL_METADATA_PATH", svm_metadata_path)
    monkeypatch.setattr(streamlit_app, "load_experimental_svm_model", lambda: object())
    monkeypatch.setattr(
        streamlit_app,
        "predict_wav",
        lambda model, path, metadata_path: ("happy", {"neutral": 0.1, "calm": 0.1, "happy": 0.5, "sad": 0.1, "angry": 0.2}),
    )
    monkeypatch.setattr(streamlit_app.st, "subheader", lambda message: messages.append(message))
    monkeypatch.setattr(streamlit_app.st, "caption", lambda message: messages.append(message))
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda message: messages.append(message))
    monkeypatch.setattr(streamlit_app, "_render_probability_chart", lambda probabilities: None)
    monkeypatch.setattr(streamlit_app, "CNN_MODEL_PATH", tmp_path / "missing.keras")

    streamlit_app._render_external_synthetic_comparison(tmp_path / "external.wav")

    assert messages[0] == "External synthetic-audio evaluation"
    assert "not added to RAVDESS or used for training" in messages[1]
    assert messages[2] == "**Experimental RBF-SVM:** Happy"
