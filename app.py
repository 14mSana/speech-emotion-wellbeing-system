"""Thin Streamlit entry point for the speech-emotion baseline interface."""

from src.ui.streamlit_app import render_app


def main() -> None:
    """Start the local Streamlit interface."""
    render_app()


if __name__ == "__main__":
    main()
