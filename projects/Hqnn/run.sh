#!/usr/bin/env bash
# Launch the Radar QML Classifier app
cd "$(dirname "$0")"
.venv/bin/streamlit run app.py --server.port 8501 --server.headless true
