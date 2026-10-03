# Fruit & Vegetable Classifier (Streamlit)

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` – the app (model definition copied from the notebook)
- `fruit_classification.pth` – trained weights (state_dict)
- `class_names.json` – 88 class names in `ImageFolder` order
- `.streamlit/config.toml` – base Streamlit config

## Features
Single-image classify (upload or camera), batch mode with CSV export,
session history, top-K bars, confidence threshold, duplicate-label merging,
and 4 switchable themes (Light, Dark, Ocean, Forest) from the sidebar.
