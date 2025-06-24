# YouTube Virality Predictor – Web Service (FastAPI)

This is the backend API for the YouTube Virality Predictor. It takes a thumbnail image, video title, description, and tags as input and returns a prediction on whether the video is likely to go viral. The model is trained on real YouTube data using Random Forest with SMOTE to handle class imbalance.

## 🚀 Features

- Accepts multipart form data (image + text fields)
- Extracts visual and textual features (color, sentiment, clickbait score, etc.)
- Loads a trained Random Forest pipeline (`rf_pipeline.pkl`)
- Returns classification result (viral / not viral)

## 🧠 Model Info

- SMOTE + Random Forest classifier
- Features: title sentiment, readability, edge density of thumbnail, clickbait score, tag sentiment, etc.
- Threshold-tuned for higher recall to reduce missed opportunities

## 🛠️ Installation

```bash
pip install -r requirements.txt
