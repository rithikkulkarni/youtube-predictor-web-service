# model.py
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# === Feature Order (must match exactly in FastAPI and frontend) ===
FEATURE_ORDER = [
    'avg_red', 'avg_green', 'avg_blue', 'brightness', 'contrast',
    'title_sentiment', 'title_subjectivity', 'num_question_marks',
    'num_exclamation_marks', 'starts_with_keyword', 'title_length',
    'word_count', 'punctuation_count', 'uppercase_word_count',
    'percent_letters_uppercase', 'num_digits', 'clickbait_score',
    'num_power_words', 'num_timed_words',
    'tag_sentiment', 'num_unique_tags', 'clickbait_phrase_match',
    'title_readability', 'is_listicle', 'is_tutorial',
    'dominant_color_hue', 'thumbnail_edge_density', 'power_word_count',
    'timed_word_count', 'avg_tag_length', 'num_faces', 'num_tags'
]

RF_PIPELINE_PATH = "group6_rf.pkl"

def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['viral'])
    y = df['viral']
    return X, y

def build_pipeline():
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

def train_and_save(csv_path: str):
    X, y = load_data(csv_path)
    pipeline = build_pipeline()
    print("Training Random Forest pipeline...")
    pipeline.fit(X, y)
    joblib.dump(pipeline, RF_PIPELINE_PATH)
    print(f"Model saved to {RF_PIPELINE_PATH}")

def predict(features: dict, threshold: float = 0.3):
    model = joblib.load(RF_PIPELINE_PATH)
    x_vec = np.array([[features[f] for f in FEATURE_ORDER]])
    proba = model.predict_proba(x_vec)[0][1]
    pred = int(proba >= threshold)
    return {'prediction': pred, 'probability': round(proba, 4)}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to the training CSV file")
    args = parser.parse_args()
    train_and_save(args.data)
