# model.py
import pandas as pd
import numpy as np
import joblib
from imblearn.pipeline import Pipeline  # ensure using imblearn's Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Define feature order (same as before)
FEATURE_ORDER = [
    'avg_red', 'avg_green', 'avg_blue', 'brightness', 'contrast',
    'title_sentiment', 'title_subjectivity', 'has_question',
    'has_exclamation', 'starts_with_keyword', 'title_length', 'word_count',
    'punctuation_count', 'uppercase_word_count',
    'percent_letters_uppercase', 'has_numbers', 'clickbait_score',
    'description_length', 'description_sentiment',
    'description_has_keywords', 'tag_count', 'tag_sentiment',
    'title_readability', 'dominant_color_hue', 'thumbnail_edge_density'
]

# Path to save the pipeline
RF_PIPELINE_PATH = 'rf_pipeline.pkl'


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[
        'video_id', 'title', 'channel_id',
        'viewCount', 'likeCount', 'commentCount',
        'viral', 'description', 'tags', 'title_embedding',
        'log_viewCount', 'log_likeCount', 'log_commentCount',
        'embedding_distance_to_known_viral', 'title_embedding_distance_to_viral',
        'num_unique_tags'
    ])
    y = df['viral']
    return X, y


def build_pipeline():
    # Build an imblearn Pipeline with SMOTE
    return Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])


def train_and_save(csv_path: str):
    X, y = load_data(csv_path)
    rf_pipeline = build_pipeline()
    print("Training Random Forest pipeline...")
    rf_pipeline.fit(X, y)
    joblib.dump(rf_pipeline, RF_PIPELINE_PATH)
    print(f"Saved RF pipeline to {RF_PIPELINE_PATH}")


def predict(features: dict, threshold: float = 0.3):
    # Build feature vector in correct order
    feature_vec = np.array([[features[f] for f in FEATURE_ORDER]])
    pipeline = joblib.load(RF_PIPELINE_PATH)
    proba = pipeline.predict_proba(feature_vec)[0][1]
    pred = int(proba >= threshold)
    return {'prediction': pred, 'probability': proba}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train and save RF pipeline')
    parser.add_argument('--data', type=str, default='video_details_v8.csv')
    args = parser.parse_args()
    train_and_save(args.data)
