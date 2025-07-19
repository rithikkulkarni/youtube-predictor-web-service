# app.py — updated with bounded cache and mmap loading

import os
import gc  # Optional: only needed if you call gc.collect() after cache_clear()
import joblib
import numpy as np
from io import BytesIO
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import cv2, string
from textblob import TextBlob
import textstat
from typing import List
from pytrends.request import TrendReq
import requests
from bs4 import BeautifulSoup
from functools import lru_cache

from model import FEATURE_ORDER

# ─── Full clickbait / power / timed word lists ────────────────────────────────
CLICKBAIT_WORDS = {
    "amazing", "incredible", "shocking", "jaw-dropping", "mind-blowing",
    "unbelievable", "you won’t believe", "you’ll never guess", "what happens next",
    "epic", "ultimate", "must", "insane", "secret", "exposed", "revealed",
    "hack", "these reasons", "10 reasons", "this trick", "don’t miss",
    "game changer", "craziest", "revealed", "the truth about", "deal of the day",
}

POWER_WORDS = {
    "best", "top", "new", "essential", "easy", "quick", "instant", "effortless",
    "guaranteed", "proven", "genius", "exclusive", "remarkable", "powerful",
    "revolutionary", "breakthrough", "must-have", "unlock", "master", "ultimate",
    "secret", "simple", "transform", "hacks", "tips", "tricks",
}

TIMED_WORDS = {
    "now", "today", "just now", "breaking", "this morning", "this afternoon",
    "this evening", "tonight", "this week", "this weekend", "this month",
    "this season", "this year", "last minute", "last week", "2023", "2024",
    "2025", "coming soon", "newly released", "upcoming", "recent", "daily",
    "weekly", "monthly", "yearly",
}

# ─── Face detector ────────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ─── FastAPI app setup ────────────────────────────────────────────────────────
app = FastAPI(
    title="YouTube Virality Predictor",
    description="Upload a thumbnail, title & tags and subscriber count to predict virality.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your domain(s)
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# serve your static landing page / JS / CSS
@app.get("/", include_in_schema=False)
def serve_index():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")

# ─── Lazy model loader with bounded cache ─────────────────────────────────────
MODEL_DIR = "models"


@lru_cache(maxsize=2)  # keep at most two models resident; LRU eviction frees RAM
def get_pipeline_for(group: str):
    """Load and cache the Random‑Forest pipeline for the given subscriber group."""
    path = os.path.join(MODEL_DIR, f"{group}_rf.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No pipeline for group '{group}'")
    # memory‑map large NumPy arrays instead of loading them eagerly into RAM
    return joblib.load(path, mmap_mode="r")


# ─── Utility feature functions ────────────────────────────────────────────────

def compute_clickbait_score(text: str) -> int:
    return sum(w.lower() in CLICKBAIT_WORDS for w in text.split())


def compute_dominant_color_hue(img: np.ndarray) -> float:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    return float(np.argmax(hist))


def compute_title_features(title: str) -> dict:
    blob = TextBlob(title)
    words = title.split()
    punctuation = set("!?.,:;-()[]{}")
    upper_words = [w for w in words if w.isupper()]
    letters = [c for c in title if c.isalpha()]
    uppercase_letters = [c for c in letters if c.isupper()]

    return {
        "title_sentiment": blob.sentiment.polarity,
        "title_subjectivity": blob.sentiment.subjectivity,
        "num_question_marks": title.count("?"),
        "num_exclamation_marks": title.count("!"),
        "starts_with_keyword": int(words[0].lower() in {"how", "what", "why", "when", "is", "are", "does", "who"} if words else 0),
        "title_length": len(title),
        "word_count": len(words),
        "punctuation_count": sum(1 for c in title if c in punctuation),
        "uppercase_word_count": len(upper_words),
        "percent_letters_uppercase": round(len(uppercase_letters) / len(letters), 3) if letters else 0,
        "num_digits": sum(c.isdigit() for c in title),
        "clickbait_score": compute_clickbait_score(title),
        "num_power_words": sum(w.lower() in POWER_WORDS for w in words),
        "num_timed_words": sum(phrase in title.lower() for phrase in TIMED_WORDS),
        "clickbait_phrase_match": int(any(p in title.lower() for p in CLICKBAIT_WORDS)),
        "title_readability": textstat.flesch_reading_ease(title),
        "is_listicle": int(title.strip().split()[0].isdigit()),
        "is_tutorial": int(title.lower().startswith("how to")),
        "power_word_count": sum(w.lower() in POWER_WORDS for w in words),
        "timed_word_count": sum(phrase in title.lower() for phrase in TIMED_WORDS),
    }


def get_google_trend_score(keywords: List[str]) -> float:
    try:
        pytrends = TrendReq()
        pytrends.build_payload(keywords, timeframe="now 1-d")
        data = pytrends.interest_over_time()
        if data.empty:
            return 0.0
        return float(data[keywords].iloc[-1].mean() / 100)
    except Exception:
        return 0.0


def get_twitter_trend_score(keywords: List[str]) -> float:
    try:
        url = "https://twitter.com/explore/tabs/trending"
        resp = requests.get(url, timeout=5)
        soup = BeautifulSoup(resp.text, "html.parser")
        hashtags = [tag.get_text().lstrip("#") for tag in soup.find_all("span")]
        matches = sum(
            1 for kw in keywords if any(kw.lower() in h.lower() for h in hashtags)
        )
        return matches / max(len(hashtags), 1)
    except Exception:
        return 0.0


def calculate_trending_score(keywords: List[str], w1: float = 0.5, w2: float = 0.5) -> float:
    g = get_google_trend_score(keywords)
    t = get_twitter_trend_score(keywords)
    return round(w1 * g + w2 * t, 4)


# ─── Main endpoint ────────────────────────────────────────────────────────────

@app.post("/predict/{group}")
async def predict_group(
    group: str,
    title: str = Form(...),
    tags: str = Form(...),
    thumbnail: UploadFile = File(...),
):
    try:
        pipeline = get_pipeline_for(group)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Unknown group '{group}'.")

    img_bytes = await thumbnail.read()
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    avg_red, avg_green, avg_blue = img.mean(axis=(0, 1))
    brightness = float(gray.mean())
    contrast = float(gray.std())

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    num_faces = int(len(faces))

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float((edges > 0).mean())
    dominant_hue = compute_dominant_color_hue(img)

    tfeats = compute_title_features(title)
    tlist = [t.strip() for t in tags.split(",") if t.strip()]
    num_tags = len(tlist)
    tag_sentiment = float(np.mean([TextBlob(t).sentiment.polarity for t in tlist])) if tlist else 0.0
    num_unique_tags = int(len(set(t.lower() for t in tlist)))
    avg_tag_length = float(np.mean([len(t) for t in tlist])) if tlist else 0.0

    trending_score = calculate_trending_score([title])

    fv = {
        "avg_red": float(avg_red),
        "avg_green": float(avg_green),
        "avg_blue": float(avg_blue),
        "brightness": brightness,
        "contrast": contrast,
        "num_faces": num_faces,
        **tfeats,
        "num_tags": num_tags,
        "tag_sentiment": tag_sentiment,
        "num_unique_tags": num_unique_tags,
        "avg_tag_length": avg_tag_length,
        "thumbnail_edge_density": edge_density,
        "dominant_color_hue": dominant_hue,
        "trending_score": trending_score,
    }

    x_vec = np.array([[fv[f] for f in FEATURE_ORDER]])
    proba = pipeline.predict_proba(x_vec)[0][1]
    pred = int(proba >= 0.3)

    return {"group": group, "prediction": pred, "probability": round(proba, 4)}
