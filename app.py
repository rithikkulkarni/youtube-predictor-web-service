from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
import string
from textblob import TextBlob
import textstat

from model import predict, FEATURE_ORDER

# Placeholder lists and functions; implement as in your feature pipeline
KEYWORD_LIST = ["how", "what", "why", "when", "is", "are", "does"]

app = FastAPI(
    title="YouTube Virality Predictor",
    description="Upload a thumbnail image and enter a video title to predict virality.",
    version="1.0.0"
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["https://rithikkulkarni.github.io"],
  allow_methods=["POST","GET"],
  allow_headers=["*"],
)

def compute_clickbait_score(text: str) -> float:
    clickbait_words = {
        "amazing", "shocking", "unbelievable", "top", "ultimate", "must",
        "insane", "you won’t believe", "secret", "revealed", "hack"
    }
    words = text.split()
    clickbait_score = sum(word.lower() in clickbait_words for word in words)
    return clickbait_score

def compute_title_readability(text: str) -> float:
    return textstat.flesch_reading_ease(text)

def compute_dominant_color_hue(img: np.ndarray) -> float:
    # Convert to HSV and compute the hue histogram peak
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    return float(np.argmax(hist))

@app.get("/", include_in_schema=False)
def serve_index():
    return FileResponse("static/index.html")

# app.mount("/", StaticFiles(directory="static", html=True), name="static")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/extract_and_predict")
async def extract_and_predict(
    title: str = Form(...),
    description: str = Form(...),
    tags: str = Form(...),
    thumbnail: UploadFile = File(...)
):
    # 1. Read and preprocess image
    img_bytes = await thumbnail.read()
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img = np.array(pil_img)

    # 2a. Image features
    avg_red = float(img[:,:,0].mean())
    avg_green = float(img[:,:,1].mean())
    avg_blue = float(img[:,:,2].mean())
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    edges = cv2.Canny(gray, 100, 200)
    thumbnail_edge_density = float(edges.astype(bool).mean())

    # 2b. Dominant color hue
    dominant_color_hue = compute_dominant_color_hue(img)

    # 3. Title features
    title_blob = TextBlob(title)
    title_sentiment = float(title_blob.sentiment.polarity)
    title_subjectivity = float(title_blob.sentiment.subjectivity)
    has_question = int("?" in title)
    has_exclamation = int("!" in title)
    starts_with_keyword = int(title.lower().split()[0] in KEYWORD_LIST)
    title_length = int(len(title))
    word_count = int(len(title.split()))
    punctuation_count = int(sum(c in string.punctuation for c in title))
    uppercase_word_count = int(sum(1 for w in title.split() if w.isupper()))
    letters = [c for c in title if c.isalpha()]
    percent_letters_uppercase = float(sum(1 for c in letters if c.isupper()) / max(1, len(letters)))
    has_numbers = int(any(c.isdigit() for c in title))
    clickbait_score = float(compute_clickbait_score(title))
    title_readability = float(compute_title_readability(title))

    # 4. Description features
    desc_blob = TextBlob(description)
    description_length = int(len(description))
    description_sentiment = float(desc_blob.sentiment.polarity)
    description_has_keywords = int(any(k in description.lower() for k in KEYWORD_LIST))

    # 5. Tag features
    tags_list = [t.strip() for t in tags.split(",") if t.strip()]
    tag_count = int(len(tags_list))
    tag_sentiment = float(np.mean([TextBlob(tag).sentiment.polarity for tag in tags_list])) if tags_list else 0.0

    # 6. Assemble feature dict
    feature_values = {
        'avg_red': avg_red,
        'avg_green': avg_green,
        'avg_blue': avg_blue,
        'brightness': brightness,
        'contrast': contrast,
        'title_sentiment': title_sentiment,
        'title_subjectivity': title_subjectivity,
        'has_question': has_question,
        'has_exclamation': has_exclamation,
        'starts_with_keyword': starts_with_keyword,
        'title_length': title_length,
        'word_count': word_count,
        'punctuation_count': punctuation_count,
        'uppercase_word_count': uppercase_word_count,
        'percent_letters_uppercase': percent_letters_uppercase,
        'has_numbers': has_numbers,
        'clickbait_score': clickbait_score,
        'description_length': description_length,
        'description_sentiment': description_sentiment,
        'description_has_keywords': description_has_keywords,
        'tag_count': tag_count,
        'tag_sentiment': tag_sentiment,
        'title_readability': title_readability,
        'dominant_color_hue': dominant_color_hue,
        'thumbnail_edge_density': thumbnail_edge_density
    }

    # 7. Predict and return
    result = predict(feature_values, threshold=0.3)
    return result
