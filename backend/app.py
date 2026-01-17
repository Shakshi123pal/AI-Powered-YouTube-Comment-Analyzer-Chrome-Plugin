import os
import io
import re
import pickle
import logging
import requests
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud


# NLTK SAFE SETUP

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

STOP_WORDS = set(stopwords.words("english")) - {
    "not", "but", "no", "however", "yet"
}
lemmatizer = WordNetLemmatizer()

# FASTAPI APP + CORS

app = FastAPI(title="YouTube Comment Sentiment Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev + interview
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LOGGING

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ENV VARIABLES

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    raise RuntimeError("YOUTUBE_API_KEY environment variable not set")


# MODEL LOADING (SAFE PATH)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "svm_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}")

if not os.path.exists(VECTORIZER_PATH):
    raise RuntimeError(f"Vectorizer not found at {VECTORIZER_PATH}")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

logger.info("✅ Model and vectorizer loaded successfully")


# TEXT PREPROCESSING

def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-z0-9\s!?.,]", "", text)

    tokens = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in STOP_WORDS
    ]
    return " ".join(tokens)


# SCHEMAS

class PredictRequest(BaseModel):
    comments: list[str]

class AnalyzeRequest(BaseModel):
    video_id: str

class TrendRequest(BaseModel):
    sentiment_data: list[dict]

class WordCloudRequest(BaseModel):
    comments: list[str]


# HEALTH CHECK

@app.get("/")
def health():
    return {"status": "Backend running successfully 🚀"}


# ANALYZE VIDEO (USED BY CHROME EXTENSION)

@app.post("/analyze")
def analyze_video(req: AnalyzeRequest):
    video_id = req.video_id
    url = "https://www.googleapis.com/youtube/v3/commentThreads"

    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": 100,
        "key": YOUTUBE_API_KEY
    }

    comments = []
    page_token = None

    try:
        while len(comments) < 100:
            if page_token:
                params["pageToken"] = page_token

            res = requests.get(url, params=params)
            data = res.json()

            for item in data.get("items", []):
                comments.append(
                    item["snippet"]["topLevelComment"]["snippet"]["textOriginal"]
                )

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        if not comments:
            raise HTTPException(status_code=404, detail="No comments found")

        processed = [preprocess(c) for c in comments]
        X = vectorizer.transform(processed)
        preds = model.predict(X)

        summary = {
            "positive": int((preds == 1).sum()),
            "neutral": int((preds == 0).sum()),
            "negative": int((preds == 2).sum()),
        }

        return {
            "video_id": video_id,
            "total_comments": len(comments),
            "summary": summary
        }

    except Exception as e:
        logger.error(f"Analyze failed: {e}")
        raise HTTPException(status_code=500, detail="Video analysis failed")

# PREDICT (OPTIONAL / TESTING)

@app.post("/predict")
def predict(req: PredictRequest):
    if not req.comments:
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        processed = [preprocess(c) for c in req.comments]
        X = vectorizer.transform(processed)
        preds = model.predict(X)

        return [
            {"comment": c, "sentiment": int(p)}
            for c, p in zip(req.comments, preds)
        ]

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Sentiment prediction failed")

# PIE CHART

@app.post("/generate_chart")
def generate_chart(data: dict):
    sentiment = data.get("sentiment_counts", {})

    sizes = [
        sentiment.get("1", 0),
        sentiment.get("0", 0),
        sentiment.get("2", 0)
    ]

    labels = ["Positive", "Neutral", "Negative"]
    colors = ["#36A2EB", "#C9CBCF", "#FF6384"]

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%")
    plt.axis("equal")

    buf = io.BytesIO()
    plt.savefig(buf, format="PNG")
    buf.seek(0)
    plt.close()

    return StreamingResponse(buf, media_type="image/png")


# WORD CLOUD

@app.post("/generate_wordcloud")
def generate_wordcloud(req: WordCloudRequest):
    text = " ".join(preprocess(c) for c in req.comments)

    wc = WordCloud(
        width=800,
        height=400,
        background_color="black",
        stopwords=STOP_WORDS
    ).generate(text)

    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

# TREND GRAPH

@app.post("/generate_trend_graph")
def generate_trend_graph(req: TrendRequest):
    df = pd.DataFrame(req.sentiment_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df["sentiment"] = df["sentiment"].astype(int)

    monthly = df.resample("M")["sentiment"].value_counts().unstack(fill_value=0)

    for col in [0, 1, 2]:
        if col not in monthly.columns:
            monthly[col] = 0

    monthly_pct = (monthly.T / monthly.sum(axis=1)).T * 100

    plt.figure(figsize=(12, 6))
    plt.plot(monthly_pct.index, monthly_pct[1], label="Positive", color="green")
    plt.plot(monthly_pct.index, monthly_pct[0], label="Neutral", color="gray")
    plt.plot(monthly_pct.index, monthly_pct[2], label="Negative", color="red")

    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format="PNG")
    buf.seek(0)
    plt.close()

    return StreamingResponse(buf, media_type="image/png")
