import os
import io
import re
import pickle
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

# -----------------------------
# FASTAPI APP + CORS ENABLED
# -----------------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# LOAD MODEL + TFIDF FROM DAGSHUB (LOCAL CACHE BY DVC)
# -----------------------------

with open(os.path.join(ROOT, "../svm_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(ROOT, "../tfidf_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

# -----------------------------
# PREPROCESSING 
# -----------------------------

def preprocess_comment(comment):
    try:
        comment = comment.lower().strip()
        comment = re.sub(r"\n", " ", comment)
        comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)

        stop_words = set(stopwords.words("english")) - {
            "not", "but", "however", "no", "yet"
        }
        comment = " ".join([w for w in comment.split() if w not in stop_words])

        lemmatizer = WordNetLemmatizer()
        comment = " ".join([lemmatizer.lemmatize(w) for w in comment.split()])

        return comment
    except:
        return comment


# -----------------------------
# PYDANTIC MODELS
# -----------------------------

class PredictRequest(BaseModel):
    comments: list


class PredictTimestampRequest(BaseModel):
    comments: list


class TrendRequest(BaseModel):
    sentiment_data: list


class WordCloudRequest(BaseModel):
    comments: list


# -----------------------------
# HOME ROUTE
# -----------------------------
@app.get("/")
def home():
    return {"message": "FastAPI Backend Running Successfully 🚀"}


# -----------------------------
# /PREDICT
# -----------------------------
@app.post("/predict")
def predict(data: PredictRequest):
    comments = data.comments

    if not comments:
        raise HTTPException(status_code=400, detail="No comments provided")

    processed = [preprocess_comment(c) for c in comments]
    transformed = vectorizer.transform(processed)
    preds = model.predict(transformed).tolist()

    response = [
        {"comment": c, "sentiment": int(s)}
        for c, s in zip(comments, preds)
    ]
    return response


# -----------------------------
# /PREDICT_WITH_TIMESTAMPS
# -----------------------------
@app.post("/predict_with_timestamps")
def predict_with_timestamps(data: PredictTimestampRequest):
    items = data.comments

    if not items:
        raise HTTPException(status_code=400, detail="No comments provided")

    comments = [item["text"] for item in items]
    timestamps = [item["timestamp"] for item in items]

    processed = [preprocess_comment(c) for c in comments]
    transformed = vectorizer.transform(processed)
    preds = model.predict(transformed).tolist()

    response = [
        {
            "comment": c,
            "sentiment": int(s),
            "timestamp": t
        }
        for c, s, t in zip(comments, preds, timestamps)
    ]
    return response


# -----------------------------
# /GENERATE_CHART
# -----------------------------
@app.post("/generate_chart")
def generate_chart(data: dict):
    sentiment = data.get("sentiment_counts")

    if not sentiment:
        raise HTTPException(status_code=400, detail="No sentiment counts provided")

    labels = ["Positive", "Neutral", "Negative"]
    sizes = [
        sentiment.get("1", 0),
        sentiment.get("0", 0),
        sentiment.get("2", 0)
    ]

    colors = ["#36A2EB", "#C9CBCF", "#FF6384"]

    plt.figure(figsize=(6, 6))
    plt.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"color": "white"}
    )
    plt.axis("equal")

    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format="PNG", transparent=True)
    img_bytes.seek(0)
    plt.close()

    return StreamingResponse(img_bytes, media_type="image/png")


# -----------------------------
# /GENERATE_WORDCLOUD
# -----------------------------
@app.post("/generate_wordcloud")
def generate_wordcloud(data: WordCloudRequest):
    comments = data.comments

    text = " ".join([preprocess_comment(c) for c in comments])

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="black",
        colormap="Blues",
        stopwords=set(stopwords.words("english")),
        collocations=False
    ).generate(text)

    img_bytes = io.BytesIO()
    wordcloud.to_image().save(img_bytes, format="PNG")
    img_bytes.seek(0)

    return StreamingResponse(img_bytes, media_type="image/png")


# -----------------------------
# /GENERATE_TREND_GRAPH
# -----------------------------
@app.post("/generate_trend_graph")
def generate_trend_graph(data: TrendRequest):
    df = pd.DataFrame(data.sentiment_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    df["sentiment"] = df["sentiment"].astype(int)

    monthly = df.resample("M")["sentiment"].value_counts().unstack(fill_value=0)
    totals = monthly.sum(axis=1)
    monthly_pct = (monthly.T / totals).T * 100

    for s in [0, 1, 2]:
        if s not in monthly_pct.columns:
            monthly_pct[s] = 0

    monthly_pct = monthly_pct[[0, 1, 2]]

    plt.figure(figsize=(12, 6))
    colors = {0: "gray", 1: "green", 2: "red"}
    labels = {0: "Neutral", 1: "Positive", 2: "Negative"}

    for s in [0, 1, 2]:
        plt.plot(
            monthly_pct.index,
            monthly_pct[s],
            marker="o",
            linestyle="-",
            color=colors[s],
            label=labels[s]
        )

    plt.title("Monthly Sentiment Trend (%)")
    plt.xlabel("Month")
    plt.ylabel("Percentage")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format="PNG")
    img_bytes.seek(0)
    plt.close()

    return StreamingResponse(img_bytes, media_type="image/png")
