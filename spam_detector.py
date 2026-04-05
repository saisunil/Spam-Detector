"""
Spam Email Detection — Core ML Module
======================================
Handles data loading, preprocessing, feature extraction,
model training, evaluation, and prediction.
"""

import re
import string
import pickle
import os

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------------------------------------------------------------------------
# NLTK data download (one-time)
# ---------------------------------------------------------------------------
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

STEMMER = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))


# ---------------------------------------------------------------------------
# Text Preprocessing
# ---------------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    """Clean and normalize a single message string."""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Tokenize, remove stopwords, and stem
    tokens = text.split()
    tokens = [STEMMER.stem(w) for w in tokens if w not in STOP_WORDS]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_data(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load the CSV dataset and return a DataFrame."""
    df = pd.read_csv(path, encoding="utf-8")
    df.columns = [c.strip().lower() for c in df.columns]
    # Map labels to binary
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    df["clean_message"] = df["message"].apply(preprocess_text)
    return df


# ---------------------------------------------------------------------------
# Model Training & Evaluation
# ---------------------------------------------------------------------------
def train_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Train a Naive Bayes classifier on TF-IDF features.

    Returns
    -------
    model, vectorizer, metrics_dict
    """
    X = df["clean_message"]
    y = df["label_num"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Multinomial Naive Bayes
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_tfidf, y_train)

    # Predictions
    y_pred = model.predict(X_test_tfidf)

    # Metrics
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, target_names=["Ham", "Spam"]),
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "spam_count": int(y.sum()),
        "ham_count": int((y == 0).sum()),
    }

    return model, vectorizer, metrics


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_model(model, vectorizer):
    """Serialize model and vectorizer to disk."""
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)


def load_model():
    """Deserialize model and vectorizer from disk."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_message(message: str, model=None, vectorizer=None):
    """
    Predict whether a message is spam or ham.

    Returns
    -------
    dict with keys: label, confidence, probabilities
    """
    if model is None or vectorizer is None:
        model, vectorizer = load_model()

    clean = preprocess_text(message)
    tfidf = vectorizer.transform([clean])

    prediction = model.predict(tfidf)[0]
    probabilities = model.predict_proba(tfidf)[0]

    label = "spam" if prediction == 1 else "ham"
    confidence = float(max(probabilities)) * 100

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "spam_probability": round(float(probabilities[1]) * 100, 2),
        "ham_probability": round(float(probabilities[0]) * 100, 2),
    }
