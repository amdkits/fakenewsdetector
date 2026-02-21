import pandas as pd
import string
import re
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ------------------------
# DATA LOADING
# ------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

def load_data(data_dir: Path):
    fake = pd.read_csv(data_dir / "Fake.csv")
    true = pd.read_csv(data_dir / "True.csv")

    fake["label"] = 0
    true["label"] = 1

    data = pd.concat([fake, true])
    data = data.sample(frac=1).reset_index(drop=True)

    return data[["text", "label"]]


# ------------------------
# TEXT CLEANING
# ------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


def preprocess_data(data):
    data["text"] = data["text"].apply(clean_text)
    return data


# ------------------------
# TRAINING
# ------------------------

def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer(max_features=10000,
                                 ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=300, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    return model, vectorizer


# ------------------------
# EVALUATION
# ------------------------

def evaluate_model(model, vectorizer, X_test, y_test):
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, report, cm


# ------------------------
# PREDICTION
# ------------------------

def predict_news(news, model, vectorizer):
    news = clean_text(news)
    vector = vectorizer.transform([news])
    proba = model.predict_proba(vector)[0]
    prediction = model.predict(vector)[0]

    print("Probabilities:", proba)

    return "Fake News" if prediction == 0 else "Real News"
