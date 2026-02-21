# STEP 1: Import Required Libraries
import pandas as pd
import numpy as np
import string
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# STEP 2: Load Dataset
fake_data = pd.read_csv("../data/raw/Fake.csv")
true_data = pd.read_csv("../data/raw/True.csv")

# STEP 3: Add Labels
fake_data["label"] = 0  # Fake = 0
true_data["label"] = 1  # Real = 1

# STEP 4: Combine Dataset
data = pd.concat([fake_data, true_data])
data = data.sample(frac=1).reset_index(drop=True)  # Shuffle data

# STEP 5: Keep Only Required Columns
data = data[["text", "label"]]

# STEP 6: Clean Text Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

data["text"] = data["text"].apply(clean_text)

# STEP 7: Split Data
X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# STEP 8: Convert Text to Numerical Form (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# STEP 9: Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# STEP 10: Make Predictions
y_pred = model.predict(X_test)

# STEP 11: Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# STEP 12: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# STEP 13: Test with Custom Input
def predict_news(news):
    news = clean_text(news)
    vector = vectorizer.transform([news])
    prediction = model.predict(vector)
    if prediction == 0:
        return "Fake News"
    else:
        return "Real News"

# Example
sample_news = input("\nEnter a news headline to test: ")
print("Prediction:", predict_news(sample_news))
