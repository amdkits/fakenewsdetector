from pathlib import Path
from sklearn.model_selection import train_test_split

from src.pipeline import (
    load_data,
    preprocess_data,
    train_model,
    evaluate_model,
    predict_news,
)
import joblib

MODEL_PATH = Path("model.pkl")
VECTORIZER_PATH = Path("vectorizer.pkl")

def main():
    data_dir = Path("data/raw")

    if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
        print("Loading saved model...")
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        # still need data for evaluate, skip if you don't care about the report
        data = preprocess_data(load_data(data_dir))
        _, X_test, _, y_test = train_test_split(
            data["text"], data["label"], test_size=0.25, random_state=42
        )
    else:
        print("Training model for the first time...")
        data = preprocess_data(load_data(data_dir))
        X_train, X_test, y_train, y_test = train_test_split(
            data["text"], data["label"], test_size=0.25, random_state=42
        )
        model, vectorizer = train_model(X_train, y_train)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        print("Model saved.")

    accuracy, report, cm = evaluate_model(model, vectorizer, X_test, y_test)

    print("Accuracy:", accuracy)
    print("\nClassification Report:\n", report)

    news = input("\nEnter a news headline to test: ")
    print("Prediction:", predict_news(news, model, vectorizer))


if __name__ == "__main__":
    main()
