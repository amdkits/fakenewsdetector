from pathlib import Path
from sklearn.model_selection import train_test_split

from src.pipeline import (
    load_data,
    preprocess_data,
    train_model,
    evaluate_model,
    predict_news,
)


def main():
    data_dir = Path("data/raw")

    data = load_data(data_dir)
    data = preprocess_data(data)

    X = data["text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model, vectorizer = train_model(X_train, y_train)

    accuracy, report, cm = evaluate_model(model, vectorizer, X_test, y_test)

    print("Accuracy:", accuracy)
    print("\nClassification Report:\n", report)

    news = input("\nEnter a news headline to test: ")
    print("Prediction:", predict_news(news, model, vectorizer))


if __name__ == "__main__":
    main()
