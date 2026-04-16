"""
run.py — entry point for fake news ensemble detector.

Usage:
  python run.py                # train + evaluate + interactive predict
  python run.py --no-tune      # skip Optuna (faster, for testing)
  python run.py --cv           # run 5-fold CV report
"""

import argparse
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()

from src.data_loader import load_all
from src.pipeline import (
    train_model,
    evaluate_model,
    predict_news,
    cross_validate_model,
)
from src.explainer import get_top_features, explain_prediction

MODEL_PATH      = Path("model.pkl")
VECTORIZER_PATH = Path("vectorizer.pkl")
DATA_DIR        = Path("data/raw")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip Optuna hyperparameter tuning")
    parser.add_argument("--cv", action="store_true",
                        help="Run cross-validation report")
    parser.add_argument("--trials", type=int, default=20,
                        help="Optuna trial count (default: 20)")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    data = load_all(DATA_DIR, use_liar=True, use_isot=True)

    X = data["text"]
    y = data["label"]

    # ── Optional CV ────────────────────────────────────────────────────────────
    if args.cv:
        print("\n=== 5-Fold Cross-Validation ===")
        cross_validate_model(X, y, n_splits=5)
        return

    # ── Train/test split ───────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
        print("Loading saved model …")
        model      = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    else:
        print("Training ensemble model …")
        model, vectorizer = train_model(
            X_train, y_train,
            tune=not args.no_tune,
            n_trials=args.trials,
        )
        joblib.dump(model,      MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        print("Model saved.")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    accuracy, report, cm = evaluate_model(model, vectorizer, X_test, y_test)
    # Per-model breakdown
    from src.pipeline import evaluate_individual_models
    evaluate_individual_models(model, vectorizer, X_test, y_test)
    print(f"\n{'='*50}")
    print(f"Test Accuracy : {accuracy:.4f}")
    print(f"\nClassification Report:\n{report}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"{'='*50}\n")

    # ── Interactive predict ────────────────────────────────────────────────────
    while True:
        news = input("Enter a news headline (or 'q' to quit): ").strip()
        if news.lower() in ("q", "quit", "exit"):
            break
        if not news:
            continue

        label, prediction, proba, feat = predict_news(news, model, vectorizer)
        print(f"\n→ Prediction : {label}")
        print(f"  Confidence : Fake={proba[0]*100:.1f}%  Real={proba[1]*100:.1f}%")

        if input("Explain? [y/n]: ").strip().lower() == "y":
            fake_f, real_f = get_top_features(feat, model, vectorizer)
            explanation = explain_prediction(news, prediction, proba, fake_f, real_f)
            print(f"\n{explanation}\n")


if __name__ == "__main__":
    main()
