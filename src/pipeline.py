"""
Ensemble pipeline:
  TF-IDF (bigrams, sublinear) + handcrafted features
  → VotingClassifier(LR + RandomForest + XGBoost), soft voting
  → Optuna hyperparameter tuning (optional)
  → StratifiedKFold cross-validation
"""

import numpy as np
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from xgboost import XGBClassifier

from src.feature_engineering import (
    clean_text_for_tfidf,
    extract_handcrafted,
    combine_features,
)


# ── VECTORIZER ─────────────────────────────────────────────────────────────────

def build_vectorizer(max_features: int = 50000) -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=clean_text_for_tfidf,
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        analyzer="word",
    )


# ── FEATURE MATRIX ─────────────────────────────────────────────────────────────

def build_feature_matrix(texts, vectorizer, fit: bool = False):
    """Return combined TF-IDF + handcrafted sparse matrix."""
    if fit:
        tfidf = vectorizer.fit_transform(texts)
    else:
        tfidf = vectorizer.transform(texts)
    hc = extract_handcrafted(texts)
    return combine_features(tfidf, hc)


# ── ENSEMBLE ───────────────────────────────────────────────────────────────────

def build_ensemble(lr_C=1.0, rf_n=200, rf_depth=None,
                   xgb_lr=0.1, xgb_n=100, xgb_depth=6) -> VotingClassifier:
    lr = LogisticRegression(
        C=lr_C, max_iter=2000, class_weight="balanced",
        solver="lbfgs", n_jobs=-1
    )
    rf = RandomForestClassifier(
        n_estimators=rf_n, max_depth=rf_depth,
        class_weight="balanced", n_jobs=-1, random_state=42
    )
    xgb = XGBClassifier(
        learning_rate=xgb_lr, n_estimators=xgb_n,
        max_depth=xgb_depth, use_label_encoder=False,
        eval_metric="logloss", n_jobs=-1, random_state=42,
        verbosity=0
    )
    return VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("xgb", xgb)],
        voting="soft",
        weights=[2, 1, 2],   # LR + XGB weighted higher
    )


# ── OPTUNA TUNING ──────────────────────────────────────────────────────────────

def tune_hyperparameters(X_train, y_train, n_trials: int = 20) -> dict:
    """
    Tune LR-C and XGB params via Optuna (3-fold CV on training set).
    Returns best params dict.
    """
    print(f"[Optuna] Starting {n_trials} trials …")

    def objective(trial):
        lr_C   = trial.suggest_float("lr_C",   0.01, 10.0, log=True)
        xgb_lr = trial.suggest_float("xgb_lr", 0.01, 0.3,  log=True)
        xgb_n  = trial.suggest_int("xgb_n",   50,   200)
        xgb_d  = trial.suggest_int("xgb_d",   3,    8)
        rf_n   = trial.suggest_int("rf_n",     100,  300)

        model = build_ensemble(
            lr_C=lr_C, rf_n=rf_n,
            xgb_lr=xgb_lr, xgb_n=xgb_n, xgb_depth=xgb_d
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring="f1_macro", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[Optuna] Best F1 (3-fold): {study.best_value:.4f}")
    return study.best_params


# ── TRAINING ───────────────────────────────────────────────────────────────────

def train_model(X_train, y_train,
                tune: bool = True,
                n_trials: int = 20):
    """
    Fit vectorizer + ensemble.
    Returns (model, vectorizer).
    """
    vectorizer = build_vectorizer()
    X_tr_feat = build_feature_matrix(X_train, vectorizer, fit=True)

    if tune:
        best = tune_hyperparameters(X_tr_feat, y_train, n_trials)
        model = build_ensemble(
            lr_C=best.get("lr_C", 1.0),
            rf_n=best.get("rf_n", 200),
            xgb_lr=best.get("xgb_lr", 0.1),
            xgb_n=best.get("xgb_n", 100),
            xgb_depth=best.get("xgb_d", 6),
        )
    else:
        model = build_ensemble()

    print("[Train] Fitting ensemble …")
    model.fit(X_tr_feat, y_train)
    return model, vectorizer


# ── CROSS-VALIDATION REPORT ────────────────────────────────────────────────────

def cross_validate_model(X, y, n_splits: int = 5) -> dict:
    """Run StratifiedKFold CV with default ensemble; return mean ± std."""
    vectorizer = build_vectorizer()
    X_feat = build_feature_matrix(X, vectorizer, fit=True)
    model = build_ensemble()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_feat, y, cv=cv,
                             scoring="f1_macro", n_jobs=-1)
    result = {
        "mean_f1": scores.mean(),
        "std_f1":  scores.std(),
        "splits":  scores.tolist(),
    }
    print(f"[CV] {n_splits}-fold F1: {scores.mean():.4f} ± {scores.std():.4f}")
    return result


# ── EVALUATION ─────────────────────────────────────────────────────────────────

def evaluate_model(model, vectorizer, X_test, y_test):
    X_feat = build_feature_matrix(X_test, vectorizer, fit=False)
    y_pred = model.predict(X_feat)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred,
                                     target_names=["Fake", "Real"])
    cm       = confusion_matrix(y_test, y_pred)
    return accuracy, report, cm

# ── PER-MODEL EVALUATION ───────────────────────────────────────────────────────

def evaluate_individual_models(model, vectorizer, X_test, y_test):
    """
    Evaluate each sub-model (LR, RF, XGB) individually + the full ensemble.
    Prints a comparison table and returns a dict of results.
    """
    X_feat = build_feature_matrix(X_test, vectorizer, fit=False)

    results = {}

    # Extract and evaluate each named estimator from the VotingClassifier
    for (name, estimator) in zip([e[0] for e in model.estimators], model.estimators_):
        y_pred = estimator.predict(X_feat)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred,
                                       target_names=["Fake", "Real"],
                                       output_dict=True)
        results[name] = {"accuracy": acc, "report": report}

    # Ensemble
    y_pred_ens = model.predict(X_feat)
    acc_ens = accuracy_score(y_test, y_pred_ens)
    report_ens = classification_report(y_test, y_pred_ens,
                                       target_names=["Fake", "Real"],
                                       output_dict=True)
    results["ensemble"] = {"accuracy": acc_ens, "report": report_ens}

    # Print comparison table
    print("\n" + "=" * 60)
    print(f"{'Model':<20} {'Accuracy':>10} {'F1 Fake':>10} {'F1 Real':>10}")
    print("=" * 60)
    labels = {"lr": "Logistic Regression", "rf": "Random Forest",
              "xgb": "XGBoost", "ensemble": "Ensemble (all)"}
    for key, res in results.items():
        acc = res["accuracy"]
        f1_fake = res["report"]["Fake"]["f1-score"]
        f1_real = res["report"]["Real"]["f1-score"]
        print(f"{labels.get(key, key):<20} {acc:>10.4f} {f1_fake:>10.4f} {f1_real:>10.4f}")
    print("=" * 60 + "\n")

    return results

# ── PREDICTION ─────────────────────────────────────────────────────────────────

def predict_news(news: str, model, vectorizer):
    """
    Returns (label_str, prediction_int, proba_array, feature_matrix_row).
    """
    import pandas as pd
    texts = pd.Series([news])
    feat  = build_feature_matrix(texts, vectorizer, fit=False)
    proba = model.predict_proba(feat)[0]
    pred  = model.predict(feat)[0]
    label = "Fake News" if pred == 0 else "Real News"
    return label, int(pred), proba, feat
