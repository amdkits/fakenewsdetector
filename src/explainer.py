"""
Feature attribution explainability for VotingClassifier ensemble.
Uses LR coefficient x feature value (reliable on single samples).
"""

import os
import numpy as np


def get_all_feature_names(vectorizer):
    from src.feature_engineering import HANDCRAFTED_FEATURE_NAMES
    return list(vectorizer.get_feature_names_out()) + HANDCRAFTED_FEATURE_NAMES


def get_top_features_coef(feat_matrix, model, vectorizer, n: int = 8):
    """LR coef x feature value — works reliably on single sparse rows."""
    lr_model = dict(model.named_estimators_)["lr"]
    feature_names = np.array(get_all_feature_names(vectorizer))
    coef = lr_model.coef_[0]
    dense = np.asarray(feat_matrix.todense())[0]
    contributions = dense * coef

    abs_sorted = np.argsort(np.abs(contributions))[::-1]
    top_idx = abs_sorted[:n * 4]

    fake_features = [
        (feature_names[i], round(float(contributions[i]), 4))
        for i in top_idx
        if contributions[i] < 0
    ][:n]

    real_features = [
        (feature_names[i], round(float(contributions[i]), 4))
        for i in top_idx
        if contributions[i] > 0
    ][:n]

    return fake_features, real_features


def get_top_features(feat_matrix, model, vectorizer, n: int = 8):
    return get_top_features_coef(feat_matrix, model, vectorizer, n)


def explain_prediction(news_text: str, prediction: int, proba,
                        fake_features, real_features) -> str:
    client_fn = _get_groq_client()
    if client_fn is None:
        return _fallback_explain(prediction, proba, fake_features, real_features)
    return client_fn(news_text, prediction, proba, fake_features, real_features)


def _get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from groq import Groq
    except ImportError:
        return None

    def call(news_text, prediction, proba, fake_f, real_f):
        client = Groq(api_key=api_key)
        label     = "Fake News" if prediction == 0 else "Real News"
        fake_conf = round(proba[0] * 100, 1)
        real_conf = round(proba[1] * 100, 1)
        prompt = f"""An ensemble ML classifier (Logistic Regression + Random Forest + XGBoost)
analyzed this news text and classified it as **{label}**.

Confidence: {fake_conf}% fake, {real_conf}% real.

Top features pushing toward FAKE (contribution scores):
{fake_f}

Top features pushing toward REAL:
{real_f}

Original text:
\"\"\"{news_text}\"\"\"

Explain in 3-5 sentences why the ensemble made this call.
Reference the actual features listed. Be direct, no hedging.
Do not explain how TF-IDF works."""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system",
                 "content": "You are an ML interpretability assistant. Explain classifier decisions using feature attribution data."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    return call


def _fallback_explain(prediction, proba, fake_features, real_features) -> str:
    label     = "Fake News" if prediction == 0 else "Real News"
    fake_conf = round(proba[0] * 100, 1)
    real_conf = round(proba[1] * 100, 1)
    top_fake  = ", ".join(f[0] for f in fake_features[:3]) or "none"
    top_real  = ", ".join(f[0] for f in real_features[:3]) or "none"
    return (
        f"Classified as {label} ({fake_conf}% fake / {real_conf}% real).\n"
        f"Key fake signals: {top_fake}.\n"
        f"Key real signals: {top_real}.\n"
        "(Set GROQ_API_KEY for LLM explanation.)"
    )
