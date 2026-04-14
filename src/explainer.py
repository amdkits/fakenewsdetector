"""
SHAP-based explainability for the VotingClassifier ensemble.
Falls back to LR coefficient attribution if SHAP unavailable.
"""

import os
import numpy as np


# ── FEATURE NAMES ──────────────────────────────────────────────────────────────

def get_all_feature_names(vectorizer):
    from src.feature_engineering import HANDCRAFTED_FEATURE_NAMES
    return list(vectorizer.get_feature_names_out()) + HANDCRAFTED_FEATURE_NAMES


# ── SHAP ATTRIBUTION ──────────────────────────────────────────────────────────

def get_top_features_shap(feat_matrix, model, vectorizer, n: int = 8):
    """
    Use SHAP LinearExplainer on the LR sub-model inside VotingClassifier.
    Returns (fake_features, real_features) as (name, shap_val) lists.
    """
    try:
        import shap
    except ImportError:
        return get_top_features_coef(feat_matrix, model, vectorizer, n)

    # Extract LR from voting ensemble
    lr_model = dict(model.named_estimators_)["lr"]
    feature_names = get_all_feature_names(vectorizer)

    explainer = shap.LinearExplainer(lr_model, feat_matrix,
                                     feature_perturbation="interventional")
    shap_values = explainer.shap_values(feat_matrix)

    # shap_values shape: (n_samples, n_features) or list for multiclass
    if isinstance(shap_values, list):
        # class 1 = real
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    sorted_idx = np.argsort(sv)
    fn = np.array(feature_names)

    fake_features = [
        (fn[i], round(float(sv[i]), 4))
        for i in sorted_idx[:n]
        if sv[i] < 0
    ]
    real_features = [
        (fn[i], round(float(sv[i]), 4))
        for i in sorted_idx[-n:][::-1]
        if sv[i] > 0
    ]
    return fake_features, real_features


# ── COEF FALLBACK ──────────────────────────────────────────────────────────────

def get_top_features_coef(feat_matrix, model, vectorizer, n: int = 8):
    """Fallback: LR coef × feature value attribution."""
    lr_model = dict(model.named_estimators_)["lr"]
    feature_names = get_all_feature_names(vectorizer)
    coef = lr_model.coef_[0]
    dense = np.asarray(feat_matrix.todense())[0]
    contributions = dense * coef
    sorted_idx = contributions.argsort()
    fn = np.array(feature_names)

    fake_features = [
        (fn[i], round(float(contributions[i]), 4))
        for i in sorted_idx[:n]
        if contributions[i] < 0
    ]
    real_features = [
        (fn[i], round(float(contributions[i]), 4))
        for i in sorted_idx[-n:][::-1]
        if contributions[i] > 0
    ]
    return fake_features, real_features


# ── ALIAS ─────────────────────────────────────────────────────────────────────

def get_top_features(feat_matrix, model, vectorizer, n: int = 8):
    """Auto-selects SHAP → coef fallback."""
    return get_top_features_shap(feat_matrix, model, vectorizer, n)


# ── LLM EXPLANATION ──────────────────────────────────────────────────────────

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
        label      = "Fake News" if prediction == 0 else "Real News"
        fake_conf  = round(proba[0] * 100, 1)
        real_conf  = round(proba[1] * 100, 1)
        prompt = f"""An ensemble ML classifier (Logistic Regression + Random Forest + XGBoost)
analyzed this news text and classified it as **{label}**.

Confidence: {fake_conf}% fake, {real_conf}% real.

Top features pushing toward FAKE (SHAP / contribution scores):
{fake_f}

Top features pushing toward REAL:
{real_f}

Original text:
\"\"\"{news_text}\"\"\"

Explain in 3-5 sentences why the ensemble made this call.
Reference the actual features listed. Be direct, no hedging.
Do not explain how TF-IDF or SHAP works."""

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
