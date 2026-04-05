import os
from openai import OpenAI


def get_top_features(news_vec, model, vectorizer, n=8):
    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]
    dense = news_vec.toarray()[0]
    contributions = dense * coef

    sorted_idx = contributions.argsort()

    fake_features = [
        (feature_names[i], round(float(contributions[i]), 4))
        for i in sorted_idx[:n]
        if contributions[i] < 0
    ]
    real_features = [
        (feature_names[i], round(float(contributions[i]), 4))
        for i in sorted_idx[-n:][::-1]
        if contributions[i] > 0
    ]
    return fake_features, real_features


def explain_prediction(news_text, prediction, proba, fake_features, real_features):
    client = OpenAI(
    api_key=os.environ["QWEN_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

    label = "Fake News" if prediction == 0 else "Real News"
    fake_conf = round(proba[0] * 100, 1)
    real_conf = round(proba[1] * 100, 1)

    prompt = f"""A TF-IDF + Logistic Regression classifier analyzed this news text and classified it as **{label}**.

Confidence: {fake_conf}% fake, {real_conf}% real.

Top words/phrases pushing toward FAKE (contribution scores):
{fake_features}

Top words/phrases pushing toward REAL:
{real_features}

Original text:
\"\"\"{news_text}\"\"\"

Explain in 3-5 sentences why the model made this call. Reference the actual features listed. Be direct, no hedging. Do not explain how TF-IDF works."""

    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {
                "role": "system",
                "content": "You are an ML interpretability assistant. Explain classifier decisions using feature attribution data.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()
