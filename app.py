"""
app.py — Streamlit UI for Fake News Ensemble Detector
Run: streamlit run app.py
"""

import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="FakeTrace",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'Space Grotesk', sans-serif;
    background: #f0ede8 !important;
    color: #1a1a1a;
}

section[data-testid="stSidebar"] { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }
header[data-testid="stHeader"] { background: transparent; }

.page-wrap {
    min-height: 100vh;
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: auto 1fr;
}

.top-bar {
    grid-column: 1 / -1;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.4rem 2.5rem;
    border-bottom: 1.5px solid #d4cfc8;
    background: #f0ede8;
}

.logo {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: #1a1a1a;
}

.logo span {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #e63946;
    margin-right: 8px;
    vertical-align: middle;
    position: relative;
    top: -1px;
}

.nav-pills {
    display: flex;
    gap: 0.5rem;
}

.pill {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 6px 14px;
    border-radius: 20px;
    border: 1px solid #c8c3bc;
    color: #666;
    background: transparent;
}

.pill.active {
    background: #1a1a1a;
    color: #f0ede8;
    border-color: #1a1a1a;
}

.left-panel {
    grid-column: 1;
    padding: 3rem 2.5rem;
    border-right: 1.5px solid #d4cfc8;
    display: flex;
    flex-direction: column;
    gap: 1.8rem;
}

.right-panel {
    grid-column: 2;
    padding: 3rem 2.5rem;
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #999;
    margin-bottom: 0.6rem;
}

.hero-text {
    font-size: 3.2rem;
    font-weight: 700;
    line-height: 1.05;
    letter-spacing: -2px;
    color: #1a1a1a;
}

.hero-text em {
    font-style: normal;
    color: #e63946;
}

.desc {
    font-size: 0.9rem;
    color: #666;
    line-height: 1.6;
    max-width: 420px;
}

.model-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

.model-chip {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    padding: 4px 10px;
    border-radius: 4px;
    background: #e8e3dc;
    color: #555;
    border: 1px solid #d4cfc8;
    letter-spacing: 0.5px;
}

/* Streamlit textarea override */
.stTextArea label { display: none !important; }
.stTextArea textarea {
    background: #e8e3dc !important;
    border: 1.5px solid #c8c3bc !important;
    border-radius: 10px !important;
    color: #1a1a1a !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.95rem !important;
    resize: none !important;
    line-height: 1.6 !important;
}
.stTextArea textarea:focus {
    border-color: #1a1a1a !important;
    box-shadow: none !important;
}

/* Button override */
.stButton > button {
    background: #1a1a1a !important;
    color: #f0ede8 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.3px !important;
    transition: background 0.15s !important;
}
.stButton > button:hover { background: #333 !important; }

/* Result card */
.result-card {
    border-radius: 14px;
    padding: 2rem;
    border: 1.5px solid;
    position: relative;
    overflow: hidden;
}

.result-fake {
    background: #fff0f0;
    border-color: #f5c6c6;
}

.result-real {
    background: #f0fff5;
    border-color: #b8e6c8;
}

.result-verdict {
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -1.5px;
    line-height: 1;
    margin-bottom: 1.2rem;
}

.verdict-fake { color: #c0392b; }
.verdict-real { color: #1a7a4a; }

.conf-row {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
}

.conf-item {
    display: flex;
    align-items: center;
    gap: 0.8rem;
}

.conf-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 1px;
    color: #888;
    width: 40px;
}

.conf-pct {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    width: 42px;
}

.conf-pct-fake { color: #c0392b; }
.conf-pct-real { color: #1a7a4a; }

.bar-track {
    flex: 1;
    height: 5px;
    background: #e0dbd4;
    border-radius: 3px;
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    border-radius: 3px;
}

.bar-fake { background: #e63946; }
.bar-real { background: #2d9e6b; }

.feature-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 0.5rem;
}

.feature-col { display: flex; flex-direction: column; gap: 0.4rem; }

.ftag {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    padding: 4px 9px;
    border-radius: 5px;
    display: inline-block;
}

.ftag-fake {
    background: #fde8e8;
    color: #c0392b;
    border: 1px solid #f5c6c6;
}

.ftag-real {
    background: #e0f5eb;
    color: #1a7a4a;
    border: 1px solid #b8e6c8;
}

.empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    gap: 1rem;
    opacity: 0.35;
    padding: 4rem 0;
}

.empty-icon {
    font-size: 3rem;
}

.empty-text {
    font-size: 0.85rem;
    color: #888;
    line-height: 1.5;
}

.expl-box {
    background: #e8e3dc;
    border: 1px solid #d4cfc8;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    font-size: 0.85rem;
    line-height: 1.7;
    color: #444;
}

.divider { border: none; border-top: 1.5px solid #d4cfc8; margin: 0.5rem 0; }

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    m, v = Path("model.pkl"), Path("vectorizer.pkl")
    if not m.exists() or not v.exists():
        return None, None
    return joblib.load(m), joblib.load(v)


model, vectorizer = load_model()

# ── TOP BAR ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
  <div class="logo"><span></span>FakeTrace</div>
  <div class="nav-pills">
    <div class="pill active">Detector</div>
    <div class="pill">About</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── TWO COLUMN LAYOUT ─────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("""
    <div class="left-panel">
      <div>
        <div class="section-label">AI-powered analysis</div>
        <div class="hero-text">Detect<br><em>Fake</em><br>News.</div>
      </div>
      <div class="desc">
        Paste any headline or article. Our ensemble model — Logistic Regression,
        Random Forest, and XGBoost — analyses linguistic patterns and flags misinformation.
      </div>
      <div>
        <div class="section-label">Models</div>
        <div class="model-chips">
          <span class="model-chip">LOGISTIC REGRESSION</span>
          <span class="model-chip">RANDOM FOREST</span>
          <span class="model-chip">XGBOOST</span>
          <span class="model-chip">TF-IDF BIGRAMS</span>
          <span class="model-chip">13 HANDCRAFTED FEATURES</span>
          <span class="model-chip">LIAR DATASET</span>
          <span class="model-chip">ISOT DATASET</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with right:
    if model is None:
        st.error("model.pkl not found. Run `python run.py --no-tune` first.")
        st.stop()

    st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
    news_input = st.text_area("", placeholder="Paste a news headline or article snippet…", height=160, label_visibility="collapsed")
    analyze = st.button("⚡ Analyze", use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)

    if analyze and news_input.strip():
        with st.spinner(""):
            from src.pipeline import predict_news
            from src.explainer import get_top_features, explain_prediction

            label, prediction, proba, feat = predict_news(news_input, model, vectorizer)
            fake_pct = round(proba[0] * 100, 1)
            real_pct = round(proba[1] * 100, 1)
            is_fake  = prediction == 0

            fake_f, real_f = get_top_features(feat, model, vectorizer, n=6)

            card_cls    = "result-fake" if is_fake else "result-real"
            verdict_cls = "verdict-fake" if is_fake else "verdict-real"
            icon        = "✗" if is_fake else "✓"

            fake_tags = "".join(f'<span class="ftag ftag-fake">{w}</span> ' for w, _ in fake_f) or "<span style='color:#ccc'>—</span>"
            real_tags = "".join(f'<span class="ftag ftag-real">{w}</span> ' for w, _ in real_f) or "<span style='color:#ccc'>—</span>"

            st.markdown(f"""
<div class="result-card {card_cls}">
  <div class="result-verdict {verdict_cls}">{icon} {label}</div>

  <div class="conf-row">
    <div class="conf-item">
      <span class="conf-label">FAKE</span>
      <span class="conf-pct conf-pct-fake">{fake_pct}%</span>
      <div class="bar-track"><div class="bar-fill bar-fake" style="width:{fake_pct}%"></div></div>
    </div>
    <div class="conf-item">
      <span class="conf-label">REAL</span>
      <span class="conf-pct conf-pct-real">{real_pct}%</span>
      <div class="bar-track"><div class="bar-fill bar-real" style="width:{real_pct}%"></div></div>
    </div>
  </div>

  <div class="section-label">Feature signals</div>
  <div class="feature-grid">
    <div class="feature-col">
      <div class="section-label" style="font-size:0.55rem">→ FAKE</div>
      {fake_tags}
    </div>
    <div class="feature-col">
      <div class="section-label" style="font-size:0.55rem">→ REAL</div>
      {real_tags}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🧠 Explain with AI  (needs GROQ_API_KEY)"):
                with st.spinner("Generating explanation…"):
                    expl = explain_prediction(news_input, prediction, proba, fake_f, real_f)
                st.markdown(f'<div class="expl-box">{expl}</div>', unsafe_allow_html=True)

    elif not analyze:
        st.markdown("""
<div class="empty-state">
  <div class="empty-icon">⚡</div>
  <div class="empty-text">Enter a headline above<br>and click Analyze</div>
</div>
""", unsafe_allow_html=True)
    else:
        st.warning("Enter some text first.")
