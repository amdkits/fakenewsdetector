"""
Handcrafted linguistic feature engineering.
Extracts statistical/stylistic signals alongside TF-IDF.
"""

import re
import string
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix


# ── TEXT CLEANING ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def clean_text_for_tfidf(text: str) -> str:
    """Removes punctuation/digits for TF-IDF."""
    text = clean_text(text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text.strip()


# ── HANDCRAFTED FEATURES ───────────────────────────────────────────────────────

def _safe_div(a, b):
    return a / b if b > 0 else 0.0


def extract_handcrafted(texts: pd.Series) -> np.ndarray:
    """
    Returns (n_samples, n_features) numpy array.

    Features:
      0  char_count
      1  word_count
      2  sentence_count
      3  avg_word_length
      4  avg_sentence_length   (words per sentence)
      5  caps_ratio            (UPPERCASE chars / total alpha)
      6  punct_ratio           (punctuation chars / total chars)
      7  exclamation_count
      8  question_count
      9  lexical_diversity     (unique words / total words)
      10 digit_ratio           (digit chars / total chars)
      11 stopword_ratio        (rough proxy via short words ≤3 chars)
      12 title_case_ratio      (Title-case words / word_count)
    """
    records = []
    for raw in texts:
        text = str(raw)
        clean = clean_text(text)

        words = clean.split()
        sentences = re.split(r'[.!?]+', clean)
        sentences = [s for s in sentences if s.strip()]

        char_count = len(clean)
        word_count = len(words)
        sentence_count = max(len(sentences), 1)
        avg_word_len = _safe_div(sum(len(w) for w in words), word_count)
        avg_sent_len = _safe_div(word_count, sentence_count)

        alpha_chars = sum(c.isalpha() for c in text)
        caps_ratio = _safe_div(sum(c.isupper() for c in text), alpha_chars)

        punct_chars = sum(c in string.punctuation for c in text)
        punct_ratio = _safe_div(punct_chars, char_count)

        excl_count = text.count('!')
        ques_count = text.count('?')

        unique_words = set(w.lower() for w in words)
        lexical_div = _safe_div(len(unique_words), word_count)

        digit_ratio = _safe_div(sum(c.isdigit() for c in text), char_count)

        short_words = sum(1 for w in words if len(w) <= 3)
        stopword_ratio = _safe_div(short_words, word_count)

        title_case_words = sum(1 for w in words if w.istitle())
        title_case_ratio = _safe_div(title_case_words, word_count)

        records.append([
            char_count, word_count, sentence_count,
            avg_word_len, avg_sent_len,
            caps_ratio, punct_ratio,
            excl_count, ques_count,
            lexical_div, digit_ratio,
            stopword_ratio, title_case_ratio
        ])

    return np.array(records, dtype=np.float32)


HANDCRAFTED_FEATURE_NAMES = [
    "char_count", "word_count", "sentence_count",
    "avg_word_length", "avg_sentence_length",
    "caps_ratio", "punct_ratio",
    "exclamation_count", "question_count",
    "lexical_diversity", "digit_ratio",
    "stopword_ratio", "title_case_ratio"
]


def combine_features(tfidf_matrix, handcrafted_matrix: np.ndarray):
    """Horizontally stack sparse TF-IDF + dense handcrafted features."""
    hc_sparse = csr_matrix(handcrafted_matrix)
    return hstack([tfidf_matrix, hc_sparse])
