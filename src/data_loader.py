"""
Multi-dataset loader.
Supports: LIAR dataset (.tsv) + ISOT dataset (Fake.csv / True.csv)
"""

import pandas as pd
from pathlib import Path


# ── LIAR ──────────────────────────────────────────────────────────────────────
# Labels: pants-fire, false, barely-true, half-true, mostly-true, true
# Binarize: pants-fire / false / barely-true → 0 (fake), rest → 1 (real)
LIAR_COLS = [
    "id", "label_raw", "statement", "subject", "speaker",
    "speaker_job", "state", "party", "barely_true_count",
    "false_count", "half_true_count", "mostly_true_count",
    "pants_on_fire_count", "context"
]
LIAR_FAKE_LABELS = {"pants-fire", "false", "barely-true"}


def load_liar(data_dir: Path) -> pd.DataFrame:
    """Load train/valid/test splits from LIAR dataset."""
    liar_dir = data_dir / "liar_dataset"
    dfs = []
    for split_file in ["train.tsv", "valid.tsv", "test.tsv"]:
        fp = liar_dir / split_file
        if not fp.exists():
            raise FileNotFoundError(f"LIAR file missing: {fp}")
        df = pd.read_csv(fp, sep="\t", header=None, names=LIAR_COLS)
        df["label"] = df["label_raw"].apply(
            lambda x: 0 if x in LIAR_FAKE_LABELS else 1
        )
        df["text"] = df["statement"].astype(str)
        df["source"] = "liar"
        dfs.append(df[["text", "label", "source"]])
    return pd.concat(dfs, ignore_index=True)


# ── ISOT ──────────────────────────────────────────────────────────────────────

def load_isot(data_dir: Path) -> pd.DataFrame:
    """Load ISOT Fake/True CSVs."""
    isot_dir = data_dir / "isot"
    fake_fp = isot_dir / "Fake.csv"
    true_fp = isot_dir / "True.csv"

    if not fake_fp.exists() or not true_fp.exists():
        raise FileNotFoundError(f"ISOT files missing in {isot_dir}")

    fake = pd.read_csv(fake_fp)
    true = pd.read_csv(true_fp)
    fake["label"] = 0
    true["label"] = 1

    for df in [fake, true]:
        # Use title + text if both available
        if "title" in df.columns and "text" in df.columns:
            df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
        elif "text" in df.columns:
            df["text"] = df["text"].fillna("")
        else:
            df["text"] = df["title"].fillna("")

    combined = pd.concat([fake, true], ignore_index=True)
    combined["source"] = "isot"
    return combined[["text", "label", "source"]]


# ── UNIFIED ───────────────────────────────────────────────────────────────────

def load_all(data_dir: Path, use_liar: bool = True, use_isot: bool = True) -> pd.DataFrame:
    """Load and merge all requested datasets."""
    frames = []

    if use_liar:
        try:
            df = load_liar(data_dir)
            frames.append(df)
            print(f"[LIAR]  loaded {len(df):,} samples")
        except FileNotFoundError as e:
            print(f"[LIAR]  skipped — {e}")

    if use_isot:
        try:
            df = load_isot(data_dir)
            frames.append(df)
            print(f"[ISOT]  loaded {len(df):,} samples")
        except FileNotFoundError as e:
            print(f"[ISOT]  skipped — {e}")

    if not frames:
        raise RuntimeError("No dataset loaded. Check data_dir paths.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[TOTAL] {len(combined):,} samples | "
          f"fake={combined['label'].eq(0).sum():,} real={combined['label'].eq(1).sum():,}")
    return combined
