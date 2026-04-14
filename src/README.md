# Fake News Detector — Setup

## Project structure
```
fakenews/
├── data/raw/
│   ├── liar_dataset/        ← from liar_dataset.zip
│   │   ├── train.tsv
│   │   ├── valid.tsv
│   │   └── test.tsv
│   └── isot/               ← from ISOT zip
│       ├── Fake.csv
│       └── True.csv
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── pipeline.py
│   ├── explainer.py
│   └── run.py (or root run.py)
└── requirements.txt
```

## Dataset download

**LIAR dataset:**
```
wget https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
unzip liar_dataset.zip -d data/raw/liar_dataset/
```

**ISOT dataset:**
Download manually from https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php
Place Fake.csv and True.csv in `data/raw/isot/`

## Install
```
pip install -r requirements.txt
```

## Run
```
# Train + evaluate (with Optuna tuning, 20 trials)
python run.py

# Skip tuning (fast test)
python run.py --no-tune

# 5-fold cross-validation report
python run.py --cv
```

## What's new vs original
| Feature | Before | After |
|---|---|---|
| Dataset | Kaggle (Fake/True CSVs) | LIAR + ISOT |
| Model | Logistic Regression only | **Ensemble: LR + RF + XGBoost** |
| Features | TF-IDF only | TF-IDF + **13 handcrafted features** |
| Tuning | None | **Optuna (Bayesian search)** |
| Validation | Single split | **StratifiedKFold CV** |
| Explainability | LR coef hack | **SHAP LinearExplainer** |
