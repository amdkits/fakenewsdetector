# Fake News Detector

A machine learning project that classifies news articles as Fake or Real using TF-IDF vectorization and Logistic Regression.

The model is trained on the Kaggle Fake and True News dataset.

## Project Structure

```
.
├── data
│   └── raw
│       ├── Fake.csv
│       └── True.csv
├── poetry.lock
├── pyproject.toml
├── README.md
├── run.py
└── src
    └── pipeline.py
```

## Requirements

* Python 3.11+ recommended
* Kaggle Fake and True News dataset placed inside:

```
data/raw/
```

---

# Running on Linux (Poetry)

## 1. Install Poetry

If Poetry is not installed:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Ensure Poetry is available:

```bash
poetry --version
```

## 2. Install Dependencies

From the project root:

```bash
poetry install
```

## 3. Run the Project

```bash
poetry run python run.py
```

The program will:

* Load dataset
* Train Logistic Regression model
* Print accuracy and classification report
* Ask for custom news input for prediction

---

# Running on Windows (VS Code)

## Option A — Using Poetry (Recommended)

1. Install Python (3.11 recommended)
2. Install Poetry
3. Open the project folder in VS Code
4. Open Terminal inside VS Code
5. Run:

```powershell
poetry install
poetry run python run.py
```

---

## Option B — Without Poetry (Using venv)

If Poetry is not preferred:

### 1. Create Virtual Environment

```powershell
python -m venv venv
```

### 2. Activate Environment

```powershell
venv\Scripts\activate
```

### 3. Install Required Packages

```powershell
pip install pandas scikit-learn matplotlib seaborn
```

### 4. Run

```powershell
python run.py
```

---

# Notes

* The dataset must exist in `data/raw/`
* The model is trained on full article text, not just headlines
* For best predictions, input a full news paragraph instead of a short headline

---

# Model Details

* Text cleaning using regex preprocessing
* TF-IDF vectorization (bag-of-words with weighting)
* Logistic Regression classifier
* Train/test split: 75/25

---

# Future Improvements

* Save trained model to disk
* Add command-line arguments instead of interactive input
* Add support for headline-only classification
* Deploy as API using FastAPI

---

