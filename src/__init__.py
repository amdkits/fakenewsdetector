[tool.poetry]
name = "fakeneews"
version = "0.1.0"
description = "Fake News Detector — Ensemble ML pipeline with SHAP explainability"
authors = ["Your Name <you@example.com>"]
readme = "README.md"
packages = [{ include = "src" }]

[tool.poetry.dependencies]
python = "^3.10"
pandas = ">=2.0"
numpy = ">=1.24"
scikit-learn = ">=1.4"
xgboost = ">=2.0"
optuna = ">=3.5"
shap = ">=0.44"
joblib = ">=1.3"
groq = ">=0.4"
scipy = ">=1.11"

[tool.poetry.scripts]
fakenews = "run:main"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry-core.masonry.api"
