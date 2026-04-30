# 🎬 IMDB Sentiment Analysis: Production-Grade NLP Pipeline

This repository contains a high-performance NLP pipeline designed for classifying sentiment in movie reviews. The project focuses on **production-ready Data Engineering**, **rigorous validation via Nested K-Fold CV**, and **Hybrid ML architectures**.

The methodology implemented here is directly applicable to large-scale e-commerce sentiment analysis (e.g., product reviews in the Kaspi.kz ecosystem).

```
done so far:
src/imdb/data/preprocessor.py
tests/
```

## 🏗 Project Architecture & Workflow
The project is structured to separate the data plumbing from the modeling logic, ensuring scalability and maintainability.

*   **ETL & Preprocessing:** Modular Python classes for text scrubbing.
*   **Feature Engineering:** Extraction of meta-features capturing human emotion and vocabulary richness.
*   **Validation:** Nested K-Fold Cross-Validation to ensure statistical robustness and avoid "lucky" splits.
*   **Experiment Tracking:** MLflow integration for logging metrics, parameters, and model versions.

---

## 🛠 Phase 1: ETL & Feature Engineering (COMPLETED)
The first stage focused on transforming raw, noisy HTML strings into a rich, structured dataset ready for high-performance boosting trees and transformers.

### 1. Data Cleaning (Level 1-3)
*   **Structural Scrubbing:** Regex-based removal of HTML tags (`<br />`) and URLs to clean the input stream without losing semantic integrity.
*   **Semantic Standardization:** Lowercasing, punctuation removal, and white-space normalization.
*   **Lemmatization:** Reducing words to their base forms (NLTK) to reduce vocabulary sparsity for the TF-IDF baseline.

### 2. Feature Engineering Logic
Six key meta-features were engineered to capture signal that is often lost during deep text cleaning:
*   **VADER Sentiment:** Rule-based polarity scores calculated on raw text to capture intensity before standardization.
*   **Lexical Complexity (TTR):** Type-Token Ratio to measure vocabulary richness.
*   **Uppercase Density:** Capturing "shouting" signals (emphasized words) before lowercasing.
*   **Punctuation Densities:** Separate density metrics for `!` and `?` to detect emotional intensity and sarcasm/confusion.
*   **Word Count:** Capturing the correlation between review length and sentiment polarization.

### 3. Engineering Rigor
*   **Unit Testing:** Comprehensive test suite using `pytest` with module-scoped fixtures for sequential pipeline testing.
*   **Configuration Management:** All paths and hyperparameters are externalized in `.yaml` files.
*   **Performance:** Vectorized Pandas operations and list comprehensions utilized to handle the 50,000-row dataset in under 3 seconds.

---

## 🚀 Phase 2: Modeling Roadmap (TODO)

### 1. Baseline: Sparse Hybrid Model
*   **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency).
*   **Model:** CatBoostClassifier.
*   **Optimization:** **Optuna** for hyperparameter tuning.
*   **Validation:** Nested K-Fold CV to generate **Out-of-Fold (OOF)** predictions.

### 2. Production: Transformer Embedding Model
*   **Feature Extraction:** Generating 768-dimensional sentence embeddings using **DistilBERT** (Hugging Face).
*   **Hybrid Head:** CatBoost trained on the concatenation of [DistilBERT Vectors + Engineered Meta-Features].
*   **Reasoning:** This hybrid approach combines the deep semantic understanding of Transformers with the structural/stylistic signal of meta-features.

### 3. Interpretability & Analysis
*   **SHAP (TreeExplainer):** Explaining the final CatBoost model to quantify the contribution of meta-features vs. semantic embeddings.
*   **Ablation Studies:** Systematically dropping feature sets to verify their impact on model performance.

---

## 💻 Tech Stack
*   **Core:** Python 3.12, Pandas, NumPy, Scikit-Learn.
*   **NLP:** NLTK, VADER, Hugging Face Transformers.
*   **ML Models:** CatBoost.
*   **Optimization/Tracking:** Optuna, MLflow.
*   **DevOps/QA:** Pytest, Ruff (Linting), PyYAML.

---

### How to Run
```bash
# Install dependencies
pip install -e .

# Run data preprocessing pipeline
python pipelines/run_preprocessing.py

# Execute test suite
pytest -v -x
```
