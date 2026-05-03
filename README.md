# 🎬 IMDB Sentiment Analysis: Production-Grade NLP Pipeline

This repository contains a high-performance NLP pipeline designed for classifying sentiment in movie reviews. The project focuses on **production-ready Software Engineering**, **rigorous validation via Nested K-Fold CV**, and **Hybrid ML Architectures** combining classical lexical features with Deep Learning semantic embeddings.

The methodology implemented here is directly applicable to large-scale e-commerce sentiment analysis (e.g., product reviews in the Kaspi.kz ecosystem).

## 🏆 Final Model Performance (Nested CV AUC)

The project utilizes a strict 5-Fold Nested Cross-Validation strategy to guarantee unbiased evaluation. The final solution is a Logistic Regression Stacking Ensemble that blends lexical and semantic models.

| Model Architecture | Features | Test/OOF AUC |
| :--- | :--- | :--- |
| **Logistic Regression (Meta-Learner)** | **OOF Probabilities (Ensemble)** | **0.9663** |
| CatBoost Baseline | TF-IDF (13k) + Meta | 0.9541 |
| PyTorch MLP (1-Layer, 128 Units) | DistilBERT [CLS] + Meta | 0.9501 |
| Logistic Regression (Elastic Net) | DistilBERT [CLS] + Meta | 0.9465 |
| CatBoost (Initial Attempt) | DistilBERT [CLS] + Meta | 0.9173 |
| CatBoost (Structural) | Meta-Features Only | 0.8230 |

---

## 🏗 Project Architecture & Workflow

The codebase underwent a massive architectural refactor to separate mathematical operations from script execution, resulting in a clean, production-grade structure:

*   **`src/imdb/`**: The core backend package housing all heavy lifting (Data loading, PyTorch `nn.Module` definitions, Scikit-Learn/CatBoost wrappers, and strict Cross-Validation logic).
*   **`pipelines/`**: Lightweight, declarative execution scripts (`01` through `08`) that orchestrate the `src/` modules.
*   **Experiment Tracking:** Fully integrated with local **MLflow** (`mlruns.db`) to persistently log metrics, hyperparameters (Optuna), and model weights.

---

## 🛠 Phase 1: ETL & Feature Engineering

The first stage focuses on transforming raw, noisy HTML strings into a rich, structured dataset ready for high-performance boosting trees and transformers.

### 1. Data Cleaning (Level 1-3)
*   **Structural Scrubbing:** Regex-based removal of HTML tags (`<br />`) and URLs to clean the input stream without losing semantic integrity.
*   **Semantic Standardization:** Lowercasing, punctuation removal, and white-space normalization.
*   **Lemmatization:** Reducing words to their base forms (NLTK) to reduce vocabulary sparsity for the TF-IDF baseline.

### 2. Feature Engineering Logic
Six key meta-features were engineered to capture signals that are often lost during deep text cleaning:
*   **VADER Sentiment:** Rule-based polarity scores calculated on raw text to capture intensity before standardization.
*   **Lexical Complexity (TTR):** Type-Token Ratio to measure vocabulary richness.
*   **Uppercase Density:** Capturing "shouting" signals (emphasized words) before lowercasing.
*   **Punctuation Densities:** Separate density metrics for `!` and `?` to detect emotional intensity and sarcasm/confusion.
*   **Word Count:** Capturing the correlation between review length and sentiment polarization.

---

## 🧠 Phase 2: Modeling & Optimization

The modeling strategy leverages the **Diversity of Errors** principle. We trained distinct models with different "worldviews" (lexical vs. semantic) to feed into a final meta-learner.

### 1. The Lexical Baseline (TF-IDF + CatBoost)
*   Extracts top 13,000 unigrams and bigrams.
*   Combined with 7 meta-features, processed by CatBoost utilizing internal quantization/binning.
*   **Strengths:** Highly accurate at identifying explicit sentiment keywords.

### 2. The Semantic Neural Classifiers (DistilBERT)
Rather than passing dense 768-D embeddings into tree models (which proved highly sub-optimal and slow, yielding only 0.9173 AUC), we utilized native mathematical classifiers:
*   **Logistic Regression:** Tuned with the SAGA solver to apply an **Elastic Net** (L1 + L2) penalty to the high-dimensional embeddings.
*   **PyTorch MLP:** A custom neural network trained via GPU. Tuned via Optuna for network depth, units, dropout, and learning rate. Regularized using **AdamW** (Weight Decay/L2).

### 3. The Stacking Ensemble
A Level-1 Logistic Regression meta-learner trained purely on the Out-Of-Fold (OOF) probability predictions of the Baseline, LogReg, and MLP models. By operating in Log-Odds space, the stacker learned to perfectly balance the lexical rigidity of TF-IDF with the contextual awareness of BERT.

---

## 🚧 Key Challenges & Solutions

**1. The Transformer Context Window (512 Tokens)**
*   *Problem:* Standard BERT truncates reviews longer than 512 tokens, causing the model to miss the "punchline" or conclusion of long movie reviews.
*   *Solution:* Implemented a **128/128 Head/Tail Tokenization Strategy**. The custom pipeline slices the first 127 tokens and the last 127 tokens, concatenating them around `[CLS]` and `[SEP]` tags. This ensures the model sees both the setup and the final verdict of the review.

**2. Data Leakage in Cross-Validation**
*   *Problem:* Applying `StandardScaler` or `TfidfVectorizer` to the entire dataset before splitting causes information from the validation/test folds to leak into the training data.
*   *Solution:* Engineered a **Universal Nested CV** function (`src/imdb/training/cv.py`) that strictly isolates fitting transformations to the training fold of each inner and outer split. 

**3. SHAP Interpretability on Dense Vectors**
*   *Problem:* Running `shap.TreeExplainer` on CatBoost models trained on DistilBERT embeddings yields useless insights (e.g., "Dimension 594 is highly predictive").
*   *Solution:* SHAP analysis is strictly partitioned. `TreeExplainer` is applied *only* to the interpretable Baseline (TF-IDF words) and Meta-Only models. For the ensemble, `shap.LinearExplainer` is used to visualize exactly how the Meta-Learner weighs the trust between the base models.

---

## 💻 Tech Stack
*   **Core:** Python 3.12, Pandas, NumPy, SciPy.
*   **NLP:** NLTK, VADER, Hugging Face Transformers (`distilbert-base-uncased`).
*   **Machine Learning:** Scikit-Learn, CatBoost, PyTorch.
*   **MLOps & Optimization:** Optuna, MLflow.
*   **DevOps/QA:** Pytest, Ruff (Linting), PyYAML.

---

## 🚀 How to Run

*Note: The trained model artifacts (`.joblib`, `.pt`, `.cbm`) are tracked in this repository, but the raw data and OOF predictions are excluded via `.gitignore`.*

### 1. Setup & Installation
```bash
# Clone the repository
git clone <repo_url>
cd imdb_reviews

# Install the local src/ package and dependencies
pip install -e .
```

### 2. Data Preparation
Download the Kaggle IMDB Dataset (50k movie reviews) and place it in the raw data directory:
`data/raw/IMDB Dataset.csv`

### 3. Execution Pipeline
Run the pipelines sequentially from the project root:

```bash
# 1. Clean data and engineer meta-features
python -m pipelines.01_run_preprocessing

# 2. Extract DistilBERT Head/Tail embeddings (GPU recommended)
python -m pipelines.04_gen_bert_head_tail

# 3. Train Base Models (Generates OOF predictions & MLflow tracking)
python -m pipelines.03_train_catboost_meta_idf
python -m pipelines.05_train_logreg_meta_bert
python -m pipelines.06_train_mlp_meta_bert

# 4. Train Level-1 Stacking Ensemble
python -m pipelines.07_train_stacking_ensemble

# 5. Generate Interpretability Visualizations (Reports saved to reports/figures/)
python -m pipelines.08_run_shap_analysis
```

### 4. Viewing MLflow Metrics
To view the experiment tracking dashboard (Nested CV fold metrics, Optuna hyperparameter traces, etc.):
```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```
