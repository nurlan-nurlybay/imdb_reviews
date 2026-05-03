"""
Pipeline 05: SHAP Interpretability Analysis

Only applies SHAP to tree-based interpretable models:
1. Baseline (TF-IDF + Meta)
2. Meta-Only
Dense deep learning vectors (768-D) are excluded as their SHAP values are uninterpretable.

Usage:
    python -m pipelines.05_shap_analysis
"""
from __future__ import annotations
import logging
import os
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import structlog
from catboost import CatBoostClassifier, Pool
from scipy.sparse import csr_matrix, hstack
from transformers import AutoTokenizer

from imdb.utils.config import load_config
from imdb.utils.logging import setup_logger

setup_logger("logs/05_shap_analysis.log", terminal_level=logging.INFO, file_level=logging.DEBUG)
logger = structlog.get_logger(__name__)

def generate_shap_plot(model: CatBoostClassifier, X: Any, X_raw: Any, feature_names: list[str], output_path: str, title: str):
    logger.info("generating_shap", title=title, num_features=len(feature_names))
    
    explainer = shap.TreeExplainer(model)
    # Use the Pool (X) for calculation, but X_raw for plotting
    shap_values = explainer.shap_values(X)
    
    plt.figure(figsize=(12, 8))
    
    # If it's the TF-IDF matrix, it's a CSR matrix. summary_plot needs a dense sample or 
    # specific handling. We'll convert our 1000-row sample to dense for the plot.
    if hasattr(X_raw, "toarray"):
        X_plot = X_raw.toarray()
    else:
        X_plot = X_raw
        
    # max_display=20 keeps the plot clean and avoids crashing on the TF-IDF features
    shap.summary_plot(
        shap_values, 
        X_plot, 
        feature_names=feature_names, 
        max_display=20, 
        show=False
    )
    
    plt.title(title, pad=20, fontsize=16)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("shap_plot_saved", path=output_path)

def main() -> None:
    logger.info("pipeline_start", pipeline="05_shap_analysis")
    cfg_paths: dict[str, Any] = load_config("configs/paths.yaml")
    cfg_params: dict[str, Any] = load_config("configs/params.yaml")
    
    # ---- Load Data ----
    df = pd.read_parquet(cfg_paths["data"]["processed"])
    # Sample 1000 rows to save time
    df = df.sample(n=1000, random_state=cfg_params["experiment"]["random_state"]).copy()
    
    meta_cols = cfg_params["data"]["meta_cols"].copy()
    
    # Calculate token_count if not present
    if 'token_count' not in df.columns:
        logger.info("calculating_token_count")
        tokenizer = AutoTokenizer.from_pretrained(cfg_params["bert"]["model_name"])
        tokens = tokenizer(df['review_lemmatized'].astype(str).tolist(), add_special_tokens=True, truncation=False)
        df['token_count'] = [len(t) for t in tokens['input_ids']]
        
    meta_cols.append('token_count')
    X_meta = df[meta_cols].to_numpy()
    
    # ==========================================
    # Model A: Baseline (TF-IDF + Meta)
    # ==========================================
    logger.info("shap_baseline_start")
    baseline_model_path = cfg_paths["models"]["best_catboost_baseline"]
    vectorizer_path = cfg_paths["models"]["best_tfidf"]
    
    if os.path.exists(baseline_model_path) and os.path.exists(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
        tfidf_features = vectorizer.transform(df["review_lemmatized"])
        
        X_baseline = hstack([tfidf_features, csr_matrix(X_meta)]).tocsr()
        feature_names_baseline = list(vectorizer.get_feature_names_out()) + meta_cols
        
        model_baseline = CatBoostClassifier()
        model_baseline.load_model(baseline_model_path)
        
        # We need to pass the Pool to CatBoost SHAP for sparse matrices, but TreeExplainer accepts Pool
        pool_baseline = Pool(X_baseline, feature_names=feature_names_baseline)
        
        generate_shap_plot(
            model_baseline, 
            pool_baseline, 
            X_baseline, # Pass the raw CSR matrix here
            feature_names_baseline, 
            cfg_paths["reports"]["shap_summary_baseline"],
            "SHAP Top 20 Features (TF-IDF + Meta Baseline)"
        )
    else:
        logger.warning("baseline_model_missing", path=baseline_model_path)

    # ==========================================
    # Model B: Meta-Only
    # ==========================================
    logger.info("shap_meta_only_start")
    meta_model_path = cfg_paths["models"]["best_catboost_meta_only"]
    
    if os.path.exists(meta_model_path):
        model_meta = CatBoostClassifier()
        model_meta.load_model(meta_model_path)
        
        pool_meta = Pool(X_meta, feature_names=meta_cols)
        
        generate_shap_plot(
            model_meta, 
            pool_meta, 
            X_meta, # Pass the raw numpy array here
            meta_cols, 
            cfg_paths["reports"]["shap_summary_meta_only"],
            "SHAP Feature Importance (Meta-Only)"
        )
    else:
        logger.warning("meta_only_model_missing", path=meta_model_path)

    logger.info("pipeline_done", pipeline="05_shap_analysis")

if __name__ == "__main__":
    main()
