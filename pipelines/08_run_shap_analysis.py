"""
Pipeline 08: SHAP Interpretability Analysis
Analyzes tree-based models and the meta stacking ensemble.
"""
import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import structlog
from scipy.sparse import csr_matrix, hstack
from catboost import CatBoostClassifier, Pool
from transformers import AutoTokenizer
import logging
from imdb.utils.config import load_config
from imdb.utils.logging import setup_logger
from imdb.evaluation.shap_utils import generate_shap_plot

setup_logger("logs/08_run_shap_analysis.log", terminal_level=logging.INFO, file_level=logging.DEBUG)
logger = structlog.get_logger(__name__)

def main() -> None:
    logger.info("pipeline_start", pipeline="08_run_shap_analysis")
    cfg_paths = load_config("configs/paths.yaml")
    cfg_params = load_config("configs/params.yaml")
    
    # ---- Load Data Sample ----
    df = pd.read_parquet(cfg_paths["data"]["processed"])
    df = df.sample(n=1000, random_state=cfg_params["experiment"]["random_state"]).copy()
    
    meta_cols = cfg_params["data"]["meta_cols"].copy()
    
    if 'token_count' not in df.columns:
        tokenizer = AutoTokenizer.from_pretrained(cfg_params["bert"]["model_name"])
        tokens = tokenizer(df['review_lemmatized'].astype(str).tolist(), add_special_tokens=True, truncation=False)
        df['token_count'] = [len(t) for t in tokens['input_ids']]
        
    meta_cols.append('token_count')
    X_meta = df[meta_cols].to_numpy()
    
    # ==========================================
    # Model A: Baseline (TF-IDF + Meta)
    # ==========================================
    logger.info("shap_baseline_start")
    baseline_model_path = cfg_paths.get("models", {}).get("best_catboost_baseline", "models/best_catboost_baseline.cbm")
    vectorizer_path = cfg_paths.get("models", {}).get("best_tfidf", "models/best_tfidf.joblib")
    
    if os.path.exists(baseline_model_path) and os.path.exists(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
        tfidf_features = vectorizer.transform(df["review_lemmatized"])
        
        X_baseline = hstack([tfidf_features, csr_matrix(X_meta)]).tocsr()
        feature_names_baseline = list(vectorizer.get_feature_names_out()) + meta_cols
        
        model_baseline = CatBoostClassifier()
        model_baseline.load_model(baseline_model_path)
        pool_baseline = Pool(X_baseline, feature_names=feature_names_baseline)
        
        generate_shap_plot(
            model_baseline, 
            pool_baseline, 
            X_baseline, 
            feature_names_baseline, 
            cfg_paths.get("reports", {}).get("shap_summary_baseline", "reports/figures/shap_summary_baseline.png"),
            "SHAP Top 20 Features (TF-IDF + Meta Baseline)"
        )
    else:
        logger.warning("baseline_model_missing")

    # ==========================================
    # Model B: Meta-Only
    # ==========================================
    logger.info("shap_meta_only_start")
    meta_model_path = cfg_paths.get("models", {}).get("best_catboost_meta_only", "models/best_catboost_meta_only.cbm")
    
    if os.path.exists(meta_model_path):
        model_meta = CatBoostClassifier()
        model_meta.load_model(meta_model_path)
        pool_meta = Pool(X_meta, feature_names=meta_cols)
        
        generate_shap_plot(
            model_meta, 
            pool_meta, 
            X_meta, 
            meta_cols, 
            cfg_paths.get("reports", {}).get("shap_summary_meta_only", "reports/figures/shap_summary_meta_only.png"),
            "SHAP Feature Importance (Meta-Only)"
        )
    else:
        logger.warning("meta_only_model_missing")

    # ==========================================
    # Model C: Stacking Ensemble (LogReg Meta Learner)
    # ==========================================
    logger.info("shap_ensemble_start")
    ensemble_model_path = cfg_paths.get("models", {}).get("best_ensemble", "models/best_ensemble.joblib")
    
    if os.path.exists(ensemble_model_path):
        meta_learner = joblib.load(ensemble_model_path)
        
        oof_paths = {
            "Baseline_CatBoost_IDF": cfg_paths.get("predictions", {}).get("baseline_oof", "data/processed/baseline_oof.parquet"),
            "MLP_DistilBERT": cfg_paths.get("predictions", {}).get("mlp_meta_bert_oof", "data/processed/mlp_meta_bert_oof.parquet"),
            "LogReg_DistilBERT": cfg_paths.get("predictions", {}).get("logreg_meta_bert_oof", "data/processed/logreg_meta_bert_oof.parquet"),
        }
        
        ensemble_features = []
        feature_names = []
        valid_paths = True
        
        for name, path in oof_paths.items():
            if os.path.exists(path):
                oof_df = pd.read_parquet(path)
                ensemble_features.append(oof_df['oof_prob'].to_numpy())
                feature_names.append(name)
            else:
                valid_paths = False
                logger.warning("ensemble_oof_missing", path=path)
                break
                
        if valid_paths:
            X_ens = np.column_stack(ensemble_features)
            # Sample it to avoid drawing huge plots
            np.random.seed(42)
            idx = np.random.choice(len(X_ens), 1000, replace=False)
            X_ens_sample = X_ens[idx]
            
            explainer = shap.LinearExplainer(meta_learner, X_ens_sample)
            shap_values = explainer.shap_values(X_ens_sample)
            
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_ens_sample, feature_names=feature_names, show=False)
            plt.title("SHAP Feature Importance (Stacking Meta Learner)", pad=20, fontsize=16)
            
            out_path = cfg_paths.get("reports", {}).get("shap_summary_ensemble", "reports/figures/shap_summary_ensemble.png")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("shap_plot_saved", path=out_path)
            
    else:
        logger.warning("ensemble_model_missing")

    logger.info("pipeline_done", pipeline="08_run_shap_analysis")

if __name__ == "__main__":
    main()
