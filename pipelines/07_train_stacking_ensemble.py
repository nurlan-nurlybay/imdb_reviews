"""
Pipeline 07: Stacking Ensemble
Blends OOF predictions from Level 0 models using a Logistic Regression Meta-Learner.
"""
import os
import joblib
import pandas as pd
import numpy as np
import structlog
import mlflow
from typing import Dict, List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import logging
from imdb.utils.config import load_config
from imdb.utils.logging import setup_logger

setup_logger("logs/07_train_stacking_ensemble.log", terminal_level=logging.INFO, file_level=logging.DEBUG)
logger = structlog.get_logger(__name__)

def main() -> None:
    logger.info("pipeline_start", pipeline="07_train_stacking_ensemble")
    cfg_paths = load_config("configs/paths.yaml")
    cfg_params = load_config("configs/params.yaml")
    
    mlflow.set_tracking_uri(cfg_paths["tracking"]["mlflow_uri"])
    mlflow.set_experiment(cfg_params.get("experiment", {}).get("name", "imdb_sentiment_ensemble") + "_ensemble")
    
    # We only stack models that provide diverse errors
    oof_paths: Dict[str, str] = {
        "Baseline (CatBoost+IDF)": cfg_paths.get("predictions", {}).get("baseline_oof", "data/processed/baseline_oof.parquet"),
        "MLP (DistilBERT)": cfg_paths.get("predictions", {}).get("mlp_meta_bert_oof", "data/processed/mlp_meta_bert_oof.parquet"),
        "LogReg (DistilBERT)": cfg_paths.get("predictions", {}).get("logreg_meta_bert_oof", "data/processed/logreg_meta_bert_oof.parquet"),
    }
    
    dfs: Dict[str, pd.DataFrame] = {}
    for name, path in oof_paths.items():
        if os.path.exists(path):
            dfs[name] = pd.read_parquet(path)
            logger.info("loaded_oof", model=name, rows=len(dfs[name]))
        else:
            logger.warning("missing_oof", model=name, path=path)
            
    if len(dfs) == 0:
        logger.error("no_oof_files_found")
        return
        
    # Build Level 1 Feature Matrix
    X_list: List[np.ndarray] = []
    y: Optional[np.ndarray] = None
    feature_names: List[str] = []
    
    for name, df in dfs.items():
        if y is None:
            y = df['label'].to_numpy()
        else:
            assert np.array_equal(y, df['label'].to_numpy()), f"Labels mismatch for {name}!"
            
        X_list.append(df['oof_prob'].to_numpy())
        feature_names.append(name)
        
    if y is None:
        raise ValueError("Could not extract labels from OOF DataFrames.")
        
    X = np.column_stack(X_list)
    
    with mlflow.start_run(run_name="LogReg_MetaLearner"):
        random_state = cfg_params.get("experiment", {}).get("random_state", 42)
        n_splits = cfg_params.get("experiment", {}).get("n_splits", 5)
        
        meta_model = LogisticRegression(random_state=random_state)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        oof_preds = cross_val_predict(meta_model, X, y, cv=cv, method='predict_proba')[:, 1]
        
        ensemble_auc = float(roc_auc_score(y, oof_preds))
        logger.info("ensemble_cv_done", auc=round(ensemble_auc, 4))
        mlflow.log_metric("ensemble_nested_cv_auc", ensemble_auc)
        
        # Fit final model
        meta_model.fit(X, y)
        coef_dict = {feat: float(coef) for feat, coef in zip(feature_names, meta_model.coef_[0])}
        logger.info("ensemble_weights", weights=coef_dict)
        mlflow.log_params(coef_dict)
        
        # Save OOF
        oof_df = pd.DataFrame({"oof_prob": oof_preds, "label": y})
        out_oof = cfg_paths.get("predictions", {}).get("ensemble_oof", "data/processed/ensemble_oof.parquet")
        os.makedirs(os.path.dirname(out_oof), exist_ok=True)
        oof_df.to_parquet(out_oof, index=False)
        
        # Save Model
        out_model = cfg_paths.get("models", {}).get("best_ensemble", "models/best_ensemble.joblib")
        os.makedirs(os.path.dirname(out_model), exist_ok=True)
        joblib.dump(meta_model, out_model)

    logger.info("pipeline_done", pipeline="07_train_stacking_ensemble")

if __name__ == "__main__":
    main()
