"""
Pipeline 07: Meta-Only Model — CatBoost on 7 Metadata Features

Goal: Train a CatBoost model using only the 7 structural metadata features.
Single Train/Val/Test split (60/20/20).

Usage:
    python -m pipelines.07_train_meta_only
"""
from __future__ import annotations
import logging
import os
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
import structlog
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer

from imdb.utils.config import load_config
from imdb.utils.logging import setup_logger

setup_logger("logs/07_train_meta_only.log", terminal_level=logging.INFO, file_level=logging.DEBUG)
logger = structlog.get_logger(__name__)

class TqdmOptunaCallback:
    def __init__(self, n_trials: int, desc: str) -> None:
        self.pbar = tqdm(total=n_trials, desc=desc, leave=False)
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        self.pbar.update(1)
    def close(self) -> None:
        self.pbar.close()

def objective(
    trial: optuna.Trial, 
    X_train: np.ndarray, y_train: np.ndarray, 
    X_val: np.ndarray, y_val: np.ndarray, 
    cfg: dict[str, Any]
) -> float:
    ss = cfg["meta_only"]["catboost"]["search_space"]
    cb_params: dict[str, Any] = cfg["meta_only"]["catboost"]["static"].copy()
    
    cb_params.update({
        "iterations": trial.suggest_int("iterations", ss["iterations"]["low"], ss["iterations"]["high"]),
        "depth": trial.suggest_int("depth", ss["depth"]["low"], ss["depth"]["high"]),
        "learning_rate": trial.suggest_float("learning_rate", float(ss["learning_rate"]["low"]), float(ss["learning_rate"]["high"]), log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", float(ss["l2_leaf_reg"]["low"]), float(ss["l2_leaf_reg"]["high"])),
        "random_seed": cfg["experiment"]["random_state"],
    })
    
    model = CatBoostClassifier(**cb_params)
    model.fit(
        Pool(X_train, y_train),
        eval_set=Pool(X_val, y_val),
        early_stopping_rounds=50,
        verbose=0
    )
    
    preds = model.predict_proba(X_val)[:, 1]
    return float(roc_auc_score(y_val, preds))

def main() -> None:
    logger.info("pipeline_start", pipeline="07_train_meta_only")
    cfg_paths: dict[str, Any] = load_config("configs/paths.yaml")
    cfg_params: dict[str, Any] = load_config("configs/params.yaml")

    mlflow.set_tracking_uri(cfg_paths["tracking"]["mlflow_uri"])
    mlflow.set_experiment(cfg_params["meta_only"]["name"])

    # ---- Load Data ----
    df = pd.read_parquet(cfg_paths["data"]["processed"])
    
    target_col = cfg_params["data"]["target_col"]
    meta_cols = cfg_params["data"]["meta_cols"].copy()
    
    # Calculate token_count if not present
    if 'token_count' not in df.columns:
        logger.info("calculating_token_count")
        tokenizer = AutoTokenizer.from_pretrained(cfg_params["bert"]["model_name"])
        tokens = tokenizer(df['review_lemmatized'].astype(str).tolist(), add_special_tokens=True, truncation=False)
        df['token_count'] = [len(t) for t in tokens['input_ids']]
        
    meta_cols.append('token_count')
    
    X = df[meta_cols].to_numpy().astype(np.float32)
    y = df[target_col].to_numpy().astype(np.float32)
    
    logger.info("data_loaded", shape=X.shape)
    
    # ---- 60/20/20 Split ----
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=cfg_params["experiment"]["random_state"], stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=cfg_params["experiment"]["random_state"], stratify=y_temp)
    
    logger.info("data_split", train=len(X_train), val=len(X_val), test=len(X_test))
    
    n_trials = cfg_params["meta_only"]["n_trials"]

    with mlflow.start_run(run_name="CatBoost_Meta_Only_Tuning"):
        study = optuna.create_study(direction="maximize")
        cb = TqdmOptunaCallback(n_trials, desc="Meta-Only Optuna")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val, cfg_params),
            n_trials=n_trials,
            callbacks=[cb]
        )
        cb.close()
        
        best_params = study.best_params
        logger.info("tuning_done", best_val_auc=round(study.best_value, 4), params=best_params)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        
        # Combine Train + Val for final refit
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])
        
        cb_params_final: dict[str, Any] = cfg_params["meta_only"]["catboost"]["static"].copy()
        cb_params_final["random_seed"] = cfg_params["experiment"]["random_state"]
        cb_params_final.update(best_params)
        
        final_model = CatBoostClassifier(**cb_params_final)
        final_model.fit(Pool(X_train_full, y_train_full), verbose=100)
        
        # Test Evaluation
        test_preds = final_model.predict_proba(X_test)[:, 1]
        test_auc = float(roc_auc_score(y_test, test_preds))
        
        logger.info("test_evaluation", test_auc=round(test_auc, 4))
        mlflow.log_metric("test_auc", test_auc)
        print(f"\\n--- Final Holdout Test AUC (Meta-Only) ---")
        print(f"CatBoost Meta-Only : {test_auc:.4f}")
        
        # Save model
        out_path = cfg_paths["models"]["best_catboost_meta_only"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        final_model.save_model(out_path)
        logger.info("model_saved", path=out_path)

    logger.info("pipeline_done", pipeline="07_train_meta_only")

if __name__ == "__main__":
    main()
