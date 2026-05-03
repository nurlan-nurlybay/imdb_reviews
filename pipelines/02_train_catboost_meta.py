"""
Pipeline 02: CatBoost on Meta Features Only (Nested CV)
"""
import os
import pandas as pd
import numpy as np
import structlog
import mlflow
from catboost import CatBoostClassifier, Pool
import logging
from imdb.utils.config import load_config
from imdb.utils.logging import setup_logger
from imdb.training.cv import run_nested_cv
from imdb.models.catboost_wrapper import objective_catboost, fit_predict_catboost
from transformers import AutoTokenizer

setup_logger("logs/02_train_catboost_meta.log", terminal_level=logging.INFO, file_level=logging.DEBUG)
logger = structlog.get_logger(__name__)

def main() -> None:
    logger.info("pipeline_start", pipeline="02_train_catboost_meta")
    cfg_paths = load_config("configs/paths.yaml")
    cfg_params = load_config("configs/params.yaml")

    mlflow.set_tracking_uri(cfg_paths["tracking"]["mlflow_uri"])
    mlflow.set_experiment(cfg_params.get("meta_only", {}).get("name", "imdb_sentiment_meta_only"))

    df = pd.read_parquet(cfg_paths["data"]["processed"])
    
    target_col = cfg_params["data"]["target_col"]
    meta_cols = cfg_params["data"]["meta_cols"].copy()
    
    if 'token_count' not in df.columns:
        logger.info("calculating_token_count")
        tokenizer = AutoTokenizer.from_pretrained(cfg_params["bert"]["model_name"])
        tokens = tokenizer(df['review_lemmatized'].astype(str).tolist(), add_special_tokens=True, truncation=False)
        df['token_count'] = [len(t) for t in tokens['input_ids']]
        
    meta_cols.append('token_count')
    
    X = df[meta_cols].to_numpy().astype(np.float32)
    y = df[target_col].to_numpy().astype(np.float32)
    
    # Notice we pass cfg_params mapped to the generic config structure
    # However, objective_catboost expects 'catboost' block. We can just pass the meta_only block directly
    cfg_mapped = {"catboost": cfg_params.get("meta_only", {}).get("catboost", {})}
    
    def bound_objective(trial, X_tr, y_tr):
        return objective_catboost(trial, X_tr, y_tr, cfg_mapped)
        
    def bound_fit_predict(X_tr, y_tr, X_te, y_te, best_params):
        return fit_predict_catboost(X_tr, y_tr, X_te, y_te, best_params)

    oof_preds, best_overall_params, outer_metrics = run_nested_cv(
        X=X,
        y=y,
        objective_fn=bound_objective,
        fit_predict_fn=bound_fit_predict,
        n_splits=cfg_params["experiment"]["n_splits"],
        n_trials=cfg_params.get("meta_only", {}).get("n_trials", 20),
        random_state=cfg_params["experiment"]["random_state"],
        direction="maximize",
        mlflow_run_name="Nested_CV_MetaOnly"
    )
    
    # Save OOF
    oof_df = pd.DataFrame({"oof_prob": oof_preds, "label": y})
    out_oof = cfg_paths.get("predictions", {}).get("catboost_meta_oof", "data/processed/catboost_meta_oof.parquet")
    os.makedirs(os.path.dirname(out_oof), exist_ok=True)
    oof_df.to_parquet(out_oof, index=False)
    
    # Final Model Retrain
    cb_params = {"loss_function": "Logloss", "eval_metric": "AUC", "verbose": 100, "random_seed": 42}
    cb_params.update(best_overall_params)
    
    final_model = CatBoostClassifier(**cb_params)
    final_model.fit(Pool(X, y))
    
    out_model = cfg_paths.get("models", {}).get("best_catboost_meta_only", "models/best_catboost_meta_only.cbm")
    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    final_model.save_model(out_model)
    
    logger.info("pipeline_done", pipeline="02_train_catboost_meta")

if __name__ == "__main__":
    main()
