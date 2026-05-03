"""
Pipeline 03: TF-IDF + CatBoost Baseline (Nested CV)
"""
import os
import joblib
import pandas as pd
import numpy as np
import structlog
import mlflow
from scipy.sparse import csr_matrix, hstack
from catboost import CatBoostClassifier
import logging
from imdb.utils.config import load_config
from imdb.utils.logging import setup_logger
from imdb.training.cv import run_nested_cv
from imdb.models.catboost_wrapper import objective_tfidf_catboost, fit_predict_tfidf_catboost
from imdb.features.text import build_tfidf_features

setup_logger("logs/03_train_catboost_meta_idf.log", terminal_level=logging.INFO, file_level=logging.DEBUG)
logger = structlog.get_logger(__name__)

def main() -> None:
    logger.info("pipeline_start", pipeline="03_train_catboost_meta_idf")
    cfg_paths = load_config("configs/paths.yaml")
    cfg_params = load_config("configs/params.yaml")

    mlflow.set_tracking_uri(cfg_paths["tracking"]["mlflow_uri"])
    mlflow.set_experiment(cfg_params["experiment"]["name"])

    df = pd.read_parquet(cfg_paths["data"]["processed"])
    meta_cols = cfg_params["data"]["meta_cols"]

    X_text = df["review_lemmatized"].to_numpy()
    X_meta = df[meta_cols].to_numpy().astype(np.float32)
    y = df[cfg_params["data"]["target_col"]].to_numpy()

    # Pass as tuple for the CV loop to split jointly
    X_tuple = (X_text, X_meta)
    
    def bound_objective(trial, X_tr, y_tr):
        return objective_tfidf_catboost(trial, X_tr, y_tr, cfg_params)
        
    def bound_fit_predict(X_tr, y_tr, X_te, y_te, best_params):
        return fit_predict_tfidf_catboost(X_tr, y_tr, X_te, y_te, best_params, cfg_params)

    oof_preds, best_overall_params, outer_metrics = run_nested_cv(
        X=X_tuple,
        y=y,
        objective_fn=bound_objective,
        fit_predict_fn=bound_fit_predict,
        n_splits=cfg_params["experiment"]["n_splits"],
        n_trials=cfg_params["experiment"]["n_trials"],
        random_state=cfg_params["experiment"]["random_state"],
        direction="maximize",
        mlflow_run_name="Nested_CV_Baseline"
    )
    
    # Save OOF
    oof_df = pd.DataFrame({"oof_prob": oof_preds, "label": y})
    out_oof = cfg_paths.get("predictions", {}).get("baseline_oof", "data/processed/baseline_oof.parquet")
    os.makedirs(os.path.dirname(out_oof), exist_ok=True)
    oof_df.to_parquet(out_oof, index=False)
    
    # Final Model Retrain
    ngram_range = tuple(cfg_params["tfidf"]["ngram_range"][best_overall_params["ngram_range_idx"]])
    final_vectorizer, tfidf_full = build_tfidf_features(
        df=df, 
        text_col="review_lemmatized", 
        max_features=best_overall_params["max_features"],
        ngram_range=ngram_range,
        use_idf=best_overall_params["use_idf"]
    )
    
    X_full_sp = hstack([tfidf_full, csr_matrix(X_meta)]).tocsr()
    
    cb_params = {"loss_function": "Logloss", "eval_metric": "AUC", "verbose": 100, "random_seed": 42}
    cb_params.update({k: v for k, v in best_overall_params.items() if k not in {"max_features", "ngram_range_idx", "use_idf"}})
    
    final_model = CatBoostClassifier(**cb_params)
    final_model.fit(X_full_sp, y)
    
    # Save Artifacts
    out_vectorizer = cfg_paths.get("models", {}).get("best_tfidf", "models/best_tfidf.joblib")
    out_model = cfg_paths.get("models", {}).get("best_catboost_baseline", "models/best_catboost_baseline.cbm")
    
    os.makedirs(os.path.dirname(out_vectorizer), exist_ok=True)
    joblib.dump(final_vectorizer, out_vectorizer)
    
    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    final_model.save_model(out_model)
    
    logger.info("pipeline_done", pipeline="03_train_catboost_meta_idf")

if __name__ == "__main__":
    main()
