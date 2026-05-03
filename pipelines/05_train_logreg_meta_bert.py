"""
Pipeline 05: Logistic Regression on BERT + Meta Features (Nested CV)
"""
import os
import joblib
import pandas as pd
import structlog
import mlflow
import logging
from imdb.utils.config import load_config
from imdb.utils.logging import setup_logger
from imdb.features.loader import load_scaled_bert_and_meta
from imdb.models.sklearn_wrapper import objective_logreg, fit_predict_logreg
from imdb.training.cv import run_nested_cv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

setup_logger("logs/05_train_logreg_meta_bert.log", terminal_level=logging.INFO, file_level=logging.DEBUG)
logger = structlog.get_logger(__name__)

def main() -> None:
    logger.info("pipeline_start", pipeline="05_train_logreg_meta_bert")
    cfg_paths = load_config("configs/paths.yaml")
    cfg_params = load_config("configs/params.yaml")

    mlflow.set_tracking_uri(cfg_paths["tracking"]["mlflow_uri"])
    mlflow.set_experiment(cfg_params["neural_classifiers"]["name"] + "_logreg")
    
    # Pre-processed data handling offloaded to src/imdb/features/loader.py
    X, y = load_scaled_bert_and_meta(cfg_paths, cfg_params)
    
    def bound_objective(trial, X_tr, y_tr):
        return objective_logreg(trial, X_tr, y_tr)
        
    def bound_fit_predict(X_tr, y_tr, X_te, y_te, best_params):
        return fit_predict_logreg(X_tr, y_tr, X_te, y_te, best_params)

    # Universal Nested CV Execution
    oof_preds, best_overall_params, outer_metrics = run_nested_cv(
        X=X,
        y=y,
        objective_fn=bound_objective,
        fit_predict_fn=bound_fit_predict,
        n_splits=cfg_params["experiment"]["n_splits"],
        n_trials=cfg_params["neural_classifiers"]["n_trials"],
        random_state=cfg_params["experiment"]["random_state"],
        direction="maximize",
        mlflow_run_name="Nested_CV_LogReg"
    )
    
    # Save OOF
    oof_df = pd.DataFrame({"oof_prob": oof_preds, "label": y})
    out_oof = cfg_paths.get("predictions", {}).get("logreg_meta_bert_oof", "data/processed/logreg_meta_bert_oof.parquet")
    os.makedirs(os.path.dirname(out_oof), exist_ok=True)
    oof_df.to_parquet(out_oof, index=False)
    
    # Train full model
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    final_model = LogisticRegression(
        penalty='elasticnet', 
        solver='saga', 
        C=best_overall_params["C"], 
        l1_ratio=best_overall_params["l1_ratio"], 
        max_iter=1000, 
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_s, y)
    
    out_model = cfg_paths.get("models", {}).get("best_logreg", "models/best_logreg_meta_bert.joblib")
    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    joblib.dump({"model": final_model, "scaler": scaler}, out_model)
    
    logger.info("pipeline_done", pipeline="05_train_logreg_meta_bert")

if __name__ == "__main__":
    main()
