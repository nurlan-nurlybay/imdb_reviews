"""
Pipeline 06: PyTorch MLP on BERT + Meta Features (Nested CV)
"""
import os
import torch
import pandas as pd
import structlog
import mlflow
import logging
from imdb.utils.config import load_config
from imdb.utils.logging import setup_logger
from imdb.features.loader import load_bert_and_meta
from imdb.models.pytorch_mlp import objective_mlp, fit_predict_mlp, train_full_mlp
from imdb.training.cv import run_nested_cv

setup_logger("logs/06_train_mlp_meta_bert.log", terminal_level=logging.INFO, file_level=logging.DEBUG)
logger = structlog.get_logger(__name__)

def main() -> None:
    logger.info("pipeline_start", pipeline="06_train_mlp_meta_bert")
    cfg_paths = load_config("configs/paths.yaml")
    cfg_params = load_config("configs/params.yaml")

    mlflow.set_tracking_uri(cfg_paths["tracking"]["mlflow_uri"])
    mlflow.set_experiment(cfg_params["neural_classifiers"]["name"] + "_mlp")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("device", device=str(device))
    
    X, y = load_bert_and_meta(cfg_paths, cfg_params)
    
    def bound_objective(trial, X_tr, y_tr):
        return objective_mlp(trial, X_tr, y_tr, cfg_params, device)
        
    def bound_fit_predict(X_tr, y_tr, X_te, y_te, best_params):
        return fit_predict_mlp(X_tr, y_tr, X_te, y_te, best_params, cfg_params, device)

    oof_preds, best_overall_params, outer_metrics = run_nested_cv(
        X=X,
        y=y,
        objective_fn=bound_objective,
        fit_predict_fn=bound_fit_predict,
        n_splits=cfg_params["experiment"]["n_splits"],
        n_trials=cfg_params["neural_classifiers"]["n_trials"],
        random_state=cfg_params["experiment"]["random_state"],
        direction="maximize",
        mlflow_run_name="Nested_CV_MLP"
    )
    
    # Save OOF
    oof_df = pd.DataFrame({"oof_prob": oof_preds, "label": y})
    # We expect configs/paths.yaml to have predictions: mlp_meta_bert_oof: "..."
    # If not present yet, default to a path
    out_oof = cfg_paths.get("predictions", {}).get("mlp_meta_bert_oof", "data/processed/mlp_meta_bert_oof.parquet")
    os.makedirs(os.path.dirname(out_oof), exist_ok=True)
    oof_df.to_parquet(out_oof, index=False)
    
    # Train full model
    final_model, final_scaler = train_full_mlp(X, y, best_overall_params, cfg_params, device)
    
    out_model = cfg_paths.get("models", {}).get("best_mlp", "models/best_mlp_meta_bert.pt")
    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    torch.save(final_model.state_dict(), out_model)
    
    logger.info("pipeline_done", pipeline="06_train_mlp_meta_bert")

if __name__ == "__main__":
    main()
