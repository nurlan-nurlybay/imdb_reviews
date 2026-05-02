"""
Pipeline 04: Production Model — CatBoost GPU on DistilBERT Embeddings

Dense dataset (768 CLS + 6 meta = 774 features) → CatBoost GPU.
Strict Nested K-Fold CV + Optuna HPO + OOF predictions + MLflow.

Usage:
    python -m pipelines.04_train_production
"""
from __future__ import annotations
import logging, os
from typing import Any
import mlflow, numpy as np, optuna, pandas as pd, structlog
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from imdb.utils.config import load_config
from imdb.utils.logging import setup_logger

setup_logger("logs/04_train_production.log", terminal_level=logging.INFO, file_level=logging.DEBUG)
logger = structlog.get_logger(__name__)

class TqdmOptunaCallback:
    def __init__(self, n_trials: int, desc: str) -> None:
        self.pbar = tqdm(total=n_trials, desc=desc, leave=False)
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        self.pbar.update(1)
    def close(self) -> None:
        self.pbar.close()

def objective(
    trial: optuna.Trial, X: np.ndarray, y: np.ndarray, cfg: dict[str, Any],
) -> float:
    """Inner CV — tunes CatBoost GPU hyperparams on dense features."""
    ss = cfg["production"]["catboost"]["search_space"]
    cb_params: dict[str, Any] = cfg["production"]["catboost"]["static"].copy()
    cb_params.update({
        "iterations": trial.suggest_int("iterations", ss["iterations"]["low"], ss["iterations"]["high"]),
        "depth": trial.suggest_int("depth", ss["depth"]["low"], ss["depth"]["high"]),
        "learning_rate": trial.suggest_float("learning_rate", ss["learning_rate"]["low"], ss["learning_rate"]["high"], log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", ss["l2_leaf_reg"]["low"], ss["l2_leaf_reg"]["high"]),
        "random_seed": cfg["experiment"]["random_state"],
    })

    inner_cv = StratifiedKFold(
        n_splits=cfg["experiment"]["inner_splits"], shuffle=True,
        random_state=cfg["experiment"]["random_state"],
    )
    aucs: list[float] = []
    for tr_idx, va_idx in inner_cv.split(X, y):
        model = CatBoostClassifier(**cb_params)
        model.fit(
            Pool(X[tr_idx], y[tr_idx]), eval_set=Pool(X[va_idx], y[va_idx]),
            early_stopping_rounds=50, verbose=0,
        )
        preds = model.predict_proba(X[va_idx])[:, 1]
        aucs.append(float(roc_auc_score(y[va_idx], preds)))
    return float(np.mean(aucs))

def main() -> None:
    logger.info("pipeline_start", pipeline="04_train_production")
    cfg_paths: dict[str, Any] = load_config("configs/paths.yaml")
    cfg_params: dict[str, Any] = load_config("configs/params.yaml")

    mlflow.set_tracking_uri(cfg_paths["tracking"]["mlflow_uri"])
    mlflow.set_experiment(cfg_params["production"]["name"])

    # ---- Load dense embeddings ----
    df = pd.read_parquet(cfg_paths["data"]["bert_embeddings"])
    target_col: str = cfg_params["data"]["target_col"]
    feature_cols = [c for c in df.columns if c != target_col]
    X: np.ndarray = df[feature_cols].to_numpy().astype(np.float32)
    y: np.ndarray = df[target_col].to_numpy()
    logger.info("data_loaded", shape=X.shape, target_balance=round(float(y.mean()), 3))

    n_samples = len(y)
    n_splits: int = cfg_params["experiment"]["n_splits"]
    n_trials: int = cfg_params["production"]["n_trials"]

    oof_preds = np.full(n_samples, np.nan, dtype=np.float64)
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg_params["experiment"]["random_state"])
    outer_aucs: list[float] = []
    best_params_list: list[tuple[float, dict[str, Any]]] = []

    logger.info("nested_cv_start", outer=n_splits, inner=cfg_params["experiment"]["inner_splits"], trials=n_trials)

    with mlflow.start_run(run_name="Nested_CV_Production"):
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            logger.info("outer_fold_start", fold=fold_idx + 1)
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            study = optuna.create_study(direction="maximize")
            cb = TqdmOptunaCallback(n_trials, desc=f"Fold {fold_idx + 1} Optuna")
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(
                lambda trial: objective(trial, X_train, y_train, cfg_params),
                n_trials=n_trials, callbacks=[cb],
            )
            cb.close()

            bp = study.best_params
            best_params_list.append((study.best_value, bp))

            cb_params: dict[str, Any] = cfg_params["production"]["catboost"]["static"].copy()
            cb_params["random_seed"] = cfg_params["experiment"]["random_state"]
            cb_params.update(bp)

            model = CatBoostClassifier(**cb_params)
            model.fit(Pool(X_train, y_train), verbose=0)

            preds = model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, preds))
            outer_aucs.append(auc)
            oof_preds[test_idx] = preds

            logger.info("outer_fold_done", fold=fold_idx + 1, auc=round(auc, 4))
            mlflow.log_metric(f"fold_{fold_idx + 1}_auc", auc, step=fold_idx)

        nested_auc = float(np.mean(outer_aucs))
        nested_std = float(np.std(outer_aucs))
        logger.info("nested_cv_done", auc=round(nested_auc, 4), std=round(nested_std, 4))
        mlflow.log_metric("true_nested_cv_auc", nested_auc)
        mlflow.log_metric("nested_cv_std", nested_std)

        # ---- Save OOF ----
        oof_df = pd.DataFrame({"oof_prob": oof_preds, "label": y})
        os.makedirs(os.path.dirname(cfg_paths["predictions"]["production_oof"]), exist_ok=True)
        oof_df.to_parquet(cfg_paths["predictions"]["production_oof"], index=False)
        logger.info("oof_saved", path=cfg_paths["predictions"]["production_oof"])

        # ---- Refit on 100% ----
        best_score, best_overall = max(best_params_list, key=lambda x: x[0])
        logger.info("refit_start", best_inner_auc=round(best_score, 4), params=best_overall)
        mlflow.log_params({f"best_{k}": v for k, v in best_overall.items()})

        cb_final: dict[str, Any] = cfg_params["production"]["catboost"]["static"].copy()
        cb_final["random_seed"] = cfg_params["experiment"]["random_state"]
        cb_final.update(best_overall)

        final_model = CatBoostClassifier(**cb_final)
        final_model.fit(Pool(X, y), verbose=100)

        os.makedirs(os.path.dirname(cfg_paths["models"]["best_catboost_production"]), exist_ok=True)
        final_model.save_model(cfg_paths["models"]["best_catboost_production"])
        logger.info("model_saved", path=cfg_paths["models"]["best_catboost_production"])

    logger.info("pipeline_done", pipeline="04_train_production")

if __name__ == "__main__":
    main()
