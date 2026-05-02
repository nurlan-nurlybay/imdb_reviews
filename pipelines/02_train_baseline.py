"""
Pipeline 02: TF-IDF + CatBoost Baseline — Strict Nested K-Fold CV

Implements:
    - Outer K-Fold: unbiased performance estimation via held-out OOF predictions
    - Inner K-Fold: Optuna HPO (TF-IDF + CatBoost params jointly) on outer-train only
    - CatBoost CPU mode with native scipy.sparse input (no dense conversion → no VRAM risk)
    - OOF predictions saved to parquet for downstream stacking / analysis
    - Final model retrained on 100% data with best overall parameters

Usage:
    python -m pipelines.02_train_baseline
"""

from __future__ import annotations

import logging
import os
from typing import Any

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import structlog
from catboost import CatBoostClassifier, Pool
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from imdb.utils.config import load_config
from imdb.utils.logging import setup_logger

setup_logger("logs/02_train_baseline.log", terminal_level=logging.INFO, file_level=logging.DEBUG)
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Optuna progress callback
# ---------------------------------------------------------------------------
class TqdmOptunaCallback:
    """Wraps a tqdm bar to track Optuna trial progress."""

    def __init__(self, n_trials: int, desc: str) -> None:
        self.pbar = tqdm(total=n_trials, desc=desc, leave=False)

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        self.pbar.update(1)

    def close(self) -> None:
        self.pbar.close()


# ---------------------------------------------------------------------------
# Inner-CV objective for Optuna
# ---------------------------------------------------------------------------
def objective(
    trial: optuna.Trial,
    X_text: np.ndarray,
    X_meta: np.ndarray,
    y: np.ndarray,
    cfg_params: dict[str, Any],
) -> float:
    """Inner CV loop — fits TF-IDF + CatBoost and returns mean AUC."""

    # ---- TF-IDF hyperparameters ----
    max_features: int = trial.suggest_int(
        "max_features",
        cfg_params["tfidf"]["max_features"]["low"],
        cfg_params["tfidf"]["max_features"]["high"],
        step=cfg_params["tfidf"]["max_features"]["step"],
    )
    ngram_idx: int = trial.suggest_categorical("ngram_range_idx", [0, 1])
    ngram_range: tuple[int, int] = tuple(cfg_params["tfidf"]["ngram_range"][ngram_idx])  # type: ignore[arg-type]
    use_idf: bool = trial.suggest_categorical("use_idf", cfg_params["tfidf"]["use_idf"])

    # ---- CatBoost hyperparameters ----
    cb_params: dict[str, Any] = cfg_params["catboost"]["static"].copy()
    ss = cfg_params["catboost"]["search_space"]
    cb_params.update(
        {
            "iterations": trial.suggest_int("iterations", ss["iterations"]["low"], ss["iterations"]["high"]),
            "depth": trial.suggest_int("depth", ss["depth"]["low"], ss["depth"]["high"]),
            "learning_rate": trial.suggest_float("learning_rate", ss["learning_rate"]["low"], ss["learning_rate"]["high"], log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", ss["l2_leaf_reg"]["low"], ss["l2_leaf_reg"]["high"]),
            "random_seed": cfg_params["experiment"]["random_state"],
        }
    )

    # ---- Inner CV ----
    inner_cv = StratifiedKFold(
        n_splits=cfg_params["experiment"]["inner_splits"],
        shuffle=True,
        random_state=cfg_params["experiment"]["random_state"],
    )
    inner_aucs: list[float] = []

    for train_idx, val_idx in inner_cv.split(X_text, y):
        text_train, text_val = X_text[train_idx], X_text[val_idx]
        meta_train, meta_val = X_meta[train_idx], X_meta[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, use_idf=use_idf)
        tfidf_train = vectorizer.fit_transform(text_train)
        tfidf_val = vectorizer.transform(text_val)

        X_train_sp = hstack([tfidf_train, csr_matrix(meta_train)]).tocsr()
        X_val_sp = hstack([tfidf_val, csr_matrix(meta_val)]).tocsr()

        model = CatBoostClassifier(**cb_params)
        model.fit(
            X_train_sp,
            y_train,
            eval_set=Pool(X_val_sp, y_val),
            early_stopping_rounds=30,
            verbose=0,
        )

        preds = model.predict_proba(X_val_sp)[:, 1]
        inner_aucs.append(float(roc_auc_score(y_val, preds)))

    return float(np.mean(inner_aucs))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("pipeline_start", pipeline="02_train_baseline")

    cfg_paths: dict[str, Any] = load_config("configs/paths.yaml")
    cfg_params: dict[str, Any] = load_config("configs/params.yaml")

    mlflow.set_tracking_uri(cfg_paths["tracking"]["mlflow_uri"])
    mlflow.set_experiment(cfg_params["experiment"]["name"])

    # ---- Load data ----
    df = pd.read_parquet(cfg_paths["data"]["processed"])
    meta_cols: list[str] = cfg_params["data"]["meta_cols"]

    X_text: np.ndarray = df["review_lemmatized"].to_numpy()
    X_meta: np.ndarray = df[meta_cols].to_numpy().astype(np.float32)
    y: np.ndarray = df[cfg_params["data"]["target_col"]].to_numpy()

    n_samples = len(y)
    n_splits: int = cfg_params["experiment"]["n_splits"]
    n_trials: int = cfg_params["experiment"]["n_trials"]

    # ---- OOF container ----
    oof_preds = np.full(n_samples, np.nan, dtype=np.float64)

    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg_params["experiment"]["random_state"])
    outer_aucs: list[float] = []
    best_params_across_folds: list[tuple[float, dict[str, Any]]] = []

    logger.info("nested_cv_start", outer_folds=n_splits, inner_folds=cfg_params["experiment"]["inner_splits"], n_trials=n_trials)

    with mlflow.start_run(run_name="Nested_CV_Baseline"):

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_text, y)):
            logger.info("outer_fold_start", fold=fold_idx + 1, total=n_splits)

            # ---- Outer split ----
            X_text_train, X_text_test = X_text[train_idx], X_text[test_idx]
            X_meta_train, X_meta_test = X_meta[train_idx], X_meta[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # ---- Optuna HPO on outer-train ----
            study = optuna.create_study(direction="maximize")
            tqdm_cb = TqdmOptunaCallback(n_trials, desc=f"Fold {fold_idx + 1} Optuna")
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            study.optimize(
                lambda trial: objective(trial, X_text_train, X_meta_train, y_train, cfg_params),
                n_trials=n_trials,
                callbacks=[tqdm_cb],
            )
            tqdm_cb.close()

            # ---- Retrain on full outer-train with best params ----
            best_params: dict[str, Any] = study.best_params
            best_params_across_folds.append((study.best_value, best_params))

            ngram_range: tuple[int, int] = tuple(cfg_params["tfidf"]["ngram_range"][best_params["ngram_range_idx"]])  # type: ignore[arg-type]
            vectorizer = TfidfVectorizer(
                max_features=best_params["max_features"],
                ngram_range=ngram_range,
                use_idf=best_params["use_idf"],
            )

            tfidf_train = vectorizer.fit_transform(X_text_train)
            tfidf_test = vectorizer.transform(X_text_test)

            X_train_sp = hstack([tfidf_train, csr_matrix(X_meta_train)]).tocsr()
            X_test_sp = hstack([tfidf_test, csr_matrix(X_meta_test)]).tocsr()

            cb_params: dict[str, Any] = cfg_params["catboost"]["static"].copy()
            cb_params["random_seed"] = cfg_params["experiment"]["random_state"]
            cb_params.update(
                {k: v for k, v in best_params.items() if k not in {"max_features", "ngram_range_idx", "use_idf"}}
            )

            model = CatBoostClassifier(**cb_params)
            model.fit(X_train_sp, y_train, verbose=0)

            # ---- Evaluate on held-out outer test ----
            preds = model.predict_proba(X_test_sp)[:, 1]
            outer_auc = float(roc_auc_score(y_test, preds))
            outer_aucs.append(outer_auc)

            # Store OOF predictions
            oof_preds[test_idx] = preds

            logger.info("outer_fold_done", fold=fold_idx + 1, auc=round(outer_auc, 4))
            mlflow.log_metric(f"fold_{fold_idx + 1}_auc", outer_auc, step=fold_idx)

        # ---- Aggregate Nested CV result ----
        nested_auc = float(np.mean(outer_aucs))
        nested_std = float(np.std(outer_aucs))
        logger.info("nested_cv_done", true_auc=round(nested_auc, 4), std=round(nested_std, 4))
        mlflow.log_metric("true_nested_cv_auc", nested_auc)
        mlflow.log_metric("nested_cv_std", nested_std)

        # ---- Save OOF predictions ----
        oof_df = pd.DataFrame({"oof_prob": oof_preds, "label": y})
        os.makedirs(os.path.dirname(cfg_paths["predictions"]["baseline_oof"]), exist_ok=True)
        oof_df.to_parquet(cfg_paths["predictions"]["baseline_oof"], index=False)
        logger.info("oof_saved", path=cfg_paths["predictions"]["baseline_oof"], nan_count=int(np.isnan(oof_preds).sum()))

        # ---- Refit on 100% data with best overall parameters ----
        best_overall_score, best_overall_params = max(best_params_across_folds, key=lambda x: x[0])
        logger.info("refit_start", best_inner_auc=round(best_overall_score, 4), params=best_overall_params)

        mlflow.log_params({f"best_{k}": v for k, v in best_overall_params.items()})

        ngram_range_final: tuple[int, int] = tuple(cfg_params["tfidf"]["ngram_range"][best_overall_params["ngram_range_idx"]])  # type: ignore[arg-type]
        final_vectorizer = TfidfVectorizer(
            max_features=best_overall_params["max_features"],
            ngram_range=ngram_range_final,
            use_idf=best_overall_params["use_idf"],
        )
        tfidf_full = final_vectorizer.fit_transform(X_text)
        X_full_sp = hstack([tfidf_full, csr_matrix(X_meta)]).tocsr()

        cb_params_final: dict[str, Any] = cfg_params["catboost"]["static"].copy()
        cb_params_final["random_seed"] = cfg_params["experiment"]["random_state"]
        cb_params_final.update(
            {k: v for k, v in best_overall_params.items() if k not in {"max_features", "ngram_range_idx", "use_idf"}}
        )

        final_model = CatBoostClassifier(**cb_params_final)
        final_model.fit(X_full_sp, y, verbose=100)

        # ---- Save artifacts ----
        os.makedirs(os.path.dirname(cfg_paths["models"]["best_tfidf"]), exist_ok=True)
        joblib.dump(final_vectorizer, cfg_paths["models"]["best_tfidf"])
        final_model.save_model(cfg_paths["models"]["best_catboost_baseline"])
        logger.info(
            "artifacts_saved",
            vectorizer=cfg_paths["models"]["best_tfidf"],
            model=cfg_paths["models"]["best_catboost_baseline"],
        )

    logger.info("pipeline_done", pipeline="02_train_baseline")


if __name__ == "__main__":
    main()
