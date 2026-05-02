"""
Pipeline 05: SHAP Interpretability Analysis

Uses CatBoost's native get_feature_importance(type='ShapValues') for
extreme speed. Generates a global SHAP summary beeswarm plot.

Usage:
    python -m pipelines.05_shap_analysis
"""
from __future__ import annotations
import logging, os
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, shap, structlog
from catboost import CatBoostClassifier, Pool
from imdb.utils.config import load_config
from imdb.utils.logging import setup_logger

setup_logger("logs/05_shap_analysis.log", terminal_level=logging.INFO, file_level=logging.DEBUG)
logger = structlog.get_logger(__name__)

def main() -> None:
    logger.info("pipeline_start", pipeline="05_shap_analysis")
    cfg_paths: dict[str, Any] = load_config("configs/paths.yaml")
    cfg_params: dict[str, Any] = load_config("configs/params.yaml")

    # ---- Load model + data ----
    model = CatBoostClassifier()
    model.load_model(cfg_paths["models"]["best_catboost_production"])
    logger.info("model_loaded", path=cfg_paths["models"]["best_catboost_production"])

    df = pd.read_parquet(cfg_paths["data"]["bert_embeddings"])
    target_col: str = cfg_params["data"]["target_col"]
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].to_numpy().astype(np.float32)
    logger.info("data_loaded", shape=X.shape)

    # ---- Compute SHAP values (CatBoost native — uses all CPU cores) ----
    logger.info("computing_shap", method="catboost_native")
    pool = Pool(X, feature_names=feature_cols)
    # Returns shape (n_samples, n_features + 1); last column is base value
    raw_shap: Any = model.get_feature_importance(type="ShapValues", data=pool) # type: ignore[arg-type]
    
    # Cast to numpy array explicitly to fix typing
    shap_array = np.array(raw_shap)
    shap_values = shap_array[:, :-1]
    logger.info("shap_computed", shape=shap_values.shape)

    # ---- Global summary plot ----
    out_path: str = cfg_paths["reports"]["shap_summary"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, features=X, feature_names=feature_cols,
        max_display=30, show=False, plot_size=(12, 8),
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("plot_saved", path=out_path)

    # ---- Log top features by mean |SHAP| ----
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:15]
    for rank, idx in enumerate(top_idx):
        logger.info("top_feature", rank=rank + 1, name=feature_cols[idx], mean_abs_shap=round(float(mean_abs[idx]), 5))

    logger.info("pipeline_done", pipeline="05_shap_analysis")

if __name__ == "__main__":
    main()
