"""
python -m pipelines.02_train_baseline
"""
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import joblib
import optuna
import mlflow
import os
import structlog
import logging
from tqdm import tqdm
from imdb.utils.config import load_config
from imdb.utils.logging import setup_logger

setup_logger("logs/02_train_baseline.log", terminal_level=logging.INFO, file_level=logging.DEBUG)
logger = structlog.get_logger(__name__)

# --- CUSTOM TQDM PROGRESS BAR FOR OPTUNA ---
class TqdmOptunaCallback:
    def __init__(self, n_trials, desc):
        self.pbar = tqdm(total=n_trials, desc=desc, leave=False)
    def __call__(self, study, trial):
        self.pbar.update(1)
    def close(self):
        self.pbar.close()

def objective(trial, X_text, X_meta, y, cfg_params):
    """Inner CV Loop for Optuna Optimization."""
    
    # 1. Suggest Params
    max_features = trial.suggest_int("max_features", cfg_params["tfidf"]["max_features"]["low"], cfg_params["tfidf"]["max_features"]["high"], step=cfg_params["tfidf"]["max_features"]["step"])
    
    # Safely handle list of lists from YAML
    ngram_idx = trial.suggest_categorical("ngram_range_idx", [0, 1])
    ngram_range = tuple(cfg_params["tfidf"]["ngram_range"][ngram_idx])
    use_idf = trial.suggest_categorical("use_idf", cfg_params["tfidf"]["use_idf"])

    cb_params = cfg_params["catboost"]["static"].copy()
    cb_params.update({
        "iterations": trial.suggest_int("iterations", cfg_params["catboost"]["search_space"]["iterations"]["low"], cfg_params["catboost"]["search_space"]["iterations"]["high"]),
        "depth": trial.suggest_int("depth", cfg_params["catboost"]["search_space"]["depth"]["low"], cfg_params["catboost"]["search_space"]["depth"]["high"]),
        "learning_rate": trial.suggest_float("learning_rate", cfg_params["catboost"]["search_space"]["learning_rate"]["low"], cfg_params["catboost"]["search_space"]["learning_rate"]["high"], log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", cfg_params["catboost"]["search_space"]["l2_leaf_reg"]["low"], cfg_params["catboost"]["search_space"]["l2_leaf_reg"]["high"]),
    })

    # 2. INNER CV Loop (3 Folds for speed inside Optuna)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg_params["experiment"]["random_state"])
    inner_aucs = []

    for train_idx, val_idx in inner_cv.split(X_text, y):
        # Split
        text_train, text_val = X_text[train_idx], X_text[val_idx]
        meta_train, meta_val = X_meta[train_idx], X_meta[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # TF-IDF
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, use_idf=use_idf)
        tfidf_train = vectorizer.fit_transform(text_train)
        tfidf_val = vectorizer.transform(text_val)

        # Hstack & DENSE CONVERSION (The CPU/GPU Optimization trick!)
        X_train_dense = hstack([tfidf_train, csr_matrix(meta_train)]).toarray().astype(np.float32)
        X_val_dense = hstack([tfidf_val, csr_matrix(meta_val)]).toarray().astype(np.float32)

        # Train CatBoost
        model = CatBoostClassifier(**cb_params)
        model.fit(X_train_dense, y_train, eval_set=(X_val_dense, y_val), early_stopping_rounds=30, verbose=0)

        # Evaluate
        preds = model.predict_proba(X_val_dense)[:, 1]
        inner_aucs.append(roc_auc_score(y_val, preds))

    return float(np.mean(inner_aucs))

def main():
    logger.info("Initializing Nested CV Baseline Pipeline")
    cfg_paths = load_config("configs/paths.yaml")
    cfg_params = load_config("configs/params.yaml")
    
    mlflow.set_tracking_uri(cfg_paths["tracking"]["mlflow_uri"])
    mlflow.set_experiment(cfg_params["experiment"]["name"])

    df = pd.read_parquet(cfg_paths["data"]["processed"])
    
    X_text = df["review_lemmatized"].to_numpy()
    meta_cols = ['qm_density', 'em_density', 'uppercase_count', 'vader_compound', 'word_count', 'TTR']
    X_meta = df[meta_cols].to_numpy()
    y = df[cfg_params["data"]["target_col"]].to_numpy()

    # --- OUTER CV LOOP ---
    outer_cv = StratifiedKFold(n_splits=cfg_params["experiment"]["n_splits"], shuffle=True, random_state=cfg_params["experiment"]["random_state"])
    outer_aucs = []
    best_params_across_folds = []

    logger.info(f"Starting Nested CV with {cfg_params['experiment']['n_splits']} Outer Folds...")

    with mlflow.start_run(run_name="Nested_CV_Evaluation"):
        
        for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_text, y)):
            logger.info(f"--- Processing Outer Fold {outer_fold + 1}/{cfg_params['experiment']['n_splits']} ---")
            
            # 1. Outer Split
            X_text_train, X_text_test = X_text[train_idx], X_text[test_idx]
            X_meta_train, X_meta_test = X_meta[train_idx], X_meta[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # 2. Run Optuna strictly on Outer Train
            study = optuna.create_study(direction="maximize")
            tqdm_cb = TqdmOptunaCallback(cfg_params["experiment"]["n_trials"], desc=f"Fold {outer_fold+1} Optuna")
            
            # Mute optuna logger to keep terminal clean
            optuna.logging.set_verbosity(optuna.logging.WARNING) 
            
            study.optimize(
                lambda trial: objective(trial, X_text_train, X_meta_train, y_train, cfg_params), 
                n_trials=cfg_params["experiment"]["n_trials"],
                callbacks=[tqdm_cb]
            )
            tqdm_cb.close()

            # 3. Train on Outer Train with Best Params
            best_params = study.best_params
            best_params_across_folds.append((study.best_value, best_params))

            ngram_range = tuple(cfg_params["tfidf"]["ngram_range"][best_params["ngram_range_idx"]])
            vectorizer = TfidfVectorizer(max_features=best_params["max_features"], ngram_range=ngram_range, use_idf=best_params["use_idf"])
            
            tfidf_train = vectorizer.fit_transform(X_text_train)
            tfidf_test = vectorizer.transform(X_text_test)

            # DENSE CONVERSION
            X_train_dense = hstack([tfidf_train, csr_matrix(X_meta_train)]).toarray().astype(np.float32)
            X_test_dense = hstack([tfidf_test, csr_matrix(X_meta_test)]).toarray().astype(np.float32)

            cb_params = cfg_params["catboost"]["static"].copy()
            cb_params.update({k: v for k, v in best_params.items() if k not in ["max_features", "ngram_range_idx", "use_idf"]})
            
            model = CatBoostClassifier(**cb_params)
            model.fit(X_train_dense, y_train, verbose=0)

            # 4. Evaluate on unseen Outer Test
            preds = model.predict_proba(X_test_dense)[:, 1]
            outer_auc = float(roc_auc_score(y_test, preds))
            outer_aucs.append(outer_auc)
            
            logger.info(f"Fold {outer_fold + 1} Outer AUC: {outer_auc:.4f}")
            mlflow.log_metric(f"fold_{outer_fold+1}_auc", outer_auc)

        # --- NESTED CV COMPLETE ---
        final_nested_auc = float(np.mean(outer_aucs))
        logger.info(f"Nested CV Complete. True Unbiased AUC: {final_nested_auc:.4f}")
        mlflow.log_metric("true_nested_cv_auc", final_nested_auc)

        # --- PRODUCTION MODEL RETRAINING ---
        # Find the parameters that yielded the highest inner validation score
        best_overall_params = sorted(best_params_across_folds, key=lambda x: x[0], reverse=True)[0][1]
        logger.info("Retraining final production model on 100% of data...", best_params=best_overall_params)
        
        ngram_range = tuple(cfg_params["tfidf"]["ngram_range"][best_overall_params["ngram_range_idx"]])
        vectorizer = TfidfVectorizer(max_features=best_overall_params["max_features"], ngram_range=ngram_range, use_idf=best_overall_params["use_idf"])
        
        tfidf_full = vectorizer.fit_transform(X_text)
        X_full_dense = hstack([tfidf_full, csr_matrix(X_meta)]).toarray().astype(np.float32)

        cb_params = cfg_params["catboost"]["static"].copy()
        cb_params.update({k: v for k, v in best_overall_params.items() if k not in ["max_features", "ngram_range_idx", "use_idf"]})
        
        final_model = CatBoostClassifier(**cb_params)
        final_model.fit(X_full_dense, y, verbose=100) # Print some output for the final run

        # Save Artifacts
        os.makedirs(os.path.dirname(cfg_paths["models"]["best_tfidf"]), exist_ok=True)
        joblib.dump(vectorizer, cfg_paths["models"]["best_tfidf"])
        final_model.save_model(cfg_paths["models"]["best_catboost_baseline"])
        logger.info("Baseline Pipeline Artifacts Saved Successfully!")

if __name__ == "__main__":
    main()
