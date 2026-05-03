from typing import Any, Tuple, Optional
import numpy as np
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def objective_catboost(
    trial: optuna.Trial, 
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    cfg: dict[str, Any],
    cat_features: Optional[list[int]] = None
) -> float:
    # We do a single Train/Val split inside the CV train fold for early stopping
    X_tr, X_va, y_tr, y_va = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    # We assume 'meta_only' or some generic catboost config block is passed or available
    # For a general wrapper, we will pull from the trial object directly using passed boundaries
    # Wait, the prompt specifies: "tune iterations, depth, learning_rate, and l2_leaf_reg."
    
    # Let's extract search space from config or hardcode sensible defaults if not passed perfectly
    ss = cfg.get("catboost", {}).get("search_space", {
        "iterations": {"low": 200, "high": 1500},
        "depth": {"low": 4, "high": 10},
        "learning_rate": {"low": 0.01, "high": 0.3},
        "l2_leaf_reg": {"low": 1.0, "high": 10.0}
    })
    
    iterations = trial.suggest_int("iterations", ss["iterations"]["low"], ss["iterations"]["high"])
    depth = trial.suggest_int("depth", ss["depth"]["low"], ss["depth"]["high"])
    learning_rate = trial.suggest_float("learning_rate", float(ss["learning_rate"]["low"]), float(ss["learning_rate"]["high"]), log=True)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", float(ss["l2_leaf_reg"]["low"]), float(ss["l2_leaf_reg"]["high"]))
    
    cb_params = {
        "iterations": iterations,
        "depth": depth,
        "learning_rate": learning_rate,
        "l2_leaf_reg": l2_leaf_reg,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "verbose": 0,
        "random_seed": 42
    }
    
    model = CatBoostClassifier(**cb_params)
    
    model.fit(
        Pool(X_tr, y_tr, cat_features=cat_features),
        eval_set=Pool(X_va, y_va, cat_features=cat_features),
        early_stopping_rounds=50,
        verbose=0
    )
    
    preds = model.predict_proba(X_va)[:, 1]
    return float(roc_auc_score(y_va, preds))

def fit_predict_catboost(
    X_tr: np.ndarray, 
    y_tr: np.ndarray, 
    X_te: np.ndarray, 
    y_te: np.ndarray, 
    best_params: dict[str, Any], 
    cat_features: Optional[list[int]] = None
) -> Tuple[np.ndarray, float]:
    
    cb_params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "verbose": 0,
        "random_seed": 42
    }
    cb_params.update(best_params)
    
    model = CatBoostClassifier(**cb_params)
    model.fit(Pool(X_tr, y_tr, cat_features=cat_features), verbose=0)
    
    preds = model.predict_proba(X_te)[:, 1]
    auc = float(roc_auc_score(y_te, preds))
    
    return preds, auc

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

def objective_tfidf_catboost(
    trial: optuna.Trial, 
    X_train: Tuple[np.ndarray, np.ndarray], 
    y_train: np.ndarray, 
    cfg: dict[str, Any]
) -> float:
    X_text, X_meta = X_train
    # Train/Val split for early stopping
    idx_tr, idx_va = train_test_split(np.arange(len(y_train)), test_size=0.2, random_state=42, stratify=y_train)
    
    text_tr, text_va = X_text[idx_tr], X_text[idx_va]
    meta_tr, meta_va = X_meta[idx_tr], X_meta[idx_va]
    y_tr, y_va = y_train[idx_tr], y_train[idx_va]
    
    # TF-IDF params
    max_features = trial.suggest_int("max_features", cfg["tfidf"]["max_features"]["low"], cfg["tfidf"]["max_features"]["high"], step=cfg["tfidf"]["max_features"]["step"])
    ngram_idx = trial.suggest_categorical("ngram_range_idx", [0, 1])
    ngram_range = tuple(cfg["tfidf"]["ngram_range"][ngram_idx])
    use_idf = trial.suggest_categorical("use_idf", cfg["tfidf"]["use_idf"])
    
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, use_idf=use_idf)
    tfidf_tr = vectorizer.fit_transform(text_tr)
    tfidf_va = vectorizer.transform(text_va)
    
    X_tr_sp = hstack([tfidf_tr, csr_matrix(meta_tr)]).tocsr()
    X_va_sp = hstack([tfidf_va, csr_matrix(meta_va)]).tocsr()
    
    ss = cfg["catboost"]["search_space"]
    cb_params = {
        "iterations": trial.suggest_int("iterations", ss["iterations"]["low"], ss["iterations"]["high"]),
        "depth": trial.suggest_int("depth", ss["depth"]["low"], ss["depth"]["high"]),
        "learning_rate": trial.suggest_float("learning_rate", ss["learning_rate"]["low"], ss["learning_rate"]["high"], log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", ss["l2_leaf_reg"]["low"], ss["l2_leaf_reg"]["high"]),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "verbose": 0,
        "random_seed": 42
    }
    
    model = CatBoostClassifier(**cb_params)
    model.fit(X_tr_sp, y_tr, eval_set=Pool(X_va_sp, y_va), early_stopping_rounds=30, verbose=0)
    
    preds = model.predict_proba(X_va_sp)[:, 1]
    return float(roc_auc_score(y_va, preds))

def fit_predict_tfidf_catboost(
    X_tr: Tuple[np.ndarray, np.ndarray], 
    y_tr: np.ndarray, 
    X_te: Tuple[np.ndarray, np.ndarray], 
    y_te: np.ndarray, 
    best_params: dict[str, Any],
    cfg: dict[str, Any]
) -> Tuple[np.ndarray, float]:
    X_text_tr, X_meta_tr = X_tr
    X_text_te, X_meta_te = X_te
    
    ngram_range = tuple(cfg["tfidf"]["ngram_range"][best_params["ngram_range_idx"]])
    vectorizer = TfidfVectorizer(
        max_features=best_params["max_features"],
        ngram_range=ngram_range,
        use_idf=best_params["use_idf"]
    )
    
    tfidf_tr = vectorizer.fit_transform(X_text_tr)
    tfidf_te = vectorizer.transform(X_text_te)
    
    X_tr_sp = hstack([tfidf_tr, csr_matrix(X_meta_tr)]).tocsr()
    X_te_sp = hstack([tfidf_te, csr_matrix(X_meta_te)]).tocsr()
    
    cb_params = {"loss_function": "Logloss", "eval_metric": "AUC", "verbose": 0, "random_seed": 42}
    cb_params.update({k: v for k, v in best_params.items() if k not in {"max_features", "ngram_range_idx", "use_idf"}})
    
    model = CatBoostClassifier(**cb_params)
    model.fit(X_tr_sp, y_tr, verbose=0)
    
    preds = model.predict_proba(X_te_sp)[:, 1]
    auc = float(roc_auc_score(y_te, preds))
    
    return preds, auc
