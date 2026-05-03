from typing import Any, Tuple
import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def objective_logreg(
    trial: optuna.Trial, 
    X_train: np.ndarray, 
    y_train: np.ndarray
) -> float:
    # Use a small internal split for HPO evaluation to prevent overfitting to the outer fold
    X_tr, X_va, y_tr, y_va = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
    
    # Scale inside the CV to prevent leakage
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    
    C = trial.suggest_float("C", 1e-4, 1e2, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    
    model = LogisticRegression(
        penalty='elasticnet', 
        solver='saga', 
        C=C, 
        l1_ratio=l1_ratio, 
        max_iter=1000, 
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_tr_s, y_tr)
    preds = model.predict_proba(X_va_s)[:, 1]
    
    return float(roc_auc_score(y_va, preds))

def fit_predict_logreg(
    X_tr: np.ndarray, 
    y_tr: np.ndarray, 
    X_te: np.ndarray, 
    y_te: np.ndarray, 
    best_params: dict[str, Any]
) -> Tuple[np.ndarray, float]:
    
    # Scale data properly for the final outer fold evaluation
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    model = LogisticRegression(
        penalty='elasticnet', 
        solver='saga', 
        C=best_params["C"], 
        l1_ratio=best_params["l1_ratio"], 
        max_iter=1000, 
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_tr_s, y_tr)
    preds = model.predict_proba(X_te_s)[:, 1]
    auc = float(roc_auc_score(y_te, preds))
    
    return preds, auc
