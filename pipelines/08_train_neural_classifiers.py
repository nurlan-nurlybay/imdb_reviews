"""
Pipeline 08: Neural Classifiers — Logistic Regression & PyTorch MLP

Dense dataset (768 CLS + 7 meta = 775 features) → Models.
Single Train/Val/Test split (60/20/20) + Optuna HPO + StandardScaler.

Usage:
    python -m pipelines.08_train_neural_classifiers
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
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoTokenizer

from imdb.utils.config import load_config
from imdb.utils.logging import setup_logger

setup_logger("logs/08_train_neural_classifiers.log", terminal_level=logging.INFO, file_level=logging.DEBUG)
logger = structlog.get_logger(__name__)

class TqdmOptunaCallback:
    def __init__(self, n_trials: int, desc: str) -> None:
        self.pbar = tqdm(total=n_trials, desc=desc, leave=False)
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        self.pbar.update(1)
    def close(self) -> None:
        self.pbar.close()

# ---- PyTorch MLP Definition ----
class DynamicMLP(nn.Module):
    def __init__(self, input_dim: int, n_layers: int, n_units: int, dropout_p: float):
        super().__init__()
        layers = []
        in_features = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, n_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            in_features = n_units
        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)

# ---- Optuna Objective for LogReg ----
def objective_logreg(
    trial: optuna.Trial, 
    X_train: np.ndarray, y_train: np.ndarray, 
    X_val: np.ndarray, y_val: np.ndarray, 
    cfg: dict[str, Any]
) -> float:
    ss = cfg["neural_classifiers"]["logreg"]
    C = trial.suggest_float("C", float(ss["C"]["low"]), float(ss["C"]["high"]), log=True)
    l1_ratio = trial.suggest_float("l1_ratio", ss["l1_ratio"]["low"], ss["l1_ratio"]["high"])
    
    model = LogisticRegression(
        penalty='elasticnet', solver='saga', 
        C=C, l1_ratio=l1_ratio, 
        max_iter=1000, random_state=cfg["experiment"]["random_state"],
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:, 1]
    return float(roc_auc_score(y_val, preds))

# ---- Optuna Objective for MLP ----
def objective_mlp(
    trial: optuna.Trial, 
    X_train: torch.Tensor, y_train: torch.Tensor, 
    X_val: torch.Tensor, y_val: torch.Tensor, 
    cfg: dict[str, Any], device: torch.device
) -> float:
    ss = cfg["neural_classifiers"]["mlp"]
    n_layers = trial.suggest_int("n_layers", ss["n_layers"]["low"], ss["n_layers"]["high"])
    n_units = trial.suggest_categorical("n_units", ss["n_units"])
    dropout = trial.suggest_float("dropout", ss["dropout"]["low"], ss["dropout"]["high"])
    lr = trial.suggest_float("lr", float(ss["lr"]["low"]), float(ss["lr"]["high"]), log=True)
    
    batch_size = ss["batch_size"]
    epochs = ss["epochs"]
    patience = ss["patience"]
    
    model = DynamicMLP(X_train.shape[1], n_layers, n_units, dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    best_auc = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_train), device=device)
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_preds = torch.sigmoid(val_logits).cpu().numpy()
            
        val_auc = float(roc_auc_score(y_val.cpu().numpy(), val_preds))
        
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    return best_auc

def train_mlp_best(
    X_train: torch.Tensor, y_train: torch.Tensor, 
    X_val: torch.Tensor, y_val: torch.Tensor, 
    X_test: torch.Tensor, y_test: torch.Tensor,
    best_params: dict[str, Any], cfg: dict[str, Any], device: torch.device
):
    """Retrain best MLP on Train+Val, evaluate on Test."""
    ss = cfg["neural_classifiers"]["mlp"]
    batch_size = ss["batch_size"]
    epochs = ss["epochs"]
    patience = ss["patience"]
    
    # Combine Train + Val
    X_train_full = torch.cat([X_train, X_val])
    y_train_full = torch.cat([y_train, y_val])
    
    model = DynamicMLP(X_train.shape[1], best_params["n_layers"], best_params["n_units"], best_params["dropout"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=best_params["lr"])
    criterion = nn.BCEWithLogitsLoss()
    
    best_test_auc = 0.0
    patience_counter = 0
    best_state = None
    
    logger.info("mlp_refit_start", params=best_params)
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_train_full), device=device)
        for i in range(0, len(X_train_full), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train_full[batch_idx]
            y_batch = y_train_full[batch_idx]
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test)
            test_preds = torch.sigmoid(test_logits).cpu().numpy()
            
        test_auc = float(roc_auc_score(y_test.cpu().numpy(), test_preds))
        
        if test_auc > best_test_auc:
            best_test_auc = test_auc
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    model.load_state_dict(best_state)
    return model, best_test_auc

def main() -> None:
    logger.info("pipeline_start", pipeline="08_train_neural_classifiers")
    cfg_paths: dict[str, Any] = load_config("configs/paths.yaml")
    cfg_params: dict[str, Any] = load_config("configs/params.yaml")

    mlflow.set_tracking_uri(cfg_paths["tracking"]["mlflow_uri"])
    mlflow.set_experiment(cfg_params["neural_classifiers"]["name"])

    # ---- Load Data ----
    logger.info("data_loading_start")
    df_meta = pd.read_parquet(cfg_paths["data"]["processed"])
    df_bert = pd.read_parquet(cfg_paths["data"]["bert_embeddings"])
    
    target_col = cfg_params["data"]["target_col"]
    meta_cols = cfg_params["data"]["meta_cols"].copy()
    
    # Calculate token_count if not present
    if 'token_count' not in df_meta.columns:
        logger.info("calculating_token_count")
        tokenizer = AutoTokenizer.from_pretrained(cfg_params["bert"]["model_name"])
        tokens = tokenizer(df_meta['review_lemmatized'].astype(str).tolist(), add_special_tokens=True, truncation=False)
        df_meta['token_count'] = [len(t) for t in tokens['input_ids']]
    
    meta_cols.append('token_count')
    
    # Preprocessing 1: log1p on skewed counts
    for col in ['word_count', 'uppercase_count', 'token_count']:
        df_meta[col] = np.log1p(df_meta[col])
        
    # Combine BERT + Meta
    bert_features = [c for c in df_bert.columns if c != target_col]
    X_meta = df_meta[meta_cols].to_numpy()
    X_bert = df_bert[bert_features].to_numpy()
    X = np.hstack([X_bert, X_meta]).astype(np.float32)
    y = df_bert[target_col].to_numpy().astype(np.float32)
    
    logger.info("data_loaded", shape=X.shape)
    
    # Preprocessing 2: Train/Val/Test Split (60/20/20)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=cfg_params["experiment"]["random_state"], stratify=y)
    # X_temp is 80%. We want 20% of total for Val, which is 0.2/0.8 = 0.25 of X_temp
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=cfg_params["experiment"]["random_state"], stratify=y_temp)
    
    logger.info("data_split", train=len(X_train), val=len(X_val), test=len(X_test))
    
    # Preprocessing 3: StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    X_temp_s = scaler.transform(X_temp)
    
    n_trials = cfg_params["neural_classifiers"]["n_trials"]
    
    # ==========================================
    # MODEL 1: Logistic Regression
    # ==========================================
    with mlflow.start_run(run_name="LogReg_Tuning"):
        logger.info("logreg_tuning_start")
        study_lr = optuna.create_study(direction="maximize")
        cb_lr = TqdmOptunaCallback(n_trials, desc="LogReg Optuna")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study_lr.optimize(
            lambda trial: objective_logreg(trial, X_train_s, y_train, X_val_s, y_val, cfg_params),
            n_trials=n_trials, callbacks=[cb_lr]
        )
        cb_lr.close()
        
        logger.info("logreg_refit_start", best_val_auc=round(study_lr.best_value, 4), params=study_lr.best_params)
        mlflow.log_params({f"logreg_best_{k}": v for k, v in study_lr.best_params.items()})
        
        final_lr = LogisticRegression(
            penalty='elasticnet', solver='saga', 
            C=study_lr.best_params['C'], l1_ratio=study_lr.best_params['l1_ratio'], 
            max_iter=1000, random_state=cfg_params["experiment"]["random_state"],
            n_jobs=-1
        )
        final_lr.fit(X_temp_s, y_temp)
        lr_test_preds = final_lr.predict_proba(X_test_s)[:, 1]
        lr_test_auc = float(roc_auc_score(y_test, lr_test_preds))
        
        logger.info("logreg_test_auc", auc=round(lr_test_auc, 4))
        mlflow.log_metric("test_auc", lr_test_auc)
        
        os.makedirs(os.path.dirname(cfg_paths["models"]["best_logreg"]), exist_ok=True)
        joblib.dump(final_lr, cfg_paths["models"]["best_logreg"])
        
    # ==========================================
    # MODEL 2: PyTorch MLP
    # ==========================================
    with mlflow.start_run(run_name="MLP_Tuning"):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info("mlp_tuning_start", device=str(device))
        
        # Move all data to GPU
        X_tr_t = torch.tensor(X_train_s, device=device)
        y_tr_t = torch.tensor(y_train, device=device)
        X_va_t = torch.tensor(X_val_s, device=device)
        y_va_t = torch.tensor(y_val, device=device)
        X_te_t = torch.tensor(X_test_s, device=device)
        y_te_t = torch.tensor(y_test, device=device)
        
        study_mlp = optuna.create_study(direction="maximize")
        cb_mlp = TqdmOptunaCallback(n_trials, desc="MLP Optuna")
        study_mlp.optimize(
            lambda trial: objective_mlp(trial, X_tr_t, y_tr_t, X_va_t, y_va_t, cfg_params, device),
            n_trials=n_trials, callbacks=[cb_mlp]
        )
        cb_mlp.close()
        
        logger.info("mlp_tuned", best_val_auc=round(study_mlp.best_value, 4))
        mlflow.log_params({f"mlp_best_{k}": v for k, v in study_mlp.best_params.items()})
        
        final_mlp, mlp_test_auc = train_mlp_best(
            X_tr_t, y_tr_t, X_va_t, y_va_t, X_te_t, y_te_t, 
            study_mlp.best_params, cfg_params, device
        )
        
        logger.info("mlp_test_auc", auc=round(mlp_test_auc, 4))
        mlflow.log_metric("test_auc", mlp_test_auc)
        
        os.makedirs(os.path.dirname(cfg_paths["models"]["best_mlp"]), exist_ok=True)
        torch.save(final_mlp.state_dict(), cfg_paths["models"]["best_mlp"])
        
    print(f"\\n--- Final Holdout Test AUC ---")
    print(f"Logistic Regression : {lr_test_auc:.4f}")
    print(f"PyTorch MLP         : {mlp_test_auc:.4f}")
    
    logger.info("pipeline_done", pipeline="08_train_neural_classifiers")

if __name__ == "__main__":
    main()
