import os
from typing import Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

def objective_mlp(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray, cfg: dict[str, Any], device: torch.device) -> float:
    # Scale inside CV
    X_tr, X_va, y_tr, y_va = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    
    X_tr_t = torch.tensor(X_tr_s, device=device)
    y_tr_t = torch.tensor(y_tr, device=device)
    X_va_t = torch.tensor(X_va_s, device=device)
    y_va_t = torch.tensor(y_va, device=device)

    ss = cfg["neural_classifiers"]["mlp"]
    n_layers = trial.suggest_int("n_layers", ss["n_layers"]["low"], ss["n_layers"]["high"])
    n_units = trial.suggest_categorical("n_units", ss["n_units"])
    dropout = trial.suggest_float("dropout", float(ss["dropout"]["low"]), float(ss["dropout"]["high"]))
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
        indices = torch.randperm(len(X_tr_t), device=device)
        for i in range(0, len(X_tr_t), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_tr_t[batch_idx]
            y_batch = y_tr_t[batch_idx]
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_logits = model(X_va_t)
            val_preds = torch.sigmoid(val_logits).cpu().numpy()
            
        val_auc = float(roc_auc_score(y_va, val_preds))
        
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    return best_auc

def fit_predict_mlp(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, best_params: dict[str, Any], cfg: dict[str, Any], device: torch.device) -> Tuple[np.ndarray, float]:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    X_tr_t = torch.tensor(X_train_s, device=device)
    y_tr_t = torch.tensor(y_train, device=device)
    X_te_t = torch.tensor(X_test_s, device=device)
    
    ss = cfg["neural_classifiers"]["mlp"]
    batch_size = ss["batch_size"]
    epochs = ss["epochs"]
    
    model = DynamicMLP(X_train.shape[1], best_params["n_layers"], best_params["n_units"], best_params["dropout"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=best_params["lr"])
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_tr_t), device=device)
        for i in range(0, len(X_tr_t), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_tr_t[batch_idx]
            y_batch = y_tr_t[batch_idx]
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        test_logits = model(X_te_t)
        test_preds = torch.sigmoid(test_logits).cpu().numpy()
        
    auc = float(roc_auc_score(y_test, test_preds))
    return test_preds, auc

def train_full_mlp(X: np.ndarray, y: np.ndarray, best_params: dict[str, Any], cfg: dict[str, Any], device: torch.device) -> Tuple[nn.Module, StandardScaler]:
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_t = torch.tensor(X_s, device=device)
    y_t = torch.tensor(y, device=device)
    
    ss = cfg["neural_classifiers"]["mlp"]
    batch_size = ss["batch_size"]
    epochs = ss["epochs"]
    
    model = DynamicMLP(X.shape[1], best_params["n_layers"], best_params["n_units"], best_params["dropout"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=best_params["lr"])
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_t), device=device)
        for i in range(0, len(X_t), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_t[batch_idx]
            y_batch = y_t[batch_idx]
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
    return model, scaler
