import os
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

import structlog

logger = structlog.get_logger(__name__)

def load_scaled_bert_and_meta(
    cfg_paths: dict[str, Any],
    cfg_params: dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads BERT embeddings and meta features, scales them, and returns (X, y).
    """
    logger.info("loading_data")
    df_meta = pd.read_parquet(cfg_paths["data"]["processed"])
    df_bert = pd.read_parquet(cfg_paths["data"]["bert_embeddings"])
    
    target_col = cfg_params["data"]["target_col"]
    meta_cols = cfg_params["data"]["meta_cols"].copy()
    
    if 'token_count' not in df_meta.columns:
        logger.info("calculating_token_count")
        tokenizer = AutoTokenizer.from_pretrained(cfg_params["bert"]["model_name"])
        tokens = tokenizer(df_meta['review_lemmatized'].astype(str).tolist(), add_special_tokens=True, truncation=False)
        df_meta['token_count'] = [len(t) for t in tokens['input_ids']]
        
    meta_cols.append('token_count')
    
    for col in ['word_count', 'uppercase_count', 'token_count']:
        df_meta[col] = np.log1p(df_meta[col])
        
    bert_features = [c for c in df_bert.columns if c != target_col]
    X_meta = df_meta[meta_cols].to_numpy()
    X_bert = df_bert[bert_features].to_numpy()
    X = np.hstack([X_bert, X_meta]).astype(np.float32)
    y = df_bert[target_col].to_numpy().astype(np.float32)
    
    # Scale entire dataset. Note: usually we scale inside CV.
    # To strictly follow the "speedrun" prompt's logic of scaling Train, we'll scale it all.
    # Actually, proper Nested CV requires scaling inside the fold!
    # But for simplicity, we will just apply StandardScaler here on the whole thing or return unscaled.
    # Returning unscaled is safer to avoid leakage. We will scale inside fit_predict.
    
    return X, y
