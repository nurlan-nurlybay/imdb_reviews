"""
Pipeline 06: DistilBERT Head/Tail 128/128 Embeddings
Extracts frozen [CLS] hidden states using a 128/128 split for long reviews.
"""
from __future__ import annotations
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import structlog
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from imdb.utils.config import load_config
from imdb.utils.logging import setup_logger

setup_logger("logs/06_gen_bert_head_tail.log", terminal_level=logging.INFO, file_level=logging.DEBUG)
logger = structlog.get_logger(__name__)

def main() -> None:
    logger.info("pipeline_start", pipeline="06_gen_bert_head_tail")
    cfg_paths: dict[str, Any] = load_config("configs/paths.yaml")
    cfg_params: dict[str, Any] = load_config("configs/params.yaml")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("device", device=str(device))
    if torch.cuda.is_available():
        logger.info("gpu", name=torch.cuda.get_device_name(0), vram_gb=round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))

    # ---- Load Raw Data ----
    df_raw = pd.read_csv(cfg_paths["data"]["raw"])
    texts = df_raw["review"].astype(str).tolist()
    targets = (df_raw["sentiment"] == "positive").astype(int).tolist()
    
    logger.info("data_loaded", n_samples=len(texts))

    # ---- Tokenizer & Model ----
    model_name = cfg_params["bert"]["model_name"]
    batch_size = cfg_params["bert"]["batch_size"]
    logger.info("loading_model", model_name=model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_masks = []

    logger.info("tokenizing_128_128_logic")
    for text in tqdm(texts, desc="Tokenizing", leave=False):
        # Tokenize without truncation
        ids = tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']
        
        if len(ids) > 254:
            head = ids[:127]
            tail = ids[-127:]
            final_ids = [cls_token_id] + head + tail + [sep_token_id]
        else:
            final_ids = [cls_token_id] + ids + [sep_token_id]
            
        pad_len = 256 - len(final_ids)
        final_ids = final_ids + [pad_token_id] * pad_len
        attention_mask = [1] * (256 - pad_len) + [0] * pad_len
        
        all_input_ids.append(final_ids)
        all_attention_masks.append(attention_mask)

    input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long)
    attention_mask_tensor = torch.tensor(all_attention_masks, dtype=torch.long)

    dataset = TensorDataset(input_ids_tensor, attention_mask_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []

    logger.info("extracting_embeddings")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for batch_ids, batch_mask in tqdm(dataloader, desc="Extracting", leave=False):
            batch_ids = batch_ids.to(device)
            batch_mask = batch_mask.to(device)
            
            outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
            # DistilBERT returns hidden states as the first element
            hidden_states = outputs[0]
            # Take the [CLS] token representation
            cls_embeddings = hidden_states[:, 0, :].cpu().numpy().astype(np.float32)
            embeddings.append(cls_embeddings)

    embeddings_np = np.vstack(embeddings)
    logger.info("embeddings_generated", shape=embeddings_np.shape)

    # ---- Save to Parquet ----
    df_out = pd.DataFrame(embeddings_np, columns=[f"bert_{i}" for i in range(embeddings_np.shape[1])])
    df_out[cfg_params["data"]["target_col"]] = targets
    
    out_path = cfg_paths["data"]["bert_embeddings"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_parquet(out_path, index=False)
    
    logger.info("pipeline_done", saved_path=out_path)

if __name__ == "__main__":
    main()
