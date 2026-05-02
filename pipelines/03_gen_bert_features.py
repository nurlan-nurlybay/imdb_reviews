"""
Pipeline 03: DistilBERT [CLS] Token Embedding Extraction

Usage:
    python -m pipelines.03_gen_bert_features
"""
from __future__ import annotations
import logging, os
from typing import Any
import numpy as np, pandas as pd, structlog, torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from imdb.utils.config import load_config
from imdb.utils.logging import setup_logger

setup_logger("logs/03_gen_bert_features.log", terminal_level=logging.INFO, file_level=logging.DEBUG)
logger = structlog.get_logger(__name__)

class TextDataset(Dataset): # type: ignore[type-arg]
    def __init__(self, encodings: Any) -> None:
        self.encodings = encodings
    def __len__(self) -> int:
        return self.encodings["input_ids"].shape[0] # type: ignore[no-any-return, index]
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {key: val[idx] for key, val in self.encodings.items()} # type: ignore[no-any-return, attr-defined]

def extract_cls_embeddings(model: PreTrainedModel, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    all_emb: list[np.ndarray] = []
    with torch.no_grad(), torch.autocast(device_type="cuda"):
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            out = model(input_ids=ids, attention_mask=mask)
            cls = out.last_hidden_state[:, 0, :]
            all_emb.append(cls.float().cpu().numpy())
    return np.concatenate(all_emb, axis=0)

def main() -> None:
    logger.info("pipeline_start", pipeline="03_gen_bert_features")
    cfg_paths: dict[str, Any] = load_config("configs/paths.yaml")
    cfg_params: dict[str, Any] = load_config("configs/params.yaml")
    bert_cfg = cfg_params["bert"]
    model_name, batch_size, max_length = bert_cfg["model_name"], bert_cfg["batch_size"], bert_cfg["max_length"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("device", device=str(device))
    if torch.cuda.is_available():
        logger.info("gpu", name=torch.cuda.get_device_name(0), vram_gb=round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1))

    df = pd.read_parquet(cfg_paths["data"]["processed"])
    meta_cols: list[str] = cfg_params["data"]["meta_cols"]
    texts = df["review_lemmatized"].astype(str).tolist()
    meta_features = df[meta_cols].to_numpy().astype(np.float32)
    logger.info("data_loaded", n=len(texts), meta=len(meta_cols))

    logger.info("tokenizing", model=model_name, max_len=max_length)
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    logger.info("tokenized", shape=list(encodings["input_ids"].shape))

    dataset = TextDataset(encodings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    logger.info("loading_model", model=model_name, dtype="float16")
    model: PreTrainedModel = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to(device) # type: ignore[assignment]
    logger.info("model_on_device", params_m=round(sum(p.numel() for p in model.parameters()) / 1e6, 1))

    cls_embeddings = extract_cls_embeddings(model, dataloader, device)
    logger.info("extracted", shape=cls_embeddings.shape)

    del model
    torch.cuda.empty_cache()

    combined = np.concatenate([cls_embeddings, meta_features], axis=1)
    emb_cols = [f"cls_{i}" for i in range(cls_embeddings.shape[1])]
    out_df = pd.DataFrame(combined, columns=emb_cols + meta_cols)
    out_df[cfg_params["data"]["target_col"]] = df[cfg_params["data"]["target_col"]].values

    out_path = cfg_paths["data"]["bert_embeddings"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    logger.info("saved", path=out_path, shape=out_df.shape, mb=round(os.path.getsize(out_path) / 1e6, 1))
    logger.info("pipeline_done", pipeline="03_gen_bert_features")

if __name__ == "__main__":
    main()
