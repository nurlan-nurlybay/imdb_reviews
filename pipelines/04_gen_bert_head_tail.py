"""
Pipeline 04: DistilBERT Head/Tail 128/128 Embeddings
Extracts frozen [CLS] hidden states using a 128/128 split for long reviews.
"""
import os
import torch
import pandas as pd
import structlog
from transformers import AutoModel, AutoTokenizer

from imdb.utils.config import load_config
from imdb.utils.logging import setup_logger
from imdb.features.bert import head_tail_tokenize, extract_bert_embeddings

setup_logger("logs/04_gen_bert_head_tail.log", terminal_level=logging.INFO, file_level=logging.DEBUG)
logger = structlog.get_logger(__name__)

def main() -> None:
    logger.info("pipeline_start", pipeline="04_gen_bert_head_tail")
    cfg_paths = load_config("configs/paths.yaml")
    cfg_params = load_config("configs/params.yaml")
    
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

    logger.info("tokenizing_128_128_logic")
    input_ids, attention_mask = head_tail_tokenize(texts, tokenizer, max_len=256)

    logger.info("extracting_embeddings")
    embeddings_np = extract_bert_embeddings(model, input_ids, attention_mask, batch_size, device)
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
