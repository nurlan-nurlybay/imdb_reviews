import torch
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedModel
from torch.utils.data import TensorDataset, DataLoader

def head_tail_tokenize(
    texts: List[str], 
    tokenizer: PreTrainedTokenizer, 
    max_len: int = 256
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenizes text using a head/tail truncation strategy for long sequences.
    
    Args:
        texts: List of strings to tokenize.
        tokenizer: The pretrained huggingface tokenizer.
        max_len: The maximum sequence length. Default 256 implies 127 head + 127 tail + CLS + SEP.
        
    Returns:
        input_ids_tensor, attention_mask_tensor
    """
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id
    
    half_len = (max_len - 2) // 2

    all_input_ids = []
    all_attention_masks = []

    for text in tqdm(texts, desc="Tokenizing", leave=False):
        ids = tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']
        
        if len(ids) > (max_len - 2):
            head = ids[:half_len]
            tail = ids[-half_len:]
            final_ids = [cls_token_id] + head + tail + [sep_token_id]
        else:
            final_ids = [cls_token_id] + ids + [sep_token_id]
            
        pad_len = max_len - len(final_ids)
        final_ids = final_ids + [pad_token_id] * pad_len
        attention_mask = [1] * (max_len - pad_len) + [0] * pad_len
        
        all_input_ids.append(final_ids)
        all_attention_masks.append(attention_mask)

    return torch.tensor(all_input_ids, dtype=torch.long), torch.tensor(all_attention_masks, dtype=torch.long)

def extract_bert_embeddings(
    model: PreTrainedModel, 
    input_ids: torch.Tensor, 
    attention_mask: torch.Tensor, 
    batch_size: int, 
    device: torch.device
) -> np.ndarray:
    """
    Extracts frozen [CLS] embeddings from a DistilBERT model.
    """
    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    
    with torch.no_grad(), torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16):
        for batch_ids, batch_mask in tqdm(dataloader, desc="Extracting", leave=False):
            batch_ids = batch_ids.to(device)
            batch_mask = batch_mask.to(device)
            
            outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
            hidden_states = outputs[0]
            cls_embeddings = hidden_states[:, 0, :].cpu().numpy().astype(np.float32)
            embeddings.append(cls_embeddings)

    return np.vstack(embeddings)
