import os
import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 2025
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Configuration
DATASET = "Beauty_5"
DATA_DIR = f"/root/autodl-tmp/{DATASET}"
SAVE_DIR = "/root/autodl-tmp/token/behavior_token"
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_SEQ_LEN = 50
HIDDEN_DIM = 32  # Match RQ-VAE's cf_emb dimension
NUM_HEADS = 2
NUM_LAYERS = 2
DROPOUT = 0.1

TRAIN_SASREC = True
TRAIN_EPOCHS = 5
TRAIN_BATCH_SIZE = 256
TRAIN_LR = 1e-4
TRAIN_WEIGHT_DECAY = 1e-5
PRINT_EVERY = 200
GRAD_CLIP_NORM = 1.0

TOKENS_OUT = os.path.join(SAVE_DIR, f"{DATASET}_behavior_tokens.pt")
MAP_OUT = os.path.join(SAVE_DIR, f"{DATASET}_id_mappings_behavior.json")
ITEM_EMB_OUT = os.path.join(SAVE_DIR, f"{DATASET}-{HIDDEN_DIM}d-sasrec.pt")

def load_user_sequences_from_json(
    data_dir: str,
    dataset: str,
    max_seq_len: int = 50,
) -> Tuple[Dict[str, List[int]], int, Dict[str, int], Dict[int, str]]:
    """
    Load user sequences from standard JSON format.
    Returns: (user_seqs, num_items, user2idx, idx2user)
    
    Note: item indices in inter.json are already 0-indexed.
    We add 1 to make them 1-indexed (0 is padding for SASRec).
    """
    inter_file = os.path.join(data_dir, f"{dataset}.inter.json")
    item_file = os.path.join(data_dir, f"{dataset}.item.json")
    
    with open(inter_file, "r") as f:
        inter_data = json.load(f)
    with open(item_file, "r") as f:
        item_data = json.load(f)
    
    num_items = len(item_data)
    print(f"Loaded {len(inter_data)} users, {num_items} items")
    
    # Build user mapping
    user2idx: Dict[str, int] = {}
    idx2user: Dict[int, str] = {}
    
    user_seqs: Dict[str, List[int]] = {}
    
    for user_idx_str, item_seq in inter_data.items():
        user_idx = int(user_idx_str)
        user2idx[user_idx_str] = user_idx
        idx2user[user_idx] = user_idx_str
        
        # Convert 0-indexed to 1-indexed (0 is padding)
        seq = [item_idx + 1 for item_idx in item_seq]
        user_seqs[user_idx_str] = seq[-max_seq_len:]
    
    return user_seqs, num_items, user2idx, idx2user

class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        hidden_dim: int = 64,
        max_len: int = 50,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].fill_(0)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: [B, T] with 0 padding
        return: [B, H] (last valid position embedding)
        """
        B, T = seq.shape
        pos = torch.arange(T, device=seq.device).unsqueeze(0).expand(B, T)

        pad_mask = (seq == 0)
        all_pad = pad_mask.all(dim=1)
        if all_pad.any():
            pad_mask = pad_mask.clone()
            pad_mask[all_pad, -1] = False

        x = self.item_emb(seq) + self.pos_emb(pos)
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        x = self.dropout(x)

        out = self.encoder(x, src_key_padding_mask=pad_mask)

        seq_len = (seq != 0).sum(dim=1) - 1
        seq_len = torch.clamp(seq_len, min=0)

        beh = out[torch.arange(B, device=seq.device), seq_len]
        return beh


class NextItemDataset(torch.utils.data.Dataset):
    def __init__(self, user_seqs: Dict[str, List[int]], max_len: int):
        self.max_len = max_len
        self.samples: List[Tuple[List[int], int]] = []

        for _, seq in user_seqs.items():
            if len(seq) < 2:
                continue
            for t in range(1, len(seq)):
                prefix = seq[:t]
                nxt = seq[t]
                self.samples.append((prefix, nxt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        prefix, nxt = self.samples[idx]
        x = torch.zeros(self.max_len, dtype=torch.long)
        prefix = prefix[-self.max_len:]
        x[-len(prefix):] = torch.tensor(prefix, dtype=torch.long)
        y = torch.tensor(nxt, dtype=torch.long)
        return x, y


def train_sasrec_minimal(model: SASRec, user_seqs: Dict[str, List[int]], num_items: int):
    ds = NextItemDataset(user_seqs, MAX_SEQ_LEN)
    loader = torch.utils.data.DataLoader(ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

    model.to(DEVICE)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WEIGHT_DECAY)

    step = 0
    for ep in range(TRAIN_EPOCHS):
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            beh = model(x)  # [B,H]
            logits = beh @ model.item_emb.weight.t()  # [B, num_items+1]
            loss = F.cross_entropy(logits, y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optim.step()

            step += 1
            if step % PRINT_EVERY == 0:
                with torch.no_grad():
                    acc = (logits.argmax(dim=-1) == y).float().mean().item()
                print(f"[TRAIN] ep={ep} step={step} loss={loss.item():.4f} acc={acc:.4f}")

    model.eval()
    print("✅ SASRec minimal training finished.")


if __name__ == "__main__":
    user_seqs, num_items, user2idx, idx2user = load_user_sequences_from_json(
        DATA_DIR, DATASET, MAX_SEQ_LEN
    )

    mappings = {
        "user2idx": user2idx,
        "idx2user": {str(k): v for k, v in idx2user.items()},
        "note": "item indices are 1..N (0 is padding). Item idx in JSON is 0-indexed, we add 1 for SASRec.",
        "num_items": num_items,
        "max_seq_len": MAX_SEQ_LEN,
        "hidden_dim": HIDDEN_DIM,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
    }
    with open(MAP_OUT, "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved mappings to: {MAP_OUT}")

    model = SASRec(
        num_items=num_items,
        hidden_dim=HIDDEN_DIM,
        max_len=MAX_SEQ_LEN,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    if TRAIN_SASREC:
        train_sasrec_minimal(model, user_seqs, num_items)
    else:
        model.to(DEVICE)
        model.eval()
        print("⚠️ TRAIN_SASREC=False: behavior tokens come from random-init SASRec (not recommended).")

    behavior_tokens: Dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for uid_raw, seq in user_seqs.items():
            seq_tensor = torch.zeros(MAX_SEQ_LEN, dtype=torch.long, device=DEVICE)
            seq = seq[-MAX_SEQ_LEN:]
            if len(seq) > 0:
                seq_tensor[-len(seq):] = torch.tensor(seq, dtype=torch.long, device=DEVICE)

            emb = model(seq_tensor.unsqueeze(0))  # [1,H]
            behavior_tokens[str(uid_raw)] = emb.squeeze(0).detach().cpu()

    torch.save(behavior_tokens, TOKENS_OUT)
    print(f"✅ Behavior tokens saved to: {TOKENS_OUT}")
    print(f"✅ users={len(behavior_tokens)} | dim={HIDDEN_DIM}")

    # Save item embeddings for RQ-VAE CF loss
    # Item embedding indices: 1..num_items in SASRec, 0..num_items-1 in item.json
    # So item_emb[1] corresponds to item_idx 0 in item.json
    item_emb = model.item_emb.weight.detach().cpu()  # [num_items+1, hidden_dim]
    item_emb = item_emb[1:]  # Remove padding, now [num_items, hidden_dim]
    torch.save(item_emb, ITEM_EMB_OUT)
    print(f"✅ Item embeddings for RQ-VAE saved to: {ITEM_EMB_OUT}")
    print(f"✅ items={item_emb.shape[0]} | dim={item_emb.shape[1]}")
