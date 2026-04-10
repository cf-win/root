import os
import json
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn

SEED = 2025
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = "/root/autodl-tmp/Beauty_5/Beauty_5_filled_label.csv"

SAVE_DIR = "/root/autodl-tmp/token/graph_token"
os.makedirs(SAVE_DIR, exist_ok=True)

TOKENS_OUT = os.path.join(SAVE_DIR, "Beauty_5_graph_tokens.pt")

MAP_OUT = os.path.join(SAVE_DIR, "Beauty_5_id_mappings_graph.json")

NUM_WINDOWS_LIMIT = 20          
WINDOW_LEN = 20000              
STRIDE = 10000                 
DROP_EMPTY_WINDOWS = True

HIDDEN_DIM = 64
GAT_HEADS = 2
GAT_DROPOUT = 0.1

CONCAT_ITEM_MEAN = True 

SAVE_SEQUENCE = True
AGGREGATE_IF_NOT_SEQUENCE = "mean"  

PRETRAINED_ENCODER_PATH = None 


def load_data_and_mappings(data_path: str):
    df = pd.read_csv(data_path, encoding="utf-8-sig")

    if "time" in df.columns:
        df = df.sort_values("time").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)

    user2idx: Dict[str, int] = {}
    idx2user: Dict[int, str] = {}
    item2idx: Dict[str, int] = {}
    idx2item: Dict[int, str] = {}

    for u, it in zip(df["user_id"].tolist(), df["item_id"].tolist()):
        if u not in user2idx:
            user2idx[u] = len(user2idx)
            idx2user[user2idx[u]] = u
        if it not in item2idx:
            item2idx[it] = len(item2idx)
            idx2item[item2idx[it]] = it

    df["u_idx"] = df["user_id"].map(user2idx).astype(int)
    df["i_idx"] = df["item_id"].map(item2idx).astype(int)

    return df, user2idx, item2idx, idx2user, idx2item

def build_sliding_windows(
    df: pd.DataFrame,
    window_len: int,
    stride: int,
    num_windows_limit: Optional[int] = None,
) -> List[pd.DataFrame]:
    n = len(df)
    window_len = max(1, int(window_len))
    stride = max(1, int(stride))

    windows: List[pd.DataFrame] = []
    starts = list(range(0, max(1, n - window_len + 1), stride))
    if num_windows_limit is not None:
        starts = starts[: int(num_windows_limit)]

    for s in starts:
        e = min(n, s + window_len)
        w = df.iloc[s:e].copy()
        windows.append(w)

    return windows

class SpatialGATEncoder(nn.Module):
    def __init__(self, num_users: int, num_items: int, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        out_feats = hidden_dim // num_heads

        self.user_emb = nn.Embedding(num_users, hidden_dim)
        self.item_emb = nn.Embedding(num_items, hidden_dim)

        self.conv = dglnn.HeteroGraphConv(
            {
                "interact": dglnn.GATConv(
                    in_feats=hidden_dim,
                    out_feats=out_feats,
                    num_heads=num_heads,
                    feat_drop=dropout,
                    attn_drop=dropout,
                    allow_zero_in_degree=True,
                ),
                "interact_rev": dglnn.GATConv(
                    in_feats=hidden_dim,
                    out_feats=out_feats,
                    num_heads=num_heads,
                    feat_drop=dropout,
                    attn_drop=dropout,
                    allow_zero_in_degree=True,
                ),
            },
            aggregate="sum",
        )

    def forward(self, g: dgl.DGLHeteroGraph) -> torch.Tensor:
        u_gid = g.nodes["user"].data["global_id"].to(torch.long)
        i_gid = g.nodes["item"].data["global_id"].to(torch.long)

        h0 = {
            "user": self.user_emb(u_gid),  
            "item": self.item_emb(i_gid),  
        }

        h = self.conv(g, h0)

        h_user = h["user"].flatten(1)
        h_item = h["item"].flatten(1)

        if CONCAT_ITEM_MEAN:
            item_mean = h_item.mean(dim=0, keepdim=True)                 
            item_expand = item_mean.repeat(h_user.size(0), 1)           
            user_repr = torch.cat([h_user, item_expand], dim=1)          
        else:
            user_repr = h_user                                           

        return user_repr


if __name__ == "__main__":
    df, user2idx, item2idx, idx2user, idx2item = load_data_and_mappings(DATA_PATH)

    mappings = {
        "user2idx": user2idx,
        "item2idx": item2idx,
        "idx2user": {str(k): v for k, v in idx2user.items()},
        "idx2item": {str(k): v for k, v in idx2item.items()},
        "note": "Token dict keys use RAW user_id strings. Graph encoder uses global indices via node.data['global_id'].",
        "concat_item_mean": CONCAT_ITEM_MEAN,
        "hidden_dim": HIDDEN_DIM,
        "gat_heads": GAT_HEADS,
        "window_len": WINDOW_LEN,
        "stride": STRIDE,
    }
    with open(MAP_OUT, "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved mappings to: {MAP_OUT}")

    windows = build_sliding_windows(df, WINDOW_LEN, STRIDE, NUM_WINDOWS_LIMIT)
    if DROP_EMPTY_WINDOWS:
        windows = [w for w in windows if len(w) > 0]
    num_windows = len(windows)
    print(f"✅ Built sliding windows: {num_windows} (window_len={WINDOW_LEN}, stride={STRIDE})")

    num_users = len(user2idx)
    num_items = len(item2idx)

    encoder = SpatialGATEncoder(num_users, num_items, HIDDEN_DIM, GAT_HEADS, GAT_DROPOUT).to(DEVICE)
    encoder.eval()

    if PRETRAINED_ENCODER_PATH and os.path.isfile(PRETRAINED_ENCODER_PATH):
        ck = torch.load(PRETRAINED_ENCODER_PATH, map_location="cpu")
        encoder.load_state_dict(ck, strict=True)
        print(f"✅ Loaded pretrained spatial encoder: {PRETRAINED_ENCODER_PATH}")
    else:
        print("⚠️ No pretrained spatial encoder loaded (random-init). For best quality, pretrain then freeze.")

    token_dim = (2 * HIDDEN_DIM) if CONCAT_ITEM_MEAN else HIDDEN_DIM
    zero_vec = torch.zeros(token_dim, dtype=torch.float32)

    user_seq: Dict[str, List[torch.Tensor]] = defaultdict(list)
    last_win_idx: Dict[str, int] = {}  

    with torch.no_grad():
        for win_id, win in enumerate(windows):
            u_global = win["u_idx"].unique().tolist()
            i_global = win["i_idx"].unique().tolist()

            if len(u_global) == 0 or len(i_global) == 0:
                print(f"⚠️ window {win_id} has no nodes, skipping")
                continue

            u_local_map = {gid: j for j, gid in enumerate(u_global)}
            i_local_map = {gid: j for j, gid in enumerate(i_global)}

            src_u = torch.tensor([u_local_map[g] for g in win["u_idx"].tolist()], dtype=torch.long)
            dst_i = torch.tensor([i_local_map[g] for g in win["i_idx"].tolist()], dtype=torch.long)

            g = dgl.heterograph(
                {
                    ("user", "interact", "item"): (src_u, dst_i),
                    ("item", "interact_rev", "user"): (dst_i, src_u),
                },
                num_nodes_dict={"user": len(u_global), "item": len(i_global)},
            ).to(DEVICE)

            g.nodes["user"].data["global_id"] = torch.tensor(u_global, dtype=torch.long, device=DEVICE)
            g.nodes["item"].data["global_id"] = torch.tensor(i_global, dtype=torch.long, device=DEVICE)

            win_user_emb = encoder(g).detach().cpu()

            for local_idx, u_gid in enumerate(u_global):
                u_raw = str(idx2user[int(u_gid)])
                emb = win_user_emb[local_idx].float()

                if u_raw in last_win_idx:
                    prev = last_win_idx[u_raw]
                    gap = win_id - prev - 1
                    if gap > 0:
                        user_seq[u_raw].extend([zero_vec.clone() for _ in range(gap)])
                else:
                    if win_id > 0:
                        user_seq[u_raw].extend([zero_vec.clone() for _ in range(win_id)])

                user_seq[u_raw].append(emb)
                last_win_idx[u_raw] = win_id

            print(f"✓ window {win_id} processed (users={len(u_global)}, items={len(i_global)})")

    for u_raw, seq_list in user_seq.items():
        if len(seq_list) < num_windows:
            seq_list.extend([zero_vec.clone() for _ in range(num_windows - len(seq_list))])

    all_users = df["user_id"].astype(str).unique().tolist()

    if SAVE_SEQUENCE:
        default_token = torch.stack([zero_vec.clone() for _ in range(num_windows)], dim=0)  # [T,D]
    else:
        default_token = zero_vec.clone()  # [D]

    num_missing = 0
    for u in all_users:
        if u not in user_seq or len(user_seq[u]) == 0:
            num_missing += 1
            if SAVE_SEQUENCE:
                user_seq[u] = [zero_vec.clone() for _ in range(num_windows)]
            else:
                user_seq[u] = [zero_vec.clone() for _ in range(num_windows)]

    print(f"✅ Filled missing users for graph tokens: {num_missing} users were absent and set to zero.")
    print(f"✅ Now user_seq users={len(user_seq)} (should match train users={len(all_users)})")

    user_graph_tokens: Dict[str, torch.Tensor] = {}
    for u_raw, seq_list in user_seq.items():
        if len(seq_list) == 0:
            continue
        seq = torch.stack(seq_list, dim=0) 
        if SAVE_SEQUENCE:
            user_graph_tokens[u_raw] = seq
        else:
            if AGGREGATE_IF_NOT_SEQUENCE == "mean":
                user_graph_tokens[u_raw] = seq.mean(dim=0)
            else:
                raise ValueError("Unsupported aggregate mode")

    torch.save(user_graph_tokens, TOKENS_OUT)
    print(f"✅ Graph tokens saved to: {TOKENS_OUT}")
    print(f"✅ users={len(user_graph_tokens)} | token_dim={token_dim} | windows={num_windows} | save_sequence={SAVE_SEQUENCE}")
