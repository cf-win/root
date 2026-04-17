import os
import json
import random
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn

SEED = 2025
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET = "Beauty_5"
DATA_DIR = f"/root/autodl-tmp/{DATASET}"
DATA_PATH = os.path.join(DATA_DIR, f"{DATASET}_filled_label.csv")
ID_MAPPING_PATH = os.path.join(DATA_DIR, f"{DATASET}.id_mapping.json")

SAVE_DIR = "/root/autodl-tmp/token/graph_token"
os.makedirs(SAVE_DIR, exist_ok=True)

TOKENS_OUT = os.path.join(SAVE_DIR, f"{DATASET}_graph_tokens.pt")
MAP_OUT = os.path.join(SAVE_DIR, f"{DATASET}_id_mappings_graph.json")

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
RESERVED_TAIL = 2  # leave-one-out: reserve last 2 interactions for valid/test


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


def build_train_interactions(df: pd.DataFrame, reserved_tail: int = 2) -> pd.DataFrame:
    """Keep only training interactions per user (drop last reserved_tail events)."""
    if reserved_tail <= 0:
        return df.copy()

    work = df.copy().reset_index(drop=True)
    work["_ord"] = work.groupby("user_id").cumcount()
    work["_cnt"] = work.groupby("user_id")["user_id"].transform("size")
    train_df = work[work["_ord"] < (work["_cnt"] - reserved_tail)].copy()

    drop_cols = [c for c in ["_ord", "_cnt"] if c in train_df.columns]
    train_df = train_df.drop(columns=drop_cols).reset_index(drop=True)
    return train_df


def load_t2rec_user_mapping(mapping_path: str) -> Dict[str, str]:
    """
    Load raw_user_id -> T2Rec user_idx(str) mapping.
    T2Rec dataset uses user_idx string keys in inter.json; graph tokens should follow
    the same key space so `self.graph_tokens.get(str(user_id))` can hit.
    """
    if not os.path.isfile(mapping_path):
        print(f"⚠️ ID mapping not found at {mapping_path}; fallback to raw user_id keys.")
        return {}

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    user_id_to_idx = mapping.get("user_id_to_idx", {})
    if not isinstance(user_id_to_idx, dict) or len(user_id_to_idx) == 0:
        print(f"⚠️ user_id_to_idx missing in {mapping_path}; fallback to raw user_id keys.")
        return {}

    return {str(raw_uid): str(uidx) for raw_uid, uidx in user_id_to_idx.items()}


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
    df_full, user2idx, item2idx, idx2user, idx2item = load_data_and_mappings(DATA_PATH)
    df = build_train_interactions(df_full, reserved_tail=RESERVED_TAIL)
    raw_to_t2rec_uid = load_t2rec_user_mapping(ID_MAPPING_PATH)

    mappings = {
        "user2idx": user2idx,
        "item2idx": item2idx,
        "idx2user": {str(k): v for k, v in idx2user.items()},
        "idx2item": {str(k): v for k, v in idx2item.items()},
        "raw_to_t2rec_user_idx": raw_to_t2rec_uid,
        "token_user_key_space": "T2Rec user_idx string if id_mapping exists, otherwise raw user_id string",
        "note": "Graph encoder uses global indices via node.data['global_id']. Saved token keys are aligned to T2Rec user ids when mapping is available.",
        "concat_item_mean": CONCAT_ITEM_MEAN,
        "hidden_dim": HIDDEN_DIM,
        "gat_heads": GAT_HEADS,
        "window_len": WINDOW_LEN,
        "stride": STRIDE,
        "train_only": True,
        "reserved_tail": RESERVED_TAIL,
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
                u_key = raw_to_t2rec_uid.get(u_raw, u_raw)
                emb = win_user_emb[local_idx].float()

                if u_key in last_win_idx:
                    prev = last_win_idx[u_key]
                    gap = win_id - prev - 1
                    if gap > 0:
                        user_seq[u_key].extend([zero_vec.clone() for _ in range(gap)])
                else:
                    if win_id > 0:
                        user_seq[u_key].extend([zero_vec.clone() for _ in range(win_id)])

                user_seq[u_key].append(emb)
                last_win_idx[u_key] = win_id

            print(f"✓ window {win_id} processed (users={len(u_global)}, items={len(i_global)})")

    for _, seq_list in user_seq.items():
        if len(seq_list) < num_windows:
            seq_list.extend([zero_vec.clone() for _ in range(num_windows - len(seq_list))])

    all_users_raw = df_full["user_id"].astype(str).unique().tolist()
    all_users = [raw_to_t2rec_uid.get(u_raw, u_raw) for u_raw in all_users_raw]

    num_missing = 0
    for u_key in all_users:
        if u_key not in user_seq or len(user_seq[u_key]) == 0:
            num_missing += 1
            user_seq[u_key] = [zero_vec.clone() for _ in range(num_windows)]

    print(f"✅ Filled missing users for graph tokens: {num_missing} users were absent and set to zero.")
    print(f"✅ Now user_seq users={len(user_seq)} (should match train users={len(all_users)})")

    user_graph_tokens: Dict[str, torch.Tensor] = {}
    for u_key, seq_list in user_seq.items():
        if len(seq_list) == 0:
            continue
        seq = torch.stack(seq_list, dim=0)
        if SAVE_SEQUENCE:
            user_graph_tokens[u_key] = seq
        else:
            if AGGREGATE_IF_NOT_SEQUENCE == "mean":
                user_graph_tokens[u_key] = seq.mean(dim=0)
            else:
                raise ValueError("Unsupported aggregate mode")

    torch.save(user_graph_tokens, TOKENS_OUT)
    print(f"✅ Graph tokens saved to: {TOKENS_OUT}")
    print(f"✅ users={len(user_graph_tokens)} | token_dim={token_dim} | windows={num_windows} | save_sequence={SAVE_SEQUENCE}")
