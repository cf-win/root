from typing import Dict, List, Set

import torch
from transformers import LogitsProcessor


class _Node:
    def __init__(self):
        self.children: Dict[int, int] = {}
        self.is_end: bool = False


class TokenTrie:
    """Token-level trie for legal item-token decoding."""

    def __init__(self):
        self.nodes: List[_Node] = [_Node()]  # root at index 0

    def add(self, seq: List[int]):
        if not seq:
            return
        cur = 0
        for tok in seq:
            nxt = self.nodes[cur].children.get(tok)
            if nxt is None:
                nxt = len(self.nodes)
                self.nodes[cur].children[tok] = nxt
                self.nodes.append(_Node())
            cur = nxt
        self.nodes[cur].is_end = True


class SingleItemConstrainedLogitsProcessor(LogitsProcessor):
    """Constrain generation to exactly one legal item sequence."""

    def __init__(self, tokenizer, candidate_items: List[str]):
        self.tokenizer = tokenizer
        self.item_trie = TokenTrie()

        for item in candidate_items:
            ids_plain = tokenizer.encode(item, add_special_tokens=False)
            ids_space = tokenizer.encode(" " + item, add_special_tokens=False)
            if ids_plain:
                self.item_trie.add(ids_plain)
            if ids_space:
                self.item_trie.add(ids_space)

        self.stop_ids: Set[int] = set()
        if tokenizer.eos_token_id is not None:
            self.stop_ids.add(tokenizer.eos_token_id)
        for tid in tokenizer.encode("\n", add_special_tokens=False):
            self.stop_ids.add(tid)

    def _allowed(self, generated_ids: List[int]) -> List[int]:
        states: Set[int] = {0}
        for tok in generated_ids:
            next_states: Set[int] = set()
            for node_id in states:
                nxt = self.item_trie.nodes[node_id].children.get(tok)
                if nxt is not None:
                    next_states.add(nxt)
            if not next_states:
                return list(self.stop_ids) if self.stop_ids else [self.tokenizer.eos_token_id]
            states = next_states

        allowed: Set[int] = set()
        for node_id in states:
            node = self.item_trie.nodes[node_id]
            allowed.update(node.children.keys())
            if node.is_end:
                allowed.update(self.stop_ids)

        if not allowed:
            return list(self.stop_ids) if self.stop_ids else [self.tokenizer.eos_token_id]
        return list(allowed)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for row in range(input_ids.size(0)):
            seq = input_ids[row].tolist()
            allowed = self._allowed(seq)
            mask = torch.full_like(scores[row], float("-inf"))
            mask[allowed] = 0.0
            scores[row] = scores[row] + mask
        return scores


def build_recommendation_logits_processor(tokenizer, candidate_items: List[str]):
    if not candidate_items:
        return None
    return SingleItemConstrainedLogitsProcessor(tokenizer, candidate_items)
