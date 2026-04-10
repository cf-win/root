from typing import Dict, List, Set, Tuple

import torch
from transformers import LogitsProcessor


class _Node:
	def __init__(self):
		self.children: Dict[int, int] = {}
		self.is_end: bool = False


class TokenTrie:
	"""Token-level trie with explicit terminal flags for constrained decoding."""

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

	def next_tokens(self, node_id: int) -> List[int]:
		return list(self.nodes[node_id].children.keys())


def _find_subseq(seq: List[int], pattern: List[int]) -> int:
	if not pattern or len(pattern) > len(seq):
		return -1
	L = len(pattern)
	end = len(seq) - L + 1
	for i in range(end):
		if seq[i : i + L] == pattern:
			return i
	return -1


class TrieConstrainedLogitsProcessor(LogitsProcessor):
	"""Apply trie constraints only after 'Final Recommendation List:' appears."""

	def __init__(self, tokenizer, candidate_items: List[str], top_k: int):
		self.tokenizer = tokenizer
		self.top_k = top_k
		self.vocab_size = len(tokenizer)

		self.first_item_trie = TokenTrie()
		self.next_item_trie = TokenTrie()

		for item in candidate_items:
			ids0 = tokenizer.encode(" " + item, add_special_tokens=False)
			ids0_plain = tokenizer.encode(item, add_special_tokens=False)
			idsn = tokenizer.encode(", " + item, add_special_tokens=False)
			idsn_tight = tokenizer.encode("," + item, add_special_tokens=False)
			if ids0:
				self.first_item_trie.add(ids0)
			if ids0_plain:
				self.first_item_trie.add(ids0_plain)
			if idsn:
				self.next_item_trie.add(idsn)
			if idsn_tight:
				self.next_item_trie.add(idsn_tight)

		self.rec_anchor_ids = tokenizer.encode("Final Recommendation List:", add_special_tokens=False)

		self.stop_ids: Set[int] = set()
		if tokenizer.eos_token_id is not None:
			self.stop_ids.add(tokenizer.eos_token_id)
		for tid in tokenizer.encode("\n", add_special_tokens=False):
			self.stop_ids.add(tid)

	def _state_closure(self, states: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
		expanded = set(states)
		changed = True
		while changed:
			changed = False
			to_add = set()
			for pos, node in expanded:
				trie = self.first_item_trie if pos == 0 else self.next_item_trie
				if trie.nodes[node].is_end and pos + 1 < self.top_k:
					nxt_state = (pos + 1, 0)
					if nxt_state not in expanded:
						to_add.add(nxt_state)
			if to_add:
				expanded.update(to_add)
				changed = True
		return expanded

	def _allowed_after_anchor(self, rec_suffix_ids: List[int]) -> List[int]:
		states: Set[Tuple[int, int]] = {(0, 0)}
		states = self._state_closure(states)

		for tok in rec_suffix_ids:
			new_states: Set[Tuple[int, int]] = set()
			for pos, node in states:
				trie = self.first_item_trie if pos == 0 else self.next_item_trie
				nxt = trie.nodes[node].children.get(tok)
				if nxt is not None:
					new_states.add((pos, nxt))
			if not new_states:
				return list(self.stop_ids) if self.stop_ids else [self.tokenizer.eos_token_id]
			states = self._state_closure(new_states)

		allowed: Set[int] = set()
		for pos, node in states:
			trie = self.first_item_trie if pos == 0 else self.next_item_trie
			allowed.update(trie.next_tokens(node))
			if pos == self.top_k - 1 and trie.nodes[node].is_end:
				allowed.update(self.stop_ids)

		if not allowed:
			return list(self.stop_ids) if self.stop_ids else [self.tokenizer.eos_token_id]
		return list(allowed)

	def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
		# input_ids may have shape [batch*beams, 0] when using inputs_embeds; handle safely.
		for row in range(input_ids.size(0)):
			seq = input_ids[row].tolist()
			anchor_pos = _find_subseq(seq, self.rec_anchor_ids)

			# 使用 inputs_embeds 生成时，解码序列里常常不包含原始 prompt，
			# 这时直接把整个已生成序列当作推荐列表前缀来约束。
			if anchor_pos < 0:
				rec_suffix_ids = seq
			else:
				rec_suffix_ids = seq[anchor_pos + len(self.rec_anchor_ids):]
			allowed = self._allowed_after_anchor(rec_suffix_ids)

			mask = torch.full_like(scores[row], float("-inf"))
			mask[allowed] = 0.0
			scores[row] = scores[row] + mask

		return scores


def build_recommendation_logits_processor(tokenizer, candidate_items: List[str], top_k: int):
	if not candidate_items or top_k <= 0:
		return None
	return TrieConstrainedLogitsProcessor(tokenizer, candidate_items, top_k)
