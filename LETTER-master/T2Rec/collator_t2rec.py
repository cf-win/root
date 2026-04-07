import copy
import torch
from prompt_t2rec import GRAPH_TOKEN, BEHAVIOR_TOKEN


class T2RecCollator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.only_train_response = args.only_train_response

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.graph_token_id = self.tokenizer.convert_tokens_to_ids(GRAPH_TOKEN)
        self.behavior_token_id = self.tokenizer.convert_tokens_to_ids(BEHAVIOR_TOKEN)

        yes_ids = self.tokenizer.encode(" Yes", add_special_tokens=False)
        no_ids = self.tokenizer.encode(" No", add_special_tokens=False)
        self.yes_token_id = yes_ids[0] if yes_ids else None
        self.no_token_id = no_ids[0] if no_ids else None

        self.judgment_ids = self.tokenizer.encode("Malicious Judgment:", add_special_tokens=False)
        self.rec_anchor_ids = self.tokenizer.encode("Final Recommendation List:", add_special_tokens=False)

    def _find_subseq(self, seq, pattern, start=0, end=None):
        if end is None:
            end = len(seq)
        L = len(pattern)
        if L == 0:
            return -1
        for i in range(start, end - L + 1):
            if seq[i:i + L] == pattern:
                return i
        return -1

    def _find_rec_anchor_pos(self, input_ids):
        ids = input_ids.tolist()
        pos = self._find_subseq(ids, self.rec_anchor_ids, start=0)
        if pos < 0:
            return -1
        return pos + len(self.rec_anchor_ids)

    def _find_anomaly_label_position(self, input_ids, target_token_id):
        """找到 anomaly label (Yes/No) 的位置"""
        input_ids_list = input_ids.tolist()
        seq_len = len(input_ids_list)
        pattern_len = len(self.judgment_ids)

        # 首先找到 "Malicious Judgment:" 的位置
        judgment_pos = -1
        for i in range(seq_len - pattern_len):
            if input_ids_list[i:i + pattern_len] == self.judgment_ids:
                judgment_pos = i + pattern_len
                break

        if judgment_pos < 0:
            return -1

        # 在 judgment 后面找 target_token_id (Yes 或 No 的第一个 token)
        for i in range(judgment_pos, min(judgment_pos + 10, seq_len)):
            if input_ids_list[i] == target_token_id:
                return i

        return judgment_pos

    def __call__(self, batch):
        input_texts = [d["input_ids"] for d in batch]  # instruction 部分
        full_texts = [d["labels"] + self.tokenizer.eos_token for d in batch]  # instruction + response

        graph_tokens = torch.stack([d["graph_token"] for d in batch], dim=0)
        behavior_tokens = torch.stack([d["behavior_token"] for d in batch], dim=0)
        anomaly_labels = [d["anomaly_label"] for d in batch]
        risk_labels = torch.tensor([d.get("risk_label", 0.0) for d in batch], dtype=torch.float)

        #每条样本的推荐item与是否真实mask（补齐部分不算loss）
        rec_items_list = [d.get("rec_items", None) for d in batch]
        rec_is_real_list = [d.get("rec_is_real", None) for d in batch]

        inputs = self.tokenizer(
            text=full_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        labels = copy.deepcopy(inputs["input_ids"])
        labels[labels == self.tokenizer.pad_token_id] = -100

        if self.only_train_response:
            input_only = self.tokenizer(
                text=input_texts,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            )
            for i in range(len(batch)):
                input_len = (input_only["input_ids"][i] != self.tokenizer.pad_token_id).sum().item()
                labels[i, :input_len] = -100

        # ========= 路线B核心：补齐出来的推荐item token 不参与loss =========
        for i in range(len(batch)):
            rec_items = rec_items_list[i]
            rec_is_real = rec_is_real_list[i]
            if not rec_items or not rec_is_real:
                continue

            seq = inputs["input_ids"][i].tolist()
            anchor = self._find_rec_anchor_pos(inputs["input_ids"][i])
            if anchor < 0:
                anchor = 0

            for item_str, is_real in zip(rec_items, rec_is_real):
                if is_real == 1:
                    continue  # 真实正例参与loss
                item_ids = self.tokenizer.encode(item_str, add_special_tokens=False)
                if not item_ids:
                    continue
                p = self._find_subseq(seq, item_ids, start=anchor)
                if p >= 0:
                    labels[i, p:p + len(item_ids)] = -100

        # anomaly_mask（只标记 Yes/No 的 token 位置用于 anomaly loss）
        anomaly_mask = torch.zeros_like(labels)
        for i, label in enumerate(anomaly_labels):
            target_id = self.yes_token_id if label == "Yes" else self.no_token_id

            pos = self._find_anomaly_label_position(inputs["input_ids"][i], target_id)
            if pos > 0 and pos < labels.size(1) and labels[i, pos] != -100:
                anomaly_mask[i, pos] = 1
            else:
                if target_id is not None:
                    positions = (inputs["input_ids"][i] == target_id).nonzero(as_tuple=True)[0]
                    if len(positions) > 0:
                        for p in positions:
                            if p > 0 and labels[i, p] != -100:
                                anomaly_mask[i, p] = 1
                                break

        inputs["labels"] = labels
        inputs["graph_tokens"] = graph_tokens.float()
        inputs["behavior_tokens"] = behavior_tokens.float()
        inputs["anomaly_mask"] = anomaly_mask
        inputs["risk_labels"] = risk_labels
        return inputs


class T2RecTestCollator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.graph_token_id = self.tokenizer.convert_tokens_to_ids(GRAPH_TOKEN)
        self.behavior_token_id = self.tokenizer.convert_tokens_to_ids(BEHAVIOR_TOKEN)

    def __call__(self, batch):
        input_texts = [d["input_ids"] for d in batch]
        simple_input_texts = [d.get("route_simple_input", d["input_ids"]) for d in batch]
        deep_input_texts = [d.get("route_deep_input", d["input_ids"]) for d in batch]
        targets = [d["labels"] for d in batch]
        graph_tokens = torch.stack([d["graph_token"] for d in batch], dim=0)
        behavior_tokens = torch.stack([d["behavior_token"] for d in batch], dim=0)
        user_ids = [d["user_id"] for d in batch]
        anomaly_labels = [d["anomaly_label"] for d in batch]

        return {
            "input_texts": input_texts,
            "simple_input_texts": simple_input_texts,
            "deep_input_texts": deep_input_texts,
            "targets": targets,
            "graph_tokens": graph_tokens.float(),
            "behavior_tokens": behavior_tokens.float(),
            "user_ids": user_ids,
            "anomaly_labels": anomaly_labels,
        }
