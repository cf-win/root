import os
import json
import random
import torch
import csv
import numpy as np
from torch.utils.data import Dataset
from prompt_t2rec import sft_prompt, all_prompt, GRAPH_TOKEN, BEHAVIOR_TOKEN


class AnomalyRecDataset(Dataset):

    def __init__(self, args, mode="train", prompt_sample_num=1, sample_num=-1):
        super().__init__()
        self.args = args
        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.sample_num = sample_num
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)
        self.max_his_len = args.max_his_len
        self.his_sep = args.his_sep
        self.index_file = args.index_file
        self.add_prefix = args.add_prefix
        self.task = args.task
        self.top_k = getattr(args, "top_k", 10)
        self.use_title = getattr(args, "use_title", False)

        # task -> prompts
        # 训练阶段只保留推荐生成目标：若传入 anomaly 任务则自动回退到 rec_only。
        effective_task = self.task
        if self.mode == "train" and self.task in {"anomaly_rec", "anomaly_only"}:
            effective_task = "rec_only"

        if effective_task == "anomaly_rec":
            self.prompts = all_prompt["anomaly_rec"]
        elif effective_task == "anomaly_only":
            self.prompts = all_prompt["anomaly_only"]
        elif effective_task == "rec_only":
            self.prompts = all_prompt["rec_only"]
        elif effective_task == "simple_rec":
            self.prompts = all_prompt["simple_rec"]
        elif effective_task == "deep_rec":
            self.prompts = all_prompt["deep_rec"]
        else:
            # fallback
            self.prompts = all_prompt["simple_rec"]

        self.new_tokens = None
        self.graph_tokens = None
        self.behavior_tokens = None
        self.item_meta = {}

        self._load_data()
        self._load_item_meta()
        self._remap_items()
        self._load_tokens()
        self._build_user_labels()

        if self.mode == "train":
            self.inter_data = self._process_train_data()
        elif self.mode == "valid":
            self.inter_data = self._process_valid_data()
        elif self.mode == "test":
            self.inter_data = self._process_test_data()

    def _load_data(self):
        # 先初始化 anomaly_labels
        self.anomaly_labels = {}

        inter_json = os.path.join(self.data_path, self.dataset + ".inter.json")
        if os.path.exists(inter_json):
            with open(inter_json, "r") as f:
                self.inters = json.load(f)
        else:
            self.inters = self._load_inters_from_csv()

        # 注意：这里的 index 路径是 dataset + index_file（如 Amazon_CD + .index.json）
        index_json = os.path.join(self.data_path, self.dataset + self.index_file)
        if os.path.exists(index_json):
            with open(index_json, "r") as f:
                self.indices = json.load(f)
        else:
            self.indices = {}

        # 从 anomaly.json 加载额外的标签（如果存在）
        anomaly_file = os.path.join(self.data_path, self.dataset + ".anomaly.json")
        if os.path.exists(anomaly_file):
            with open(anomaly_file, "r") as f:
                extra_labels = json.load(f)
                # 统一格式：0 -> "Yes" (恶意), 1 -> "No" (正常)
                for uid, lbl in extra_labels.items():
                    if isinstance(lbl, int):
                        self.anomaly_labels[uid] = "Yes" if lbl == 0 else "No"
                    else:
                        self.anomaly_labels[uid] = lbl

    def _load_inters_from_csv(self):
        csv_file = os.path.join(self.data_path, self.dataset + "_filled_label.csv")
        if not os.path.exists(csv_file):
            csv_file = os.path.join(self.data_path, self.dataset + ".csv")
        if not os.path.exists(csv_file):
            return {}

        inters = {}
        user_item_time = []
        with open(csv_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row.get("user_id", row.get("\ufeffuser_id", ""))
                iid = row.get("item_id", "")
                t = int(row.get("time", 0))
                label = row.get("label", "-1")
                if not uid or not iid:
                    continue
                user_item_time.append((uid, iid, t, label))

        user_item_time.sort(key=lambda x: (x[0], x[2]))
        for uid, iid, t, label in user_item_time:
            if uid not in inters:
                inters[uid] = []
            inters[uid].append(iid)
            if label != "-1" and label != "":
                # label=0 是恶意用户(Yes), label=1 是正常用户(No)
                lbl = "Yes" if label == "0" else "No"
                self.anomaly_labels[uid] = lbl
        return inters

    def _load_item_meta(self):
        meta_file = os.path.join(self.data_path, "meta_" + self.dataset.replace("_5", "") + ".json")
        if not os.path.exists(meta_file):
            alt_patterns = [
                os.path.join(self.data_path, "meta_" + self.dataset + ".json"),
                os.path.join(self.data_path, "meta.json"),
            ]
            for alt in alt_patterns:
                if os.path.exists(alt):
                    meta_file = alt
                    break

        if os.path.exists(meta_file):
            with open(meta_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        asin = item.get("asin", "")
                        if asin:
                            self.item_meta[asin] = {
                                "title": item.get("title", ""),
                                "description": item.get("description", ""),
                                "categories": item.get("categories", []),
                            }
                    except:
                        continue

        csv_with_meta = os.path.join(self.data_path, self.dataset + "_filled_label.csv")
        if not os.path.exists(csv_with_meta):
            csv_with_meta = os.path.join(self.data_path, self.dataset + "_filled.csv")
        if os.path.exists(csv_with_meta):
            with open(csv_with_meta, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    iid = row.get("item_id", "")
                    if iid and iid not in self.item_meta:
                        self.item_meta[iid] = {
                            "title": row.get("title", ""),
                            "description": row.get("description", ""),
                            "categories": [],
                        }

    def _build_user_labels(self):
        for uid in self.inters.keys():
            if uid not in self.anomaly_labels:
                self.anomaly_labels[uid] = "No"

    def _load_tokens(self):
        graph_token_path = getattr(self.args, "graph_token_path", None)
        behavior_token_path = getattr(self.args, "behavior_token_path", None)

        if graph_token_path and os.path.exists(graph_token_path):
            self.graph_tokens = torch.load(graph_token_path, map_location="cpu")
        else:
            self.graph_tokens = {}

        if behavior_token_path and os.path.exists(behavior_token_path):
            self.behavior_tokens = torch.load(behavior_token_path, map_location="cpu")
        else:
            self.behavior_tokens = {}

        if len(self.graph_tokens) > 0:
            sample_key = list(self.graph_tokens.keys())[0]
            sample_val = self.graph_tokens[sample_key]
            if sample_val.dim() == 2:
                self.graph_dim = sample_val.shape[1]
                self.graph_seq_len = sample_val.shape[0]
            else:
                self.graph_dim = sample_val.shape[0]
                self.graph_seq_len = 1
        else:
            self.graph_dim = 128
            self.graph_seq_len = 1

        if len(self.behavior_tokens) > 0:
            sample_key = list(self.behavior_tokens.keys())[0]
            self.behavior_dim = self.behavior_tokens[sample_key].shape[-1]
        else:
            self.behavior_dim = 64

    def _remap_items(self):
        self.remapped_inters = dict()
        self.item_titles = dict()
        self.token_to_meta = dict()
        self.token_to_item = dict()

        for uid, items in self.inters.items():
            new_items = []
            for i in items:
                if str(i) in self.indices:
                    token_repr = "".join(self.indices[str(i)])
                    self.token_to_item[token_repr] = str(i)
                else:
                    token_repr = str(i)
                    self.token_to_item[token_repr] = str(i)

                new_items.append(token_repr)

                if str(i) in self.item_meta:
                    self.item_titles[token_repr] = self.item_meta[str(i)].get("title", "")
                    self.token_to_meta[token_repr] = self.item_meta[str(i)]

            self.remapped_inters[uid] = new_items

        # 全量 item token 池（用于补齐）
        self.all_item_tokens = list(self.token_to_item.keys())

    def _get_item_display(self, item_token):
        if self.use_title:
            title = self.item_titles.get(item_token, "")
            if title:
                return f"{item_token} ({title[:50]})"
        return item_token

    def _extract_preferences(self, history_items):
        categories = []
        for item in history_items[-5:]:
            meta = self.token_to_meta.get(item, {})
            cats = meta.get("categories", [])
            if cats and len(cats) > 0:
                for cat_list in cats:
                    if isinstance(cat_list, list) and len(cat_list) > 1:
                        categories.append(cat_list[-1])
        if categories:
            from collections import Counter
            cat_counts = Counter(categories)
            top_cats = cat_counts.most_common(2)
            pref1 = f'"{top_cats[0][0]}", confidence: 0.{min(9, top_cats[0][1] * 2)}'
            pref2 = f'"{top_cats[1][0]}", confidence: 0.{min(8, top_cats[1][1] * 2)}' if len(top_cats) > 1 else '"general", confidence: 0.5'
            return pref1, pref2
        return '"general", confidence: 0.7', '"variety", confidence: 0.5'

    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens
        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens.add(GRAPH_TOKEN)
        self.new_tokens.add(BEHAVIOR_TOKEN)
        self.new_tokens = sorted(list(self.new_tokens))
        return self.new_tokens

    def _pad_rec_items(self, target_items, history_items):
        """
        补齐到 top_k，但返回 is_real mask:
          is_real=1 表示真实正例（来自未来序列）
          is_real=0 表示补齐项（不参与loss）
        """
        real_items = list(target_items)[:self.top_k]
        is_real = [1] * len(real_items)

        if len(real_items) < self.top_k:
            ban = set(history_items) | set(real_items)
            pool = [x for x in self.all_item_tokens if x not in ban]
            need = self.top_k - len(real_items)

            if len(pool) >= need:
                pad = random.sample(pool, need)
            else:
                pad = pool  # 极端情况：候选不足

            real_items.extend(pad)
            is_real.extend([0] * len(pad))

        real_items = real_items[:self.top_k]
        is_real = is_real[:self.top_k]
        return real_items, is_real

    def _process_train_data(self):
        """
        训练集构建：
        - 原逻辑：先构造全量 inter_data，再按 sample_num 抽样
        - ✅ 这里加入 early-stop：如果 sample_num>0，则构造到 sample_num 立即返回
          这样可以非常快速地验证训练流程，避免长时间预处理/内存暴涨。
        """
        inter_data = []
        k = self.top_k  # 这里的 k 就是你要留给 valid/test 的最后 k 个

        # 仅用于快速验证时的提示
        show_progress = (self.mode == "train" and self.sample_num is not None and self.sample_num > 0)

        for uid in self.remapped_inters:
            full_seq = self.remapped_inters[uid]

            # 只用前面部分做训练：去掉最后 k 个 item
            if len(full_seq) <= k + 1:
                continue
            items = full_seq[:-k]

            anomaly_label = self.anomaly_labels.get(uid, "No")

            # 滑窗：对 items 里的每个位置 i 构造一个样本
            for i in range(1, len(items)):
                one_data = dict()
                one_data["user_id"] = uid

                # history
                history_full = items[:i]
                history_for_text = history_full[-self.max_his_len:] if self.max_his_len > 0 else history_full

                # target（真实未来序列，可能不足 k）
                target_items = items[i:i + k]
                full_items, is_real = self._pad_rec_items(target_items, history_items=history_full)

                one_data["rec_list"] = ", ".join(full_items)
                one_data["rec_items"] = full_items
                one_data["rec_is_real"] = is_real

                one_data["item"] = items[i]
                one_data["anomaly_label"] = anomaly_label
                one_data["top_k"] = k

                pref1, pref2 = self._extract_preferences(history_for_text)
                one_data["preference_1"] = pref1
                one_data["preference_2"] = pref2

                if self.add_prefix:
                    history_text = [
                        str(idx + 1) + ". " + self._get_item_display(item_idx)
                        for idx, item_idx in enumerate(history_for_text)
                    ]
                else:
                    history_text = [self._get_item_display(item_idx) for item_idx in history_for_text]

                one_data["inters"] = self.his_sep.join(history_text)
                inter_data.append(one_data)

                # ✅ early-stop：达到 sample_num 立即返回（快速验证训练）
                if self.sample_num is not None and self.sample_num > 0 and len(inter_data) >= self.sample_num:
                    if show_progress:
                        print(f"[Dataset] Early-stop train set building at {len(inter_data)} samples (sample_num={self.sample_num}).")
                    return inter_data

        # 走到这里说明 sample_num<=0 或者数据本身不足
        if self.sample_num is not None and self.sample_num > 0 and self.sample_num < len(inter_data):
            # 理论上 early-stop 已经 return，这里只是兜底
            all_idx = list(range(len(inter_data)))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)
            inter_data = [inter_data[i] for i in sample_idx]

        return inter_data

    def _process_valid_data(self):
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            anomaly_label = self.anomaly_labels.get(uid, "No")

            # 保证 target 是 top_k 个真实 item
            if len(items) <= self.top_k:
                continue

            one_data = dict()
            one_data["user_id"] = uid

            history_full = items[:-self.top_k]
            target_items = items[-self.top_k:]  # top_k 个真实目标（与 test 一样）

            history_for_text = history_full
            if self.max_his_len > 0:
                history_for_text = history_for_text[-self.max_his_len:]

            # valid 的标准答案直接用真实 target_items，不随机补齐
            one_data["rec_list"] = ", ".join(target_items)
            one_data["rec_items"] = target_items
            one_data["rec_is_real"] = [1] * len(target_items)

            one_data["item"] = target_items[0]  # 保留字段
            one_data["anomaly_label"] = anomaly_label
            one_data["top_k"] = self.top_k

            pref1, pref2 = self._extract_preferences(history_for_text)
            one_data["preference_1"] = pref1
            one_data["preference_2"] = pref2

            if self.add_prefix:
                history_text = [
                    str(k + 1) + ". " + self._get_item_display(item_idx)
                    for k, item_idx in enumerate(history_for_text)
                ]
            else:
                history_text = [self._get_item_display(item_idx) for item_idx in history_for_text]
            one_data["inters"] = self.his_sep.join(history_text)

            inter_data.append(one_data)

        return inter_data

    def _process_test_data(self):
        """
        改动点：
        - 目标（target）改为最后 top_k 个真实 item：items[-top_k:]
        - 历史（history）使用 items[:-top_k]
        - 为了保证 target 全是真实的 top_k 个，如果 len(items) <= top_k，则跳过该用户
        """
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            anomaly_label = self.anomaly_labels.get(uid, "No")

            # ===== 保证 target 是 top_k 个真实 item =====
            if len(items) <= self.top_k:
                continue

            one_data = dict()
            one_data["user_id"] = uid

            history_full = items[:-self.top_k]
            target_items = items[-self.top_k:]  # top_k 个真实目标

            history_for_text = history_full
            if self.max_his_len > 0:
                history_for_text = history_for_text[-self.max_his_len:]

            # ===== test 的标准答案直接用真实 target_items，不随机补齐 =====
            one_data["rec_list"] = ", ".join(target_items)
            one_data["rec_items"] = target_items
            one_data["rec_is_real"] = [1] * len(target_items)

            one_data["item"] = target_items[0]  # 保留字段，实际 test 评估不依赖它
            one_data["anomaly_label"] = anomaly_label
            one_data["top_k"] = self.top_k

            pref1, pref2 = self._extract_preferences(history_for_text)
            one_data["preference_1"] = pref1
            one_data["preference_2"] = pref2

            if self.add_prefix:
                history_text = [str(k + 1) + ". " + self._get_item_display(item_idx) for k, item_idx in enumerate(history_for_text)]
            else:
                history_text = [self._get_item_display(item_idx) for item_idx in history_for_text]
            one_data["inters"] = self.his_sep.join(history_text)

            inter_data.append(one_data)

        if self.sample_num is not None and self.sample_num > 0 and self.sample_num < len(inter_data):
            all_idx = list(range(len(inter_data)))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)
            inter_data = [inter_data[i] for i in sample_idx]
        return inter_data

    def _get_text_data(self, data, prompt):
        system = prompt.get("system", "")
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)
        input_text = sft_prompt.format(system=system, instruction=instruction, response="")
        output_text = sft_prompt.format(system=system, instruction=instruction, response=response)
        if self.mode == "test":
            return input_text, response
        return input_text, output_text

    def _get_tokens(self, user_id):
        graph_token = self.graph_tokens.get(str(user_id), None)
        behavior_token = self.behavior_tokens.get(str(user_id), None)

        if graph_token is None:
            if self.graph_seq_len > 1:
                graph_token = torch.zeros(self.graph_seq_len, self.graph_dim)
            else:
                graph_token = torch.zeros(self.graph_dim)

        if behavior_token is None:
            behavior_token = torch.zeros(self.behavior_dim)

        return graph_token, behavior_token

    def __len__(self):
        if self.mode == "train":
            return len(self.inter_data) * self.prompt_sample_num
        else:
            return len(self.inter_data)

    def __getitem__(self, index):
        idx = index // self.prompt_sample_num if self.mode == "train" else index
        d = self.inter_data[idx]
        prompt = self.prompts[0]

        input_text, output_text = self._get_text_data(d, prompt)
        graph_token, behavior_token = self._get_tokens(d["user_id"])

        return dict(
            input_ids=input_text,
            labels=output_text,
            graph_token=graph_token,
            behavior_token=behavior_token,
            user_id=d["user_id"],
            anomaly_label=d["anomaly_label"],
            risk_label=1.0 if d["anomaly_label"] == "Yes" else 0.0,
            rec_items=d.get("rec_items", None),
            rec_is_real=d.get("rec_is_real", None),
            route_simple_input=self._get_text_data(d, all_prompt["simple_rec"][0])[0] if self.mode == "test" else None,
            route_deep_input=self._get_text_data(d, all_prompt["deep_rec"][0])[0] if self.mode == "test" else None,
        )
