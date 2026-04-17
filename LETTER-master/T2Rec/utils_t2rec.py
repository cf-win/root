import os
import json
import random
import datetime
import numpy as np
import torch
from torch.utils.data import ConcatDataset
from data_t2rec import AnomalyRecDataset


def parse_global_args(parser):
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_model", type=str, default="./llama-7b/")
    parser.add_argument("--output_dir", type=str, default="./ckpt/")
    return parser


def parse_dataset_args(parser):
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--dataset", type=str, default="Instruments")
    parser.add_argument("--index_file", type=str, default=".index.json")
    parser.add_argument("--max_his_len", type=int, default=20)
    parser.add_argument("--add_prefix", action="store_true", default=False)
    parser.add_argument("--his_sep", type=str, default=", ")
    parser.add_argument("--only_train_response", action="store_true", default=False)
    parser.add_argument("--train_prompt_sample_num", type=int, default=1)
    parser.add_argument("--train_data_sample_num", type=int, default=0)
    parser.add_argument("--valid_prompt_sample_num", type=int, default=1)
    parser.add_argument("--graph_token_path", type=str, default="")
    parser.add_argument("--behavior_token_path", type=str, default="")
    parser.add_argument("--task", type=str, default="rec_train",
                        help="rec_train, simple_rec, deep_rec (train always uses rec_train)")
    parser.add_argument("--use_title", action="store_true", default=False)
    return parser


def parse_train_args(parser):
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--logging_step", type=int, default=10)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str,
                        default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj")
    parser.add_argument("--lora_modules_to_save", type=str, default="embed_tokens,lm_head")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--save_and_eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_and_eval_steps", type=int, default=1000)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--deepspeed", type=str, default="./config/ds_z2_bf16.json")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lambda_anomaly", type=float, default=0.0,
                        help="deprecated: LLM anomaly text loss is disabled; this flag is kept for compatibility")
    parser.add_argument("--lambda_risk", type=float, default=1.0)
    parser.add_argument("--graph_dim", type=int, default=128)
    parser.add_argument("--behavior_dim", type=int, default=64)
    parser.add_argument("--probe_dim", type=int, default=64)
    return parser


def parse_test_args(parser):
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--lora", action="store_true", default=True)
    parser.add_argument("--results_file", type=str, default="./results/test.json")
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--sample_num", type=int, default=-1)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10")
    parser.add_argument("--risk_threshold", type=float, default=0.5)
    parser.add_argument("--auto_risk_threshold", action="store_true", default=False)
    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_step", type=float, default=0.05)
    parser.add_argument("--probe_dim", type=int, default=64)
    return parser


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")
    return cur


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def load_datasets(args):
    train_data = AnomalyRecDataset(
        args,
        mode="train",
        prompt_sample_num=args.train_prompt_sample_num,
        sample_num=args.train_data_sample_num
    )
    valid_data = AnomalyRecDataset(
        args,
        mode="valid",
        prompt_sample_num=args.valid_prompt_sample_num
    )
    return train_data, valid_data


def load_test_dataset(args):
    test_data = AnomalyRecDataset(
        args,
        mode="test",
        sample_num=args.sample_num
    )
    return test_data


def load_json(file):
    with open(file, "r") as f:
        data = json.load(f)
    return data


def compute_metrics(predictions, targets, anomaly_preds, anomaly_labels, metrics_str):
    """
    Leave-one-out 推荐评估（每个用户仅 1 个真实目标）
    - Hit@k: top-k 推荐中是否包含目标物品
    - NDCG@k: 单目标下 IDCG=1.0，因此 NDCG@k = DCG@k
      即命中时 1/log2(rank+1)，未命中为 0
    """
    fixed_preds = []
    for p in predictions:
        if isinstance(p, (list, tuple)):
            fixed_preds.append(list(p))
        else:
            fixed_preds.append([])
    predictions = fixed_preds

    def to_single_target(t):
        if isinstance(t, (list, tuple)):
            return str(t[0]) if len(t) > 0 else None
        if isinstance(t, set):
            if len(t) == 0:
                return None
            return str(next(iter(t)))
        if t is None:
            return None
        return str(t)

    single_targets = [to_single_target(t) for t in targets]

    metrics = metrics_str.split(",")
    results = {}
    n = min(len(predictions), len(single_targets))

    for metric in metrics:
        metric = metric.strip()
        if metric.startswith("hit@"):
            k = int(metric.split("@")[1])
            hits = 0
            for pred, target in zip(predictions[:n], single_targets[:n]):
                if target is not None and target in pred[:k]:
                    hits += 1
            results[metric] = hits / n if n > 0 else 0.0

        elif metric.startswith("ndcg@"):
            k = int(metric.split("@")[1])
            ndcg_sum = 0.0
            for pred, target in zip(predictions[:n], single_targets[:n]):
                if target is not None and target in pred[:k]:
                    rank = pred[:k].index(target) + 1
                    ndcg_sum += 1.0 / np.log2(rank + 1)
            results[metric] = ndcg_sum / n if n > 0 else 0.0

    if len(anomaly_preds) > 0 and len(anomaly_labels) > 0:
        correct = sum([1 for p, l in zip(anomaly_preds, anomaly_labels) if p == l])
        results["anomaly_acc"] = correct / len(anomaly_labels)
        tp = sum([1 for p, l in zip(anomaly_preds, anomaly_labels) if p == "Yes" and l == "Yes"])
        fp = sum([1 for p, l in zip(anomaly_preds, anomaly_labels) if p == "Yes" and l == "No"])
        fn = sum([1 for p, l in zip(anomaly_preds, anomaly_labels) if p == "No" and l == "Yes"])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        results["anomaly_precision"] = precision
        results["anomaly_recall"] = recall
        results["anomaly_f1"] = f1

    return results

