"""
将 CSV 格式的数据转换为 LETTER 标准 JSON 格式
输入: {dataset}_filled_label.csv
输出:
  - {dataset}.item.json (item_idx -> {title, description})
  - {dataset}.inter.json (user_idx -> [item_idx_1, item_idx_2, ...])
  - {dataset}.anomaly.json (user_idx -> label)
"""
import argparse
import csv
import json
import os
import random
from collections import defaultdict


def main(args):
    random.seed(args.seed)
    csv_file = os.path.join(args.data_dir, f"{args.dataset}_filled_label.csv")
    output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading CSV: {csv_file}")
    print(
        "User label rule: assign 0/1 only when labeled evidence is strong; "
        "otherwise assign -1 (unlabeled)."
    )
    print(f"  - min_labeled_reviews = {args.min_labeled_reviews}")
    print(f"  - min_confidence_ratio = {args.min_confidence_ratio}")
    print(f"  - det_test_ratio = {args.det_test_ratio}")
    print(f"  - random seed = {args.seed}")

    # 读取 CSV
    rows = []
    with open(csv_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"Total rows: {len(rows)}")

    # 构建 item 映射 (item_id -> item_idx)
    item_id_set = set()
    for row in rows:
        item_id_set.add(row["item_id"])

    item_id_to_idx = {item_id: idx for idx, item_id in enumerate(sorted(item_id_set))}
    print(f"Unique items: {len(item_id_to_idx)}")

    # 构建 user 映射 (user_id -> user_idx)
    user_id_set = set()
    for row in rows:
        user_id_set.add(row["user_id"])

    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(sorted(user_id_set))}
    print(f"Unique users: {len(user_id_to_idx)}")

    # 构建 item.json
    item_info = {}
    for row in rows:
        item_id = row["item_id"]
        item_idx = item_id_to_idx[item_id]
        if item_idx not in item_info:
            item_info[item_idx] = {
                "title": row.get("title", ""),
                "description": row.get("description", "")
            }

    # 构建 inter.json (按时间排序的用户交互序列)
    user_interactions = defaultdict(list)
    for row in rows:
        user_id = row["user_id"]
        item_id = row["item_id"]
        time = int(row.get("time", 0))
        user_idx = user_id_to_idx[user_id]
        item_idx = item_id_to_idx[item_id]
        user_interactions[user_idx].append((time, item_idx))

    # 按时间排序
    inter_data = {}
    for user_idx, interactions in user_interactions.items():
        interactions.sort(key=lambda x: x[0])
        inter_data[user_idx] = [item_idx for _, item_idx in interactions]

    # 构建 anomaly.json (用户异常标签)
    # 用户标签:
    #   -1 = 未标注用户（证据不足或标签冲突，不强行归类）
    #    1 = 正常用户
    #    0 = 异常用户
    # 评论标签:
    #   -1 / 空字符串 = 未标注评论
    #    1 = 正常评论
    #    0 = 异常评论
    anomaly_data = {}
    user_labels = defaultdict(list)
    for row in rows:
        user_id = row["user_id"]
        label = row.get("label", "-1")
        if label != "-1" and label != "":
            user_idx = user_id_to_idx[user_id]
            user_labels[user_idx].append(int(label))

    for user_idx in user_id_to_idx.values():
        labels = user_labels.get(user_idx, [])
        labeled_count = len(labels)
        if labeled_count < args.min_labeled_reviews:
            anomaly_data[user_idx] = -1
            continue

        malicious_count = sum(1 for label in labels if label == 0)
        normal_count = sum(1 for label in labels if label == 1)
        malicious_ratio = malicious_count / labeled_count
        normal_ratio = normal_count / labeled_count

        # 只有在样本量足够且某一类占比足够高时，才将用户标为异常/正常；
        # 否则统一视为未标注用户，以保证未标注用户占主体。
        if malicious_ratio >= args.min_confidence_ratio:
            anomaly_data[user_idx] = 0
        elif normal_ratio >= args.min_confidence_ratio:
            anomaly_data[user_idx] = 1
        else:
            anomaly_data[user_idx] = -1

    print(f"Total users with labels exported: {len(anomaly_data)}")
    print(f"  - Abnormal (0): {sum(1 for v in anomaly_data.values() if v == 0)}")
    print(f"  - Normal (1): {sum(1 for v in anomaly_data.values() if v == 1)}")
    print(f"  - Unlabeled (-1): {sum(1 for v in anomaly_data.values() if v == -1)}")

    # 生成用户元信息，用于训练/测试时区分 det_train/det_test 与 risk loss mask
    user_meta = {}
    good_users = [uid for uid, lbl in anomaly_data.items() if lbl == 1]
    bad_users = [uid for uid, lbl in anomaly_data.items() if lbl == 0]
    random.shuffle(good_users)
    random.shuffle(bad_users)
    good_split_idx = int(len(good_users) * (1.0 - args.det_test_ratio))
    bad_split_idx = int(len(bad_users) * (1.0 - args.det_test_ratio))
    det_train_good = set(good_users[:good_split_idx])
    det_train_bad = set(bad_users[:bad_split_idx])
    det_train_user_ids = det_train_good | det_train_bad
    for user_idx, label in anomaly_data.items():
        if label == -1:
            user_type = "unlabel"
            risk_label = -1
            det_train_flag = False
            det_test_flag = False
            risk_loss_mask = False
        else:
            user_type = "good" if label == 1 else "bad"
            det_train_flag = user_idx in det_train_user_ids
            det_test_flag = not det_train_flag
            risk_loss_mask = det_train_flag
            risk_label = 0 if label == 1 else 1
        user_meta[user_idx] = {
            "user_type": user_type,
            "risk_label": risk_label,
            "det_train_flag": det_train_flag,
            "det_test_flag": det_test_flag,
            "risk_loss_mask": risk_loss_mask,
        }

    print(f"User meta counts: det_train={sum(1 for v in user_meta.values() if v['det_train_flag'])}, det_test={sum(1 for v in user_meta.values() if v['det_test_flag'])}")

    # 保存文件
    item_file = os.path.join(output_dir, f"{args.dataset}.item.json")
    with open(item_file, "w") as f:
        json.dump(item_info, f, indent=2)
    print(f"Saved: {item_file}")

    inter_file = os.path.join(output_dir, f"{args.dataset}.inter.json")
    with open(inter_file, "w") as f:
        json.dump(inter_data, f, indent=2)
    print(f"Saved: {inter_file}")

    anomaly_file = os.path.join(output_dir, f"{args.dataset}.anomaly.json")
    with open(anomaly_file, "w") as f:
        json.dump(anomaly_data, f, indent=2)
    print(f"Saved: {anomaly_file}")

    user_meta_file = os.path.join(output_dir, f"{args.dataset}.user_meta.json")
    with open(user_meta_file, "w", encoding="utf-8") as f:
        json.dump(user_meta, f, indent=2)
    print(f"Saved: {user_meta_file}")

    # 保存 ID 映射（用于后续解析）
    mapping_file = os.path.join(output_dir, f"{args.dataset}.id_mapping.json")
    mapping = {
        "item_id_to_idx": item_id_to_idx,
        "user_id_to_idx": user_id_to_idx,
        "idx_to_item_id": {v: k for k, v in item_id_to_idx.items()},
        "idx_to_user_id": {v: k for k, v in user_id_to_idx.items()}
    }
    with open(mapping_file, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved: {mapping_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Beauty_5")
    parser.add_argument("--data_dir", type=str, default="T2Rec/data/Beauty_5")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument(
        "--min_labeled_reviews",
        type=int,
        default=3,
        help="用户至少需要多少条已标注评论，才有资格被划到正常/异常。"
    )
    parser.add_argument(
        "--min_confidence_ratio",
        type=float,
        default=0.5,
        help="某一类评论占已标注评论的最小比例，达到后才将用户划为该类。"
    )
    parser.add_argument(
        "--det_test_ratio",
        type=float,
        default=0.2,
        help="比例：在有标签用户中划分为 det_test 的比例，剩余用于 det_train。"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，用于 det_train/det_test 划分。"
    )
    args = parser.parse_args()
    main(args)
