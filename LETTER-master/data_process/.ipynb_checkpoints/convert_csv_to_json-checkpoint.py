"""
将 CSV 格式的数据转换为 LETTER 标准 JSON 格式
输入: {dataset}_filled_label.csv
输出: 
  - {dataset}.item.json (item_idx -> {title, description})
  - {dataset}.inter.json (user_idx -> [item_idx_1, item_idx_2, ...])
  - {dataset}.anomaly.json (user_idx -> label) [可选]
"""
import argparse
import csv
import json
import os
from collections import defaultdict


def main(args):
    csv_file = os.path.join(args.data_dir, f"{args.dataset}_filled.csv")
    output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading CSV: {csv_file}")
    
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
    # label: -1=未标注, 1=正常, 0=恶意
    anomaly_data = {}
    user_labels = defaultdict(list)
    for row in rows:
        user_id = row["user_id"]
        label = row.get("label", "-1")
        if label != "-1" and label != "":
            user_idx = user_id_to_idx[user_id]
            user_labels[user_idx].append(int(label))
    
    for user_idx, labels in user_labels.items():
        # 如果有任何 0（恶意）标签，则标记为恶意
        if 0 in labels:
            anomaly_data[user_idx] = 0
        else:
            anomaly_data[user_idx] = 1
    print(f"Users with anomaly labels: {len(anomaly_data)}")
    print(f"  - Malicious (0): {sum(1 for v in anomaly_data.values() if v == 0)}")
    print(f"  - Normal (1): {sum(1 for v in anomaly_data.values() if v == 1)}")
    
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
    args = parser.parse_args()
    main(args)

