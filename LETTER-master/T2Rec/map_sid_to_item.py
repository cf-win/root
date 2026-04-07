"""
SID 到 Item ID 映射脚本
用法: python map_sid_to_item.py --results results/Beauty_5_test.json --dataset Beauty_5
"""

import json
import re
import argparse
from pathlib import Path


def load_mappings(data_path, dataset):
    """加载 SID -> item_id 映射和 item 元数据"""
    # 加载 index.json
    index_file = Path(data_path) / dataset / f"{dataset}.index.json"
    with open(index_file) as f:
        index = json.load(f)
    
    # 构建 SID -> item_id 映射
    sid_to_item = {}
    for item_id, tokens in index.items():
        sid = ''.join(tokens)
        sid_to_item[sid] = item_id
    
    # 加载 item 元数据
    item_file = Path(data_path) / dataset / f"{dataset}.item.json"
    items = {}
    if item_file.exists():
        with open(item_file) as f:
            items = json.load(f)
    
    return sid_to_item, items


def map_sid(sid, sid_to_item, items):
    """将单个 SID 映射为 item 信息"""
    item_id = sid_to_item.get(sid)
    if item_id:
        item_info = items.get(item_id, {})
        return {
            "item_id": item_id,
            "title": item_info.get("title", "")[:60],
            "valid": True
        }
    return {"sid": sid, "valid": False}


def process_results(results_file, sid_to_item, items):
    """处理测试结果并映射 SID"""
    with open(results_file) as f:
        results = json.load(f)
    
    mapped_results = []
    stats = {"total": 0, "valid_target": 0, "valid_pred": 0, "total_pred": 0}
    
    for p in results['predictions']:
        stats["total"] += 1
        
        # 提取真实目标 SID
        target_sids = re.findall(r'<a_\d+><b_\d+><c_\d+><d_\d+>', p['target'])
        target_mapped = map_sid(target_sids[0], sid_to_item, items) if target_sids else None
        if target_mapped and target_mapped['valid']:
            stats["valid_target"] += 1
        
        # 提取预测 SID
        pred_sids = p['predictions'][0] if p['predictions'] else []
        pred_mapped = []
        for sid in pred_sids:
            m = map_sid(sid, sid_to_item, items)
            pred_mapped.append(m)
            stats["total_pred"] += 1
            if m['valid']:
                stats["valid_pred"] += 1
        
        mapped_results.append({
            "anomaly_pred": p.get('anomaly_pred', ''),
            "anomaly_label": p.get('anomaly_label', ''),
            "target": target_mapped,
            "predictions": pred_mapped
        })
    
    return mapped_results, stats


def main():
    parser = argparse.ArgumentParser(description='将 SID 映射回 Item ID')
    parser.add_argument('--results', type=str, required=True, help='测试结果文件路径')
    parser.add_argument('--dataset', type=str, default='Beauty_5', help='数据集名称')
    parser.add_argument('--data_path', type=str, default='../data', help='数据目录')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径（默认为 results 文件同目录下的 _mapped.json）')
    parser.add_argument('--show', type=int, default=5, help='显示前N个样本')
    args = parser.parse_args()
    
    # 加载映射
    print(f"加载映射数据 ({args.dataset})...")
    sid_to_item, items = load_mappings(args.data_path, args.dataset)
    print(f"  共 {len(sid_to_item)} 个有效 SID")
    
    # 处理结果
    print(f"处理结果文件 ({args.results})...")
    mapped_results, stats = process_results(args.results, sid_to_item, items)
    
    # 统计
    print("\n=== 映射统计 ===")
    print(f"总样本数: {stats['total']}")
    print(f"真实目标有效率: {stats['valid_target']}/{stats['total']} = {stats['valid_target']/stats['total']*100:.1f}%")
    if stats['total_pred'] > 0:
        print(f"预测推荐有效率: {stats['valid_pred']}/{stats['total_pred']} = {stats['valid_pred']/stats['total_pred']*100:.1f}%")
    else:
        print("预测推荐有效率: N/A (无预测)")
    
    # 显示示例
    print(f"\n=== 前 {args.show} 个样本 ===")
    for i, r in enumerate(mapped_results[:args.show]):
        print(f"\n样本 {i}:")
        print(f"  异常判断: 预测={r['anomaly_pred']} | 真实={r['anomaly_label']}")
        
        if r['target']:
            if r['target']['valid']:
                print(f"  真实目标: {r['target']['item_id']} - {r['target']['title']}")
            else:
                print(f"  真实目标: {r['target']['sid']} (无效)")
        
        valid_preds = [p for p in r['predictions'] if p['valid']]
        invalid_count = len(r['predictions']) - len(valid_preds)
        
        if valid_preds:
            print(f"  预测推荐 (有效 {len(valid_preds)}/{len(r['predictions'])}):")
            for j, p in enumerate(valid_preds[:5]):
                print(f"    {j+1}. {p['item_id']} - {p['title']}")
            if len(valid_preds) > 5:
                print(f"    ... 还有 {len(valid_preds)-5} 个")
        else:
            print(f"  预测推荐: {len(r['predictions'])} 个，全部无效")
    
    # 保存结果
    output_file = args.output or args.results.replace('.json', '_mapped.json')
    output = {
        "stats": {
            "total_samples": stats['total'],
            "valid_target_rate": f"{stats['valid_target']/stats['total']*100:.1f}%",
            "valid_pred_rate": f"{stats['valid_pred']/stats['total_pred']*100:.1f}%" if stats['total_pred'] > 0 else "N/A"
        },
        "results": mapped_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()

