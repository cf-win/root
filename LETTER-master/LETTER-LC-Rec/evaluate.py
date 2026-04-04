import math

def get_topk_results(predictions, scores, targets, k, all_items=None):
    results = []
    B = len(targets)
    predictions = [_.split("Response:")[-1] for _ in predictions]
    predictions = [_.strip().replace(" ","") for _ in predictions]

    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000

    for b in range(B):
        batch_seqs = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]

        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        target_item = targets[b]
        if isinstance(target_item, (list, tuple, set)):
            target_items = set(target_item)
        else:
            target_items = {target_item}

        one_results = []
        for sorted_pred in sorted_pairs:
            if sorted_pred[0] in target_items:
                one_results.append(1)
            else:
                one_results.append(0)

        results.append({"hits": one_results, "target_count": len(target_items)})

    return results

def get_metrics_results(topk_results, metrics):
    res = {}
    for m in metrics:
        if m.lower().startswith("hit"):
            k = int(m.split("@")[1])
            res[m] = hit_k(topk_results, k)
        elif m.lower().startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k)
        else:
            raise NotImplementedError

    return res


def ndcg_k(topk_results, k):

    ndcg = 0.0
    for row in topk_results:
        res = row["hits"][:k]
        target_count = row["target_count"]
        dcg = 0.0
        for i in range(len(res)):
            dcg += res[i] / math.log(i + 2, 2)

        ideal_len = min(k, target_count)
        if ideal_len == 0:
            continue

        idcg = 0.0
        for i in range(ideal_len):
            idcg += 1.0 / math.log(i + 2, 2)

        ndcg += dcg / idcg
    return ndcg


def hit_k(topk_results, k):
    hit = 0.0
    for row in topk_results:
        res = row["hits"][:k]
        if sum(res) > 0:
            hit += 1
    return hit

