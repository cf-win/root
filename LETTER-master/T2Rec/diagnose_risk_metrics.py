import argparse
import json
from typing import Dict, List, Tuple


def safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def confusion_and_metrics(preds: List[int], labels: List[int]) -> Dict[str, float]:
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    acc = safe_div(tp + tn, len(labels))

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "acc": acc,
    }


def to_binary_label(label: str) -> int:
    return 1 if str(label).strip().lower() == "yes" else 0


def quantiles(values: List[float]) -> Tuple[float, float, float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    xs = sorted(values)

    def pick(q: float) -> float:
        idx = int(round((len(xs) - 1) * q))
        return xs[max(0, min(idx, len(xs) - 1))]

    return (xs[0], pick(0.1), pick(0.5), pick(0.9), xs[-1])


def derive_preds_from_scores(scores: List[float], threshold: float) -> List[int]:
    return [1 if s >= threshold else 0 for s in scores]


def parse_entries(data: Dict) -> Tuple[List[float], List[int], List[int]]:
    entries = data.get("predictions", [])
    scores, labels, route_preds = [], [], []

    for e in entries:
        if "risk_score" not in e or "anomaly_label" not in e:
            continue
        s = float(e["risk_score"])
        y = to_binary_label(e["anomaly_label"])
        rp = e.get("risk_route_pred", None)
        if rp is None:
            rp_bin = None
        else:
            rp_bin = to_binary_label(rp)

        scores.append(s)
        labels.append(y)
        route_preds.append(rp_bin if rp_bin is not None else -1)

    return scores, labels, route_preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick diagnostic for risk-route metrics")
    parser.add_argument("--results_file", type=str, required=True, help="Path to *_test.json")
    parser.add_argument("--min_threshold", type=float, default=0.05)
    parser.add_argument("--max_threshold", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()

    with open(args.results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    scores, labels, route_preds = parse_entries(data)
    n = len(labels)
    if n == 0:
        print("No valid entries with risk_score + anomaly_label found.")
        return

    yes_ratio = safe_div(sum(labels), n)
    print("=== Basic Stats ===")
    print(f"samples: {n}")
    print(f"label_yes_ratio: {yes_ratio:.6f}")

    yes_scores = [s for s, y in zip(scores, labels) if y == 1]
    no_scores = [s for s, y in zip(scores, labels) if y == 0]

    y_q = quantiles(yes_scores)
    n_q = quantiles(no_scores)

    print("\n=== Score Distribution ===")
    print("format: min / p10 / p50 / p90 / max")
    print(f"Yes scores: {y_q[0]:.4f} / {y_q[1]:.4f} / {y_q[2]:.4f} / {y_q[3]:.4f} / {y_q[4]:.4f}")
    print(f"No  scores: {n_q[0]:.4f} / {n_q[1]:.4f} / {n_q[2]:.4f} / {n_q[3]:.4f} / {n_q[4]:.4f}")

    cfg_threshold = float(data.get("args", {}).get("risk_threshold", 0.5))

    has_route_pred = all(x in (0, 1) for x in route_preds)
    if has_route_pred:
        m_route = confusion_and_metrics(route_preds, labels)
        print("\n=== Current Route Pred (from file) ===")
        print(f"tp={m_route['tp']} fp={m_route['fp']} tn={m_route['tn']} fn={m_route['fn']}")
        print(
            "acc={:.6f} precision={:.6f} recall={:.6f} f1={:.6f}".format(
                m_route["acc"], m_route["precision"], m_route["recall"], m_route["f1"]
            )
        )
        yes_pred_ratio = safe_div(sum(route_preds), n)
        print(f"pred_yes_ratio={yes_pred_ratio:.6f}")

    current_preds = derive_preds_from_scores(scores, cfg_threshold)
    m_cfg = confusion_and_metrics(current_preds, labels)
    print(f"\n=== Threshold @ {cfg_threshold:.4f} ===")
    print(f"tp={m_cfg['tp']} fp={m_cfg['fp']} tn={m_cfg['tn']} fn={m_cfg['fn']}")
    print(
        "acc={:.6f} precision={:.6f} recall={:.6f} f1={:.6f}".format(
            m_cfg["acc"], m_cfg["precision"], m_cfg["recall"], m_cfg["f1"]
        )
    )

    sweep = []
    t = args.min_threshold
    while t <= args.max_threshold + 1e-12:
        preds = derive_preds_from_scores(scores, t)
        m = confusion_and_metrics(preds, labels)
        m["threshold"] = round(t, 6)
        sweep.append(m)
        t += args.step

    sweep_sorted = sorted(sweep, key=lambda x: (x["f1"], x["acc"]), reverse=True)

    print("\n=== Top Thresholds By F1 ===")
    print("threshold  acc      precision recall   f1      tp  fp  tn  fn")
    for row in sweep_sorted[: max(1, args.topk)]:
        print(
            "{:<9.4f} {:<8.4f} {:<9.4f} {:<8.4f} {:<7.4f} {:<3d} {:<3d} {:<3d} {:<3d}".format(
                row["threshold"],
                row["acc"],
                row["precision"],
                row["recall"],
                row["f1"],
                int(row["tp"]),
                int(row["fp"]),
                int(row["tn"]),
                int(row["fn"]),
            )
        )

    best = sweep_sorted[0]
    print("\n=== Recommended Threshold (max F1) ===")
    print(
        "best_threshold={:.4f}, f1={:.6f}, precision={:.6f}, recall={:.6f}, acc={:.6f}".format(
            best["threshold"], best["f1"], best["precision"], best["recall"], best["acc"]
        )
    )


if __name__ == "__main__":
    main()
