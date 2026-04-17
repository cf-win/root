import argparse
import os
import json
import re

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LogitsProcessorList
from peft import PeftModel

from modeling_t2rec import T2Rec
from prompt_t2rec import GRAPH_TOKEN, BEHAVIOR_TOKEN
from generation_trie_t2rec import build_recommendation_logits_processor
from utils_t2rec import (
    parse_global_args,
    parse_test_args,
    parse_dataset_args,
    set_seed,
    ensure_dir,
    load_datasets,
    load_test_dataset,
    compute_metrics,
)
from collator_t2rec import T2RecTestCollator


def extract_response_text(decoded: str) -> str:
    if "### Response:" in decoded:
        return decoded.split("### Response:")[-1].strip()
    if "Response:" in decoded:
        return decoded.split("Response:")[-1].strip()
    return decoded.strip()


def normalize_item_text(text: str) -> str:
    return "".join(text.strip().split())


def parse_target_item(target: str) -> str:
    t = target.strip()
    if "Final Recommendation List:" in t:
        m = re.search(r"Final Recommendation List:\s*(.*)", t)
        if m:
            sid = "".join(re.findall(r"<[^>]+>", m.group(1)))
            if sid:
                return sid
    sid = "".join(re.findall(r"<[^>]+>", t))
    return sid if sid else normalize_item_text(t)


def _f1_from_scores(scores, labels, threshold):
    tp = fp = fn = 0
    for s, y in zip(scores, labels):
        p = 1 if s >= threshold else 0
        if p == 1 and y == 1:
            tp += 1
        elif p == 1 and y == 0:
            fp += 1
        elif p == 0 and y == 1:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def tune_risk_threshold_on_valid(args, model, tokenizer):
    _, valid_data = load_datasets(args)
    collator = T2RecTestCollator(args, tokenizer)
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
    )

    scores, labels = [], []
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="TuneRiskThreshold(valid)"):
            graph_dev = next(model.temporal_aggregator.parameters()).device
            beh_dev = next(model.behavior_projector.parameters()).device
            graph_tokens = batch["graph_tokens"].to(graph_dev)
            behavior_tokens = batch["behavior_tokens"].to(beh_dev)
            risk_logits = model.compute_risk_logit(graph_tokens, behavior_tokens)
            risk_scores = torch.sigmoid(risk_logits).detach().cpu().tolist()
            scores.extend(risk_scores)
            labels.extend([1 if x == "Yes" else 0 for x in batch["anomaly_labels"]])

    if len(scores) == 0:
        return args.risk_threshold

    best_f1 = -1.0
    best_th = args.risk_threshold
    th = args.threshold_min
    while th <= args.threshold_max + 1e-12:
        f1 = _f1_from_scores(scores, labels, th)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
        th += args.threshold_step

    print(f"[AutoRiskThreshold] selected={best_th:.4f}, best_valid_f1={best_f1:.6f}")
    return best_th


def test(args):
    set_seed(args.seed)
    ensure_dir(os.path.dirname(args.results_file))

    tokenizer = AutoTokenizer.from_pretrained(
        args.ckpt_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    test_data = load_test_dataset(args)
    all_candidate_items = getattr(test_data, "all_item_tokens", [])
    candidate_set = set(all_candidate_items)

    max_item_new_tokens = 12
    if all_candidate_items:
        max_len_plain = max(len(tokenizer.encode(x, add_special_tokens=False)) for x in all_candidate_items)
        max_len_space = max(len(tokenizer.encode(" " + x, add_special_tokens=False)) for x in all_candidate_items)
        max_item_new_tokens = min(32, max(max_len_plain, max_len_space) + 1)

    graph_dim = test_data.graph_dim
    behavior_dim = test_data.behavior_dim
    collator = T2RecTestCollator(args, tokenizer)
    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
    )

    if args.lora:
        base_model = T2Rec.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            graph_dim=graph_dim,
            behavior_dim=behavior_dim,
            probe_dim=args.probe_dim,
            trust_remote_code=True,
        )
        base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, args.ckpt_path)
        model = model.merge_and_unload()
    else:
        model = T2Rec.from_pretrained(
            args.ckpt_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            graph_dim=graph_dim,
            behavior_dim=behavior_dim,
            probe_dim=args.probe_dim,
            trust_remote_code=True,
        )

    graph_token_id = tokenizer.convert_tokens_to_ids(GRAPH_TOKEN)
    behavior_token_id = tokenizer.convert_tokens_to_ids(BEHAVIOR_TOKEN)
    model.set_special_token_ids(graph_token_id, behavior_token_id)
    model.eval()

    selected_risk_threshold = args.risk_threshold
    if args.auto_risk_threshold:
        selected_risk_threshold = tune_risk_threshold_on_valid(args, model, tokenizer)

    all_predictions = []  # List[List[str]]
    all_targets = []      # List[str]
    all_anomaly_preds = []
    all_anomaly_labels = []
    all_risk_scores = []
    all_user_types = []
    all_det_test_flags = []
    all_route_types = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            targets = batch["targets"]
            anomaly_labels = batch["anomaly_labels"]
            simple_input_texts = batch["simple_input_texts"]
            deep_input_texts = batch["deep_input_texts"]
            user_types = batch.get("user_types", ["unlabel"] * len(batch["targets"]))
            det_test_flags = batch.get("det_test_flags", [False] * len(batch["targets"]))
            embed_device = model.get_input_embeddings().weight.device

            graph_dev = next(model.temporal_aggregator.parameters()).device
            beh_dev = next(model.behavior_projector.parameters()).device
            graph_tokens = batch["graph_tokens"].to(graph_dev)
            behavior_tokens = batch["behavior_tokens"].to(beh_dev)

            risk_logits = model.compute_risk_logit(graph_tokens, behavior_tokens)
            risk_scores = torch.sigmoid(risk_logits)

            route_texts = []
            route_preds = []
            for i in range(risk_scores.size(0)):
                score = float(risk_scores[i].item())
                is_deep = score >= selected_risk_threshold
                route_texts.append(deep_input_texts[i] if is_deep else simple_input_texts[i])
                route_preds.append("Yes" if is_deep else "No")

            inputs = tokenizer(
                text=route_texts,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_attention_mask=True,
            )

            input_ids = inputs["input_ids"].to(embed_device)
            attention_mask = inputs["attention_mask"].to(embed_device)
            inputs_embeds = model.get_input_embeddings()(input_ids)

            if graph_tokens is not None:
                if graph_tokens.dim() == 3:
                    graph_tokens = graph_tokens.to(dtype=next(model.temporal_aggregator.parameters()).dtype)
                    behavior_tokens = behavior_tokens.to(dtype=next(model.behavior_projector.parameters()).dtype)
                    graph_emb = model.temporal_aggregator(graph_tokens)
                else:
                    graph_emb = graph_tokens
                graph_emb = model.graph_projector(graph_emb).to(embed_device)
                for i in range(input_ids.size(0)):
                    pos = (input_ids[i] == graph_token_id).nonzero(as_tuple=True)[0]
                    if len(pos) > 0:
                        inputs_embeds[i, pos[0]] = graph_emb[i]

            if behavior_tokens is not None:
                behavior_emb = model.behavior_projector(behavior_tokens).to(embed_device)
                for i in range(input_ids.size(0)):
                    pos = (input_ids[i] == behavior_token_id).nonzero(as_tuple=True)[0]
                    if len(pos) > 0:
                        inputs_embeds[i, pos[0]] = behavior_emb[i]

            logits_processor = None
            rec_processor = build_recommendation_logits_processor(
                tokenizer=tokenizer,
                candidate_items=all_candidate_items,
            )
            if rec_processor is not None:
                logits_processor = LogitsProcessorList([rec_processor])

            output = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_item_new_tokens,
                num_beams=args.num_beams,
                do_sample=False,
                logits_processor=logits_processor,
                num_return_sequences=args.num_beams,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
            )

            sequences = output.sequences
            seq_scores = output.sequences_scores

            for i, target in enumerate(targets):
                start_idx = i * args.num_beams
                end_idx = start_idx + args.num_beams

                beam_pairs = []
                for j in range(start_idx, end_idx):
                    decoded = tokenizer.decode(sequences[j], skip_special_tokens=True)
                    pred_item = normalize_item_text(extract_response_text(decoded))
                    if candidate_set and pred_item not in candidate_set:
                        continue
                    beam_pairs.append((pred_item, float(seq_scores[j].item())))

                beam_pairs.sort(key=lambda x: x[1], reverse=True)
                ranked = []
                used = set()
                for pred_item, _ in beam_pairs:
                    if pred_item and pred_item not in used:
                        ranked.append(pred_item)
                        used.add(pred_item)
                all_predictions.append(ranked)

                all_targets.append(parse_target_item(target))
                all_anomaly_preds.append(route_preds[i])
                all_anomaly_labels.append(anomaly_labels[i])
                all_risk_scores.append(float(risk_scores[i].item()))
                all_route_types.append("deep" if route_preds[i] == "Yes" else "simple")
                all_user_types.append(user_types[i])
                all_det_test_flags.append(det_test_flags[i])

                if len(all_predictions) <= 3:
                    tqdm.write("=== DEBUG SAMPLE ===")
                    tqdm.write(f"TARGET: {all_targets[-1]}")
                    tqdm.write(f"PRED_TOPK: {all_predictions[-1][:10]}")

    eval_mask = [ut != "unlabel" and dt for ut, dt in zip(all_user_types, all_det_test_flags)]
    eval_anomaly_preds = [p for p, m in zip(all_anomaly_preds, eval_mask) if m]
    eval_anomaly_labels = [l for l, m in zip(all_anomaly_labels, eval_mask) if m]

    results = compute_metrics(
        all_predictions,
        all_targets,
        eval_anomaly_preds,
        eval_anomaly_labels,
        args.metrics,
    )
    results["anomaly_eval_count"] = len(eval_anomaly_labels)
    print("Results:", results)

    lens = [len(p) for p in all_predictions]
    print("avg candidate len:", sum(lens) / len(lens) if lens else 0)
    print("num empty:", sum(1 for x in lens if x == 0))

    output_data = {
        "args": vars(args),
        "selected_risk_threshold": selected_risk_threshold,
        "results": results,
        "predictions": [
            {
                "target": t,
                "risk_route_pred": ap,
                "risk_score": rs,
                "route_type": rt,
                "user_type": ut,
                "det_test_flag": dt,
                "predictions": p,
                "anomaly_label": al,
            }
            for t, p, ap, al, rs, rt, ut, dt in zip(
                all_targets,
                all_predictions,
                all_anomaly_preds,
                all_anomaly_labels,
                all_risk_scores,
                all_route_types,
                all_user_types,
                all_det_test_flags,
            )
        ],
    }
    with open(args.results_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {args.results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T2Rec Test")
    parser = parse_global_args(parser)
    parser = parse_test_args(parser)
    parser = parse_dataset_args(parser)
    args = parser.parse_args()
    test(args)
