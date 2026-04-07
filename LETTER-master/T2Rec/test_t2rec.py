import argparse
import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import re

from modeling_t2rec import T2Rec
from prompt_t2rec import GRAPH_TOKEN, BEHAVIOR_TOKEN
from transformers import AutoTokenizer, AutoConfig, LogitsProcessorList
from peft import PeftModel
from generation_trie_t2rec import build_recommendation_logits_processor
from utils_t2rec import (
    parse_global_args,
    parse_test_args,
    parse_dataset_args,
    set_seed,
    ensure_dir,
    load_test_dataset,
    compute_metrics,
)
from collator_t2rec import T2RecTestCollator


def extract_response_text(decoded: str) -> str:
    # 优先用你训练模板里的 "### Response:"
    if "### Response:" in decoded:
        return decoded.split("### Response:")[-1].strip()
    # 兼容你原来的
    if "Response:" in decoded:
        return decoded.split("Response:")[-1].strip()
    return decoded.strip()


def parse_rec_sids(text: str):
    """
    从模型输出中提取推荐列表的 SID 列表
    """
    m = re.search(r"Final Recommendation List:\s*(.*)", text)
    if not m:
        return []
    rec_line = m.group(1).strip()
    # 严格按逗号分
    parts = [p.strip() for p in rec_line.split(",")]
    sids = []
    for p in parts:
        sid = "".join(re.findall(r"<[^>]+>", p))
        if sid:  # 没解析出来就丢掉
            sids.append(sid)
    return sids


def sid_to_item_list(text, token_to_item):
    """
    输入:
      "<a_1><b_2><c_3>, <a_4><b_5><c_6>"
    输出:
      ["item_id_1", "item_id_2"]
    """
    items = []
    for part in text.split(","):
        sid = "".join(re.findall(r"<[^>]+>", part))
        if sid in token_to_item:
            items.append(token_to_item[sid])
        else:
            items.append(sid)  # fallback
    return items


def test(args):
    set_seed(args.seed)
    ensure_dir(os.path.dirname(args.results_file))
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.ckpt_path,
        trust_remote_code=True
    )
    print("=== TOKENIZER CHECK ===", flush=True)
    print("TOKENIZER LOADED FROM:", args.ckpt_path, flush=True)
    print("VOCAB SIZE:", len(tokenizer), flush=True)

    unk = tokenizer.unk_token_id
    print("UNK_TOKEN_ID:", unk, flush=True)

    for tok in ["<a_1>", "<b_1>", "<c_1>", "<d_1>", "<graph_token>", "<behav_token>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        print(f"TOKEN {tok} -> id {tid} (is_unk={tid==unk})", flush=True)
    print("========================", flush=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    test_data = load_test_dataset(args)
    sid_to_itemid = {"".join(tok_list): str(iid) for iid, tok_list in getattr(test_data, "indices", {}).items()}
    token_to_item = getattr(test_data, "token_to_item", {})
    all_candidate_items = getattr(test_data, "all_item_tokens", [])

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
            torch_dtype=torch.bfloat16,  # 使用 bfloat16 避免 Qwen 数值问题
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
            torch_dtype=torch.bfloat16,  # 使用 bfloat16 避免 Qwen 数值问题
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

    all_predictions = []     # List[List[str]]，每条样本的预测SID列表
    all_targets = []         # List[List[str]]，每条样本的多目标SID列表
    all_anomaly_preds = []
    all_anomaly_labels = []
    all_risk_scores = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            targets = batch["targets"]
            user_ids = batch["user_ids"]
            anomaly_labels = batch["anomaly_labels"]
            simple_input_texts = batch["simple_input_texts"]
            deep_input_texts = batch["deep_input_texts"]
            embed_device = model.get_input_embeddings().weight.device

            graph_dev = next(model.temporal_aggregator.parameters()).device
            beh_dev = next(model.behavior_projector.parameters()).device
            graph_tokens = batch["graph_tokens"].to(graph_dev)
            behavior_tokens = batch["behavior_tokens"].to(beh_dev)

            # Stage-1: explicit risk probing.
            risk_logits = model.compute_risk_logit(graph_tokens, behavior_tokens)
            risk_scores = torch.sigmoid(risk_logits)

            route_texts = []
            route_preds = []
            for i in range(risk_scores.size(0)):
                score = float(risk_scores[i].item())
                is_deep = score >= args.risk_threshold
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
                graph_emb = model.graph_projector(graph_emb)
                graph_emb = graph_emb.to(embed_device)
                for i in range(input_ids.size(0)):
                    pos = (input_ids[i] == graph_token_id).nonzero(as_tuple=True)[0]
                    if len(pos) > 0:
                        inputs_embeds[i, pos[0]] = graph_emb[i]

            if behavior_tokens is not None:
                behavior_emb = model.behavior_projector(behavior_tokens)
                behavior_emb = behavior_emb.to(embed_device)
                for i in range(input_ids.size(0)):
                    pos = (input_ids[i] == behavior_token_id).nonzero(as_tuple=True)[0]
                    if len(pos) > 0:
                        inputs_embeds[i, pos[0]] = behavior_emb[i]

            logits_processor = None
            rec_processor = build_recommendation_logits_processor(
                tokenizer=tokenizer,
                candidate_items=all_candidate_items,
                top_k=args.top_k,
            )
            if rec_processor is not None:
                logits_processor = LogitsProcessorList([rec_processor])

            outputs = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=600,
                num_beams=args.num_beams,
                do_sample=False,
                logits_processor=logits_processor,
                num_return_sequences=args.num_beams,
            )

            for i, target in enumerate(targets):
                start_idx = i * args.num_beams
                end_idx = start_idx + args.num_beams
                beam_outputs = outputs[start_idx:end_idx]
                predictions = []
                response = ""

                for output in beam_outputs:
                    decoded = tokenizer.decode(output, skip_special_tokens=True)
                    response = extract_response_text(decoded)

                    pred_sids = parse_rec_sids(response)
                    predictions.append(pred_sids)

                # 取第一个 beam
                all_predictions.append(predictions[0])

                # ===== 改动点：target 解析为多目标SID列表 =====
                # target 在 dataset(test) 里是 response 模板输出（包含 "Final Recommendation List: ..."）
                mt = re.search(r"Final Recommendation List:\s*(.*)", target)
                if mt:
                    # 保留整行，便于 parse_rec_sids 解析
                    target_line_full = "Final Recommendation List: " + mt.group(1).strip()
                else:
                    target_line_full = target.strip().splitlines()[0].strip()

                target_sids = parse_rec_sids(target_line_full)
                all_targets.append(target_sids)

                all_anomaly_preds.append(route_preds[i])
                all_anomaly_labels.append(anomaly_labels[i])
                all_risk_scores.append(float(risk_scores[i].item()))

                if len(all_predictions) <= 3:
                    tqdm.write("=== DEBUG SAMPLE ===")
                    tqdm.write(f"TARGET_SIDS: {all_targets[-1]}")
                    tqdm.write(f"PRED_SIDS: {all_predictions[-1][:10] if isinstance(all_predictions[-1], list) else all_predictions[-1]}")
                    tqdm.write("RAW_RESPONSE:\n" + str(response))

    results = compute_metrics(
        all_predictions,
        all_targets,
        all_anomaly_preds,
        all_anomaly_labels,
        args.metrics
    )
    print("Results:", results)
    
    lens = [len(p) for p in all_predictions]
    print("avg parsed rec len:", sum(lens)/len(lens) if lens else 0)
    print("num zero:", sum(1 for x in lens if x == 0))


    output_data = {
        "args": vars(args),
        "results": results,
        "predictions": [
            {
                "target": t,  # 现在是 list
                "risk_route_pred": ap,
                "risk_score": rs,
                "predictions": p,
                "anomaly_label": al
            }
            for t, p, ap, al, rs in zip(all_targets, all_predictions, all_anomaly_preds, all_anomaly_labels, all_risk_scores)
        ]
    }
    with open(args.results_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {args.results_file}")

    # 一些可选 debug 输出
    if len(all_predictions) > 0:
        last_pred = all_predictions[-1]
        last_tgt = all_targets[-1]
        hit10_any = False
        if isinstance(last_tgt, (list, tuple, set)):
            hit10_any = any(x in set(last_tgt) for x in last_pred[:10])
        print("LAST_TARGET_SIDS:", last_tgt)
        print("LAST_PRED_SIDS:", last_pred[:10])
        print("HIT@10_THIS_SAMPLE(ANY_TARGET):", hit10_any)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T2Rec Test")
    parser = parse_global_args(parser)
    parser = parse_test_args(parser)
    parser = parse_dataset_args(parser)
    args = parser.parse_args()
    test(args)
