import argparse
import os
import sys
import torch

from modeling_t2rec import T2Rec
from prompt_t2rec import GRAPH_TOKEN, BEHAVIOR_TOKEN

import transformers
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoConfig
from utils_t2rec import (
    parse_global_args,
    parse_train_args,
    parse_dataset_args,
    set_seed,
    ensure_dir,
    load_datasets,
)
from collator_t2rec import T2RecCollator


class T2RecTrainer(transformers.Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_branch_log_step = -1
        self._reset_branch_loss_meter()

    def _reset_branch_loss_meter(self):
        self._branch_loss_meter = {
            "rec_loss": {"sum": 0.0, "count": 0},
            "risk_loss": {"sum": 0.0, "count": 0},
            "total_loss": {"sum": 0.0, "count": 0},
        }

    def _update_branch_loss_meter(self, outputs):
        for name in ("rec_loss", "risk_loss", "total_loss"):
            value = getattr(outputs, name, None)
            if value is None:
                continue
            meter = self._branch_loss_meter[name]
            meter["sum"] += float(value.detach().to(torch.float32).item())
            meter["count"] += 1

    def _get_avg_branch_loss_log(self):
        log_data = {}
        for name, meter in self._branch_loss_meter.items():
            if meter["count"] <= 0:
                continue
            log_data["avg_" + name] = meter["sum"] / meter["count"]
        return log_data

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        graph_tokens = inputs.pop("graph_tokens", None)
        behavior_tokens = inputs.pop("behavior_tokens", None)
        risk_labels = inputs.pop("risk_labels", None)
        risk_loss_mask = inputs.pop("risk_loss_mask", None)
        labels = inputs.get("labels")
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
            graph_tokens=graph_tokens,
            behavior_tokens=behavior_tokens,
            risk_labels=risk_labels,
            risk_loss_mask=risk_loss_mask,
        )
        loss = outputs.loss
        current_step = int(getattr(self.state, "global_step", 0))
        log_every = int(getattr(self.args, "logging_steps", 0) or 0)
        should_log = (
            self.model.training
            and log_every > 0
            and current_step > 0
            and current_step % log_every == 0
            and getattr(self, "_last_branch_log_step", -1) != current_step
        )
        if should_log:
            log_data = self._get_avg_branch_loss_log()
            if log_data:
                self.log(log_data)
                self._last_branch_log_step = current_step
                self._reset_branch_loss_meter()
        if self.model.training:
            self._update_branch_loss_meter(outputs)
        return (loss, outputs) if return_outputs else loss


def train(args):
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print(vars(args))
    if ddp:
        device_map = {"": local_rank}
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        model_max_length=args.model_max_length,
        padding_side="right",
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    train_data, valid_data = load_datasets(args)
    risk_supervised_samples = [
        d for d in train_data.inter_data
        if bool(d.get("risk_loss_mask", False)) and d.get("risk_label", -1) in (0, 1)
    ]
    pos_num = sum(1 for d in risk_supervised_samples if d.get("risk_label", -1) == 1)
    neg_num = sum(1 for d in risk_supervised_samples if d.get("risk_label", -1) == 0)
    risk_pos_weight = (neg_num / pos_num) if pos_num > 0 and neg_num > 0 else None
    new_tokens = train_data.get_new_tokens()
    add_num = tokenizer.add_tokens(new_tokens)
    if local_rank == 0:
        print("add {} new tokens.".format(add_num))
        print("data num:", len(train_data))
        print(
            "risk supervised samples:",
            len(risk_supervised_samples),
            "risk_pos:",
            pos_num,
            "risk_neg:",
            neg_num,
        )
        tokenizer.save_pretrained(args.output_dir)
    graph_dim = train_data.graph_dim
    behavior_dim = train_data.behavior_dim
    if local_rank == 0:
        print("graph_dim:", graph_dim, "behavior_dim:", behavior_dim)
    collator = T2RecCollator(args, tokenizer)
    model = T2Rec.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map=device_map,
        graph_dim=graph_dim,
        behavior_dim=behavior_dim,
        probe_dim=args.probe_dim,
        trust_remote_code=True,
    )
    model.set_hyper(args.temperature, args.lambda_anomaly, args.lambda_risk)
    model.set_risk_pos_weight(risk_pos_weight)
    model.resize_token_embeddings(len(tokenizer))
    graph_token_id = tokenizer.convert_tokens_to_ids(GRAPH_TOKEN)
    behavior_token_id = tokenizer.convert_tokens_to_ids(BEHAVIOR_TOKEN)
    model.set_special_token_ids(graph_token_id, behavior_token_id)
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)
    model.enable_input_require_grads()
    target_modules = args.lora_target_modules.split(",")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        modules_to_save=[
            "graph_projector",
            "behavior_projector",
            "temporal_aggregator",
            "risk_behavior_align",
            "risk_graph_align",
            "risk_behavior_norm",
            "risk_graph_norm",
            "risk_mlp",
            "embed_tokens",
            "lm_head"
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    if args.resume_from_checkpoint:
        checkpoint_name = os.path.join(args.resume_from_checkpoint, "adapter_model.bin")
        args.resume_from_checkpoint = False
        if os.path.exists(checkpoint_name):
            if local_rank == 0:
                print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            if local_rank == 0:
                print(f"Checkpoint {checkpoint_name} not found")
    if local_rank == 0:
        if risk_pos_weight is not None:
            print("risk_pos_weight:", risk_pos_weight)
        model.print_trainable_parameters()
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    trainer = T2RecTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
            seed=args.seed,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            fp16=args.fp16,
            bf16=args.bf16,
            logging_steps=args.logging_step,
            optim=args.optim,
            gradient_checkpointing=True,
            evaluation_strategy=args.save_and_eval_strategy,
            save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps,
            save_steps=args.save_and_eval_steps,
            output_dir=args.output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            deepspeed=args.deepspeed if args.deepspeed else None,
            ddp_find_unused_parameters=True if ddp else None,
            eval_delay=1 if args.save_and_eval_strategy == "epoch" else 2000,
            remove_unused_columns=False,
        ),
        tokenizer=tokenizer,
        data_collator=collator,
    )
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T2Rec")
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    args = parser.parse_args()
    train(args)
