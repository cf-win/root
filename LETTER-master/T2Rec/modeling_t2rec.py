import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss


class GraphProjector(nn.Module):
    def __init__(self, in_dim, llm_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, x):
        return self.proj(x)


class BehaviorProjector(nn.Module):
    def __init__(self, in_dim, llm_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, x):
        return self.proj(x)


class TemporalAggregator(nn.Module):
    def __init__(self, d_model, num_layers=2, num_heads=4, max_len=64):
        super().__init__()
        self.d_model = d_model
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = x + self.pos_emb(pos)
        if mask is not None:
            out = self.transformer(x, src_key_padding_mask=mask)
        else:
            out = self.transformer(x)
        out = self.norm(out[:, -1, :])
        return out


class T2Rec(nn.Module):
    def __init__(self, base_model, config, graph_dim=128, behavior_dim=64, probe_dim=64):
        super().__init__()
        self.config = config
        self.base_model = base_model
        self.temperature = 1.0
        self.lambda_risk = 1.0
        self.risk_pos_weight = None
        llm_dim = config.hidden_size
        self.graph_projector = GraphProjector(graph_dim, llm_dim)
        self.behavior_projector = BehaviorProjector(behavior_dim, llm_dim)
        self.temporal_aggregator = TemporalAggregator(graph_dim, num_layers=2, num_heads=4)
        # Risk probe: align behavior/graph to probe_dim, then concat [b'; g'].
        self.risk_behavior_align = nn.Linear(behavior_dim, probe_dim)
        self.risk_graph_align = nn.Linear(graph_dim, probe_dim)
        self.risk_behavior_norm = nn.LayerNorm(probe_dim)
        self.risk_graph_norm = nn.LayerNorm(probe_dim)
        self.risk_mlp = nn.Sequential(
            nn.Linear(probe_dim * 2, probe_dim),
            nn.GELU(),
            nn.Linear(probe_dim, 1),
        )
        self.graph_token_id = None
        self.behavior_token_id = None

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.base_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.base_model.set_output_embeddings(value)

    def resize_token_embeddings(self, new_num_tokens):
        return self.base_model.resize_token_embeddings(new_num_tokens)

    def set_special_token_ids(self, graph_token_id, behavior_token_id):
        self.graph_token_id = graph_token_id
        self.behavior_token_id = behavior_token_id

    def set_hyper(self, temperature, lambda_anomaly=1.0, lambda_risk=1.0):
        self.temperature = temperature
        _ = lambda_anomaly
        self.lambda_risk = lambda_risk

    def set_risk_pos_weight(self, pos_weight: Optional[float] = None):
        self.risk_pos_weight = pos_weight

    def compute_risk_logit(self, graph_tokens, behavior_tokens):
        graph_feat = graph_tokens
        if graph_feat.dim() == 3:
            graph_feat = self.temporal_aggregator(graph_feat)
        graph_feat = graph_feat.to(dtype=self.risk_graph_align.weight.dtype)
        behavior_feat = behavior_tokens.to(dtype=self.risk_behavior_align.weight.dtype)

        graph_aligned = self.risk_graph_norm(self.risk_graph_align(graph_feat))
        behavior_aligned = self.risk_behavior_norm(self.risk_behavior_align(behavior_feat))
        fused = torch.cat([behavior_aligned, graph_aligned], dim=-1)
        return self.risk_mlp(fused).squeeze(-1)

    def compute_risk_score(self, graph_tokens, behavior_tokens):
        return torch.sigmoid(self.compute_risk_logit(graph_tokens, behavior_tokens))

    def compute_loss(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        vocab_size = logits.size(-1)
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        return loss_fct(shift_logits / self.temperature, shift_labels)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        graph_tokens: Optional[torch.FloatTensor] = None,
        behavior_tokens: Optional[torch.FloatTensor] = None,
        risk_labels: Optional[torch.FloatTensor] = None,
        risk_loss_mask: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if graph_tokens is not None and self.graph_token_id is not None:
            if graph_tokens.dim() == 3:
                graph_emb = self.temporal_aggregator(graph_tokens)
            else:
                graph_emb = graph_tokens
            graph_emb = self.graph_projector(graph_emb)
            graph_positions = (input_ids == self.graph_token_id)
            if graph_positions.any():
                inputs_embeds = inputs_embeds.clone()
                for i in range(input_ids.size(0)):
                    pos = (input_ids[i] == self.graph_token_id).nonzero(as_tuple=True)[0]
                    if len(pos) > 0:
                        inputs_embeds[i, pos[0]] = graph_emb[i]
        if behavior_tokens is not None and self.behavior_token_id is not None:
            behavior_emb = self.behavior_projector(behavior_tokens)
            behavior_positions = (input_ids == self.behavior_token_id)
            if behavior_positions.any():
                inputs_embeds = inputs_embeds.clone()
                for i in range(input_ids.size(0)):
                    pos = (input_ids[i] == self.behavior_token_id).nonzero(as_tuple=True)[0]
                    if len(pos) > 0:
                        inputs_embeds[i, pos[0]] = behavior_emb[i]
        outputs = self.base_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits
        rec_loss = None
        risk_loss = None
        total_loss = None
        risk_logit = None
        if labels is not None:
            rec_loss = self.compute_loss(logits, labels)
        if graph_tokens is not None and behavior_tokens is not None:
            risk_logit = self.compute_risk_logit(graph_tokens, behavior_tokens)
            if risk_labels is not None and risk_loss_mask is not None:
                pos_weight = None
                if self.risk_pos_weight is not None:
                    pos_weight = torch.tensor(
                        [self.risk_pos_weight],
                        device=risk_logit.device,
                        dtype=risk_logit.dtype,
                    )
                raw_risk_loss = F.binary_cross_entropy_with_logits(
                    risk_logit,
                    risk_labels.to(device=risk_logit.device, dtype=risk_logit.dtype),
                    pos_weight=pos_weight,
                    reduction="none",
                )
                mask = risk_loss_mask.to(device=risk_logit.device, dtype=risk_logit.dtype)
                masked_loss = raw_risk_loss * mask
                denom = mask.sum().clamp_min(1.0)
                risk_loss = masked_loss.sum() / denom
        if rec_loss is not None and risk_loss is not None:
            total_loss = rec_loss + self.lambda_risk * risk_loss
        elif rec_loss is not None:
            total_loss = rec_loss
        elif risk_loss is not None:
            total_loss = self.lambda_risk * risk_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (total_loss,) + output if total_loss is not None else output
        result = CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        result.rec_loss = rec_loss
        result.risk_loss = risk_loss
        result.total_loss = total_loss
        result.risk_logits = risk_logit
        return result

    def generate(self, **kwargs):
        return self.base_model.generate(**kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        return self.base_model.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, **kwargs)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.base_model.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        self.base_model.enable_input_require_grads()

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, graph_dim=128, behavior_dim=64, probe_dim=64, **kwargs):
        torch_dtype = kwargs.pop("torch_dtype", torch.float16)
        device_map = kwargs.pop("device_map", "auto")
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        model = cls(base_model, config, graph_dim=graph_dim, behavior_dim=behavior_dim, probe_dim=probe_dim)
        if torch_dtype == torch.float16:
            model.graph_projector = model.graph_projector.half()
            model.behavior_projector = model.behavior_projector.half()
            model.temporal_aggregator = model.temporal_aggregator.half()
            model.risk_behavior_align = model.risk_behavior_align.half()
            model.risk_graph_align = model.risk_graph_align.half()
            model.risk_behavior_norm = model.risk_behavior_norm.half()
            model.risk_graph_norm = model.risk_graph_norm.half()
            model.risk_mlp = model.risk_mlp.half()
        device = next(base_model.parameters()).device
        model.graph_projector = model.graph_projector.to(device)
        model.behavior_projector = model.behavior_projector.to(device)
        model.temporal_aggregator = model.temporal_aggregator.to(device)
        model.risk_behavior_align = model.risk_behavior_align.to(device)
        model.risk_graph_align = model.risk_graph_align.to(device)
        model.risk_behavior_norm = model.risk_behavior_norm.to(device)
        model.risk_graph_norm = model.risk_graph_norm.to(device)
        model.risk_mlp = model.risk_mlp.to(device)
        return model

    def print_trainable_parameters(self):
        trainable_params = 0
        all_params = 0
        for _, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.2f}")
